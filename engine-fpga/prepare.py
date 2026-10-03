#!/usr/bin/env python3
"""Generate ROM and frozen float32 inputs from guarded active artifacts.
FPGA boundary is the existing 20 selected, signed-log1p/StandardScaler inputs.
Preprocessing remains on the host; there is no FPGA log-transform claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch


def prepare(root, output, vectors):
    root=Path(root).resolve(); output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root))
    from model.artifacts import HeaderReference, json_hash, load_ternary_bundle
    from model.validate_export import frozen_test_loader, verify_active
    torch.set_num_threads(1)
    active=json.loads((root/'model/active_model.json').read_text())
    checkpoint=root/active['checkpoint'];header=root/active['header']
    manifest=json.loads((root/active['manifest']).read_text())
    config=json.loads((root/active['preprocessing']).read_text())
    hashes=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),preprocessing_sha256=json_hash(config))
    for key,value in hashes.items():
        if value!=manifest[key]:raise ValueError('Active artifact mismatch: '+key)
    ref=HeaderReference(header)
    if ref.dims.tolist()!=[20,64,32,11] or not ref.norm:raise ValueError('Unsupported architecture')
    text=header.read_text()
    packed=[]
    for layer in range(3):
        body=re.search(r'tg_weights_'+str(layer)+r'\[\]\s*=\s*\{(.*?)\};',text,re.S).group(1)
        packed.extend(int(t.strip(),0) for t in body.split(',') if t.strip())
    if any((b>>(2*k))&3==3 for b in packed for k in range(4)):raise ValueError('Reserved code')
    bias=np.concatenate([layer[1] for layer in ref.layers]);gamma=np.concatenate([layer[3] for layer in ref.layers[:2]])
    def case_function(name,values,width):
        lines=[f'    reg [{width-1}:0] {name}_mem[0:{len(values)-1}];','    initial begin']
        for i,value in enumerate(values):lines.append(f"        {name}_mem[{i}]={width}'h{int(value):0{width//4}x};")
        lines.extend(['    end',f'    function automatic [{width-1}:0] {name}(input integer address);',
            f"        {name}=(address>=0 && address<{len(values)}) ? {name}_mem[address] : {width}'b0;",'    endfunction'])
        return '\n'.join(lines)
    rom='''`timescale 1ns/1ps
// Generated from the EXACT active header; do not hand-edit.
module model_rom (
    input wire [1:0] layer,
    input wire [6:0] column,base_row,row,norm_index,
    output reg [15:0] codes,
    output reg [31:0] bias,gamma,scale,epsilon
);
'''
    rom+=case_function('weight_byte',packed,8)+'\n'
    rom+=case_function('bias_word',bias.view(np.uint32),32)+'\n'
    rom+=case_function('gamma_word',gamma.view(np.uint32),32)+'\n'
    rom+='''    integer lane,r,address;
    reg [7:0] packed_byte;
    always @* begin
        codes=0; bias=0; gamma=0; scale=0; epsilon=0; packed_byte=0;
        r=0; address=0;
        for (lane=0;lane<8;lane=lane+1) begin
            r=integer'(base_row)+lane;
            case (layer)
                0: address=(r<<2)+r+(integer'(column)>>2);
                1: address=320+(r<<4)+(integer'(column)>>2);
                default: address=832+(r<<3)+(integer'(column)>>2);
            endcase
            packed_byte=weight_byte(address);
            if (r<(layer==0 ? 64 : layer==1 ? 32 : 11))
                codes[lane*2+:2]=2'(packed_byte>>{column[1:0],1'b0});
        end
        case (layer)
'''
    for i,(_,_,scale,gam,eps) in enumerate(ref.layers):
        base=[0,64,96][i]
        rom+=f'            {i}: begin bias=bias_word({base}+integer\'(row)); '
        rom+=f"scale=32'h{int(np.float32(scale).view(np.uint32)):08x}; "
        if gam is not None:
            rom+=f'gamma=gamma_word({0 if i==0 else 20}+integer\'(norm_index)); '
            rom+=f"epsilon=32'h{int(np.float32(eps).view(np.uint32)):08x}; "
        rom+='end\n'
    rom+='''            default: begin bias=0; gamma=0; scale=0; epsilon=0; end
        endcase
    end
endmodule
'''
    (output/'model_rom.sv').write_text(rom)
    model,_=load_ternary_bundle(checkpoint)
    if vectors=='test':
        verified=verify_active(root/'model/active_model.json',root/'data/raw/nbaiot')
        print('ACTIVE ARTIFACT VERIFICATION '+json.dumps(verified),flush=True)
        loader=frozen_test_loader(checkpoint,root/'data/raw/nbaiot')
        inputs,labels=loader.dataset.tensors
        inputs=inputs.numpy();labels=labels.numpy()
    else:
        with np.load(root/active['golden_vectors']) as data:
            inputs=data['inputs'].copy();labels=data['labels'].copy()
    logits=[]
    with torch.no_grad():
        for start in range(0,len(inputs),1024):logits.append(model(torch.from_numpy(inputs[start:start+1024])).numpy())
    expected=np.concatenate(logits)
    if not np.isfinite(inputs).all():raise ValueError('Nonfinite input')
    bits=inputs.view(np.uint32)
    subnormal=((bits&0x7f800000)==0)&((bits&0x007fffff)!=0)
    if subnormal.any():raise ValueError('Unsupported subnormal input')
    (output/'inputs.mem').write_text(''.join(f'{int(v):08x}\n' for v in bits.flatten()))
    np.savez(output/'reference.npz',inputs=inputs,labels=labels,logits=expected)
    metadata=dict(scope='HOST-preprocessed inputs; frozen checkpoint oracle; no physical FPGA',
        vectors=vectors,samples=len(inputs),active_hashes=hashes,
        rom_sha256=hashlib.sha256(rom.encode()).hexdigest(),
        input_sha256=hashlib.sha256((output/'inputs.mem').read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256((output/'reference.npz').read_bytes()).hexdigest(),
        test_row_identity_sha256=manifest.get('provenance',{}).get('data_manifest_sha256'),
        fp_contract='binary32 RNE, separate operations; reject subnormal/nonfinite inputs and intermediates')
    # Correct row identity comes from the guarded checkpoint's data manifest.
    if vectors=='test':
        data_manifest=json.loads((checkpoint.parent/'data_manifest.json').read_text())
        metadata['test_row_identity_sha256']=data_manifest['partitions']['test']['row_identity_sha256']
    else:metadata.pop('test_row_identity_sha256')
    (output/'vectors.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)
    return metadata

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vectors',choices=['golden','test'],default='test')
    a=p.parse_args();prepare(a.project_root,a.output,a.vectors)
