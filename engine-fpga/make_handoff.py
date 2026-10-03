#!/usr/bin/env python3
"""Portable code + guarded vectors + oracle; no raw dataset required on laptop."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def package(root,vectors,target):
    root=Path(root).resolve();vectors=Path(vectors).resolve();target=Path(target).resolve()
    meta=json.loads((vectors/'vectors.json').read_text())
    if meta['vectors']!='test' or meta['samples']!=69040:raise ValueError('Full frozen test vectors required')
    files={}
    for pattern in ['rtl/*.sv','tb/*.sv','tb/*.c','tb/*.cpp','scripts/*.tcl','constraints/*.xdc','*.py','README.md']:
        for p in (root/'engine-fpga').glob(pattern):files['engine-fpga/'+str(p.relative_to(root/'engine-fpga'))]=p.read_bytes()
    for name in ['inputs.mem','model_rom.sv','reference.npz','vectors.json']:
        files['vectors/'+name]=(vectors/name).read_bytes()
    for name,key in [('inputs.mem','input_sha256'),('model_rom.sv','rom_sha256'),('reference.npz','reference_sha256')]:
        if hashlib.sha256(files['vectors/'+name]).hexdigest()!=meta[key]:raise ValueError('Generated artifact changed: '+name)
    active=json.loads((root/'model/active_model.json').read_text())
    for name in ['model/active_model.json','model/export_manifest.json',active['header'],active['preprocessing'],active['checkpoint']]:
        files[name]=(root/name).read_bytes()
    for name,key in [(active['checkpoint'],'checkpoint_sha256'),(active['header'],'header_sha256')]:
        if hashlib.sha256(files[name]).hexdigest()!=meta['active_hashes'][key]:raise ValueError('Active artifact changed: '+name)
    files['PHASE7_MANUAL.md']=(root/'docs/phase7_manual_vivado.md').read_bytes()
    files['evidence/local-parity.json']=(vectors/'verilator-parity.json').read_bytes()
    files['evidence/local-commands.log']=(vectors/'compiled/commands.log').read_bytes()
    manifest=dict(active_hashes=meta['active_hashes'],files={name:hashlib.sha256(content).hexdigest() for name,content in sorted(files.items())})
    files['handoff_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in files.items():archive.writestr(name,content)
    print(json.dumps(dict(package=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(files)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vectors',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();package(Path(__file__).resolve().parents[1],a.vectors,a.output)
