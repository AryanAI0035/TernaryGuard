#!/usr/bin/env python3
"""Full host float32-port probe and reproducible hardware test packet selection.
This does not claim AVR-libc parity or physical hardware measurements.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'engine-software'))
sys.path.insert(0,str(ROOT))
from validate import validate
from model.validate_export import frozen_test_loader

class Workspace(ct.Structure):
    _fields_=[('a',ct.c_float*64),('b',ct.c_float*64),('fixed',ct.c_int64*64)]

def prepare(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    reference=output/'reference'
    validate(reference,ROOT/'data/raw/nbaiot')
    subprocess.run([sys.executable,str(ROOT/'engine-arduino/generate_config.py'),'--output',str(output/'generated')],check=True)
    libpath=output/'nano-host.so'
    command=['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-ffp-contract=off','-fno-fast-math','-shared','-fPIC',
             '-I'+str(ROOT/'engine-arduino/include'),'-I'+str(output/'generated'),str(ROOT/'engine-arduino/src/ternary_avr.c'),'-lm','-o',str(libpath)]
    print('$ '+' '.join(command),flush=True);subprocess.run(command,check=True)
    lib=ct.CDLL(str(libpath))
    fp=ct.POINTER(ct.c_float)
    lib.nano_preprocess.argtypes=[fp]
    lib.tg_infer_preprocessed.argtypes=[ct.POINTER(Workspace),fp,fp]
    cfg=json.loads((ROOT/'model/feature_config.json').read_text())
    raw=np.fromfile(reference/'test_raw_f64.bin',dtype=np.float64).reshape(-1,115)
    inputs=np.ascontiguousarray(raw[:,cfg['selected_features']],dtype=np.float32)
    transformed=inputs.copy();actual=np.empty((len(raw),11),dtype=np.float32)
    workspace=Workspace()
    for i,row in enumerate(transformed):
        assert row.flags.c_contiguous and row.strides==(4,)
        if lib.nano_preprocess(row.ctypes.data_as(fp)) or lib.tg_infer_preprocessed(ct.byref(workspace),row.ctypes.data_as(fp),actual[i].ctypes.data_as(fp)):
            raise ValueError('C port rejected test row '+str(i))
    expected=np.fromfile(reference/'logits_f32.bin',dtype=np.float32).reshape(-1,11)
    x,y=frozen_test_loader(ROOT/'checkpoints/phase3_final_seed42/ternary_2bit.pt',ROOT/'data/raw/nbaiot').dataset.tensors
    labels=y.numpy();predicted=actual.argmax(1);ref=expected.argmax(1)
    bad=np.flatnonzero(predicted!=ref)
    tcp=np.flatnonzero(labels==9)
    tcp_hits=tcp[ref[tcp]==9]
    if len(tcp)!=5555 or len(tcp_hits)!=1: raise ValueError('Frozen TCP reference changed')
    rng=np.random.default_rng(42)
    subset=np.concatenate([rng.choice(np.flatnonzero(labels==label),30,replace=False) for label in range(11)])
    if tcp_hits[0] not in subset: subset[270]=tcp_hits[0]
    np.savez(output/'serial_cases.npz',raw_selected=inputs,preprocessed=x.numpy(),labels=labels,
             expected_logits=expected,expected_predictions=ref,subset=subset,tcp=tcp)
    report=dict(scope='HOST build of AVR source using host libm; NOT an AVR simulator or physical board',
        preprocessing='on-device float32 candidate; hardware acceptance pending',samples=len(raw),
        prediction_disagreements=len(bad),mismatch_indices=bad.tolist(),
        max_preprocessing_error=float(np.max(np.abs(transformed-x.numpy()))),
        max_logit_error=float(np.max(np.abs(actual-expected))),
        tcp_total=len(tcp),tcp_true_positives=int(np.sum(predicted[tcp]==9)),
        proposed_stratified_rows=len(subset),proposed_tcp_rows=len(tcp),sole_tcp_hit_test_index=int(tcp_hits[0]))
    print(json.dumps(report,indent=2),flush=True)
    (output/'host_precision.json').write_text(json.dumps(report,indent=2)+'\n')
    if len(bad) or np.sum(predicted[tcp]==9)!=1: raise AssertionError('Float32 port host prediction parity failed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output)
