"""Independent integer-kernel, preprocessing and frozen-vector C regressions."""
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

class Workspace(ct.Structure):
    _fields_ = [('a',ct.c_float*64),('b',ct.c_float*64),('fixed',ct.c_int64*64)]

@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    folder=tmp_path_factory.mktemp('c-engine')
    subprocess.run([sys.executable,str(ROOT/'engine-software/build.py'),'--output',str(folder)],check=True)
    library=folder/'engine.so'
    subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-pedantic',
        '-ffp-contract=off','-fno-fast-math','-shared','-fPIC',
        '-I'+str(folder),'-I'+str(ROOT),str(ROOT/'engine-software/ternary_infer.c'),
        '-lm','-o',str(library)],check=True)
    lib=ct.CDLL(str(library))
    lib.tg_packed_dot.argtypes=[ct.POINTER(ct.c_uint8),ct.c_uint16,ct.POINTER(ct.c_int64),ct.c_uint16,ct.POINTER(ct.c_int64)]
    lib.tg_preprocess.argtypes=[ct.POINTER(ct.c_double),ct.POINTER(ct.c_float)]
    lib.tg_infer_preprocessed.argtypes=[ct.POINTER(Workspace),ct.POINTER(ct.c_float),ct.POINTER(ct.c_float)]
    return lib,folder/'ternary_infer'


def test_integer_dot_all_codes_unaligned_rows_and_large_signed_values(engine):
    lib,_=engine
    rng=np.random.default_rng(425)
    for width in (1,3,20,32,63,64):
        codes=rng.integers(0,3,size=width*7,dtype=np.uint8)
        packed=np.zeros((len(codes)+3)//4,dtype=np.uint8)
        for i,code in enumerate(codes):
            packed[i//4] |= int(code) << (2*(i%4))
        x=rng.integers(-(2**40-1),2**40,size=width,dtype=np.int64)
        for row in range(7):
            result=ct.c_int64()
            assert lib.tg_packed_dot(packed.ctypes.data_as(ct.POINTER(ct.c_uint8)),row*width,
                x.ctypes.data_as(ct.POINTER(ct.c_int64)),width,ct.byref(result))==0
            weights=np.choose(codes[row*width:(row+1)*width],[0,1,-1])
            assert result.value==sum(int(a)*int(b) for a,b in zip(x,weights))


def test_integer_dot_rejects_reserved_code_and_oversized_input(engine):
    lib,_=engine
    output=ct.c_int64()
    assert lib.tg_packed_dot((ct.c_uint8*1)(3),0,(ct.c_int64*1)(4),1,ct.byref(output))==-1
    assert lib.tg_packed_dot((ct.c_uint8*1)(0),0,(ct.c_int64*1)(0),65,ct.byref(output))==-1


def test_preprocessing_signed_values_and_nonfinite_rejection(engine):
    lib,_=engine
    cfg=json.loads((ROOT/'model/feature_config.json').read_text())
    selected=cfg['selected_features']
    raw=np.zeros(115,dtype=np.float64)
    raw[selected]=np.array([0.,-0.,1e-30,-1e-30,1.,-1.,1e100,-1e100,1e-6,-1e-6]*2)
    result=(ct.c_float*20)()
    assert lib.tg_preprocess(raw.ctypes.data_as(ct.POINTER(ct.c_double)),result)==0
    values=raw[selected]
    expected=((np.sign(values)*np.log1p(np.abs(values))-cfg['scaler']['mean'])/cfg['scaler']['scale']).astype(np.float32)
    np.testing.assert_allclose(np.asarray(result),expected,atol=1e-6,rtol=1e-6)
    for bad in (float('nan'),float('inf'),float('-inf')):
        raw[selected[0]]=bad
        assert lib.tg_preprocess(raw.ctypes.data_as(ct.POINTER(ct.c_double)),result)==-1


def test_frozen_golden_vectors_match_logits_and_predictions(engine):
    lib,_=engine
    with np.load(ROOT/'model/golden_vectors.npz',allow_pickle=False) as vectors:
        actual=[]
        workspace=Workspace()
        for row in vectors['inputs']:
            output=(ct.c_float*11)()
            assert lib.tg_infer_preprocessed(ct.byref(workspace),row.ctypes.data_as(ct.POINTER(ct.c_float)),output)==0
            actual.append(list(output))
        actual=np.array(actual,dtype=np.float32)
        np.testing.assert_allclose(actual,vectors['logits'],atol=2e-5,rtol=2e-5)
        np.testing.assert_array_equal(actual.argmax(1),vectors['logits'].argmax(1))


def test_cli_rejects_partial_raw_record(engine,tmp_path):
    _,executable=engine
    partial=tmp_path/'partial.bin'
    partial.write_bytes(b'\x00'*919)
    result=subprocess.run([str(executable),'--run',str(partial),str(tmp_path/'out.bin')],capture_output=True,text=True)
    assert result.returncode==1 and 'Invalid input at row 0' in result.stderr
