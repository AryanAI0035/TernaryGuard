"""Host-only Nano source and serial protocol tests; not hardware certification."""
import binascii
import ctypes as ct
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('nano_serial',ROOT/'engine-arduino/serial_validate.py')
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)

class Workspace(ct.Structure):
    _fields_=[('a',ct.c_float*64),('b',ct.c_float*64),('fixed',ct.c_int64*64)]

@pytest.fixture(scope='module')
def nano(tmp_path_factory):
    folder=tmp_path_factory.mktemp('nano-port')
    subprocess.run([sys.executable,str(ROOT/'engine-arduino/generate_config.py'),'--output',str(folder)],check=True)
    output=folder/'nano.so'
    subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-ffp-contract=off','-fno-fast-math','-shared','-fPIC',
        '-I'+str(folder),'-I'+str(ROOT/'engine-arduino/include'),str(ROOT/'engine-arduino/src/ternary_avr.c'),'-lm','-o',str(output)],check=True)
    lib=ct.CDLL(str(output));fp=ct.POINTER(ct.c_float)
    lib.nano_preprocess.argtypes=[fp]
    lib.tg_infer_preprocessed.argtypes=[ct.POINTER(Workspace),fp,fp]
    return lib


def test_nano_output_alias_matches_frozen_golden_vectors(nano):
    # Firmware receives directly into work.a and returns logits in work.b.
    work=Workspace()
    with np.load(ROOT/'model/golden_vectors.npz') as z:
        results=[]
        for row in z['inputs']:
            for i,value in enumerate(row): work.a[i]=value
            assert nano.tg_infer_preprocessed(ct.byref(work),work.a,work.b)==0
            results.append(list(work.b)[:11])
        np.testing.assert_allclose(results,z['logits'],atol=2e-5,rtol=2e-5)
        np.testing.assert_array_equal(np.argmax(results,axis=1),z['logits'].argmax(1))


def test_nano_float32_preprocessing_signed_tiny_values_and_nonfinite(nano):
    import json
    cfg=json.loads((ROOT/'model/feature_config.json').read_text())
    values=np.array([0.,-0.,1e-30,-1e-30,1.,-1.,1e30,-1e30,1e-6,-1e-6]*2,dtype=np.float32)
    original=values.copy()
    assert nano.nano_preprocess(values.ctypes.data_as(ct.POINTER(ct.c_float)))==0
    reference=((np.sign(original)*np.log1p(np.abs(original))-np.asarray(cfg['scaler']['mean'],dtype=np.float32))/np.asarray(cfg['scaler']['scale'],dtype=np.float32))
    np.testing.assert_allclose(values,reference,atol=2e-6,rtol=2e-6)
    for invalid in (float('nan'),float('inf'),float('-inf')):
        values[0]=invalid
        assert nano.nano_preprocess(values.ctypes.data_as(ct.POINTER(ct.c_float)))==-1


def test_serial_protocol_crc_sequence_status_and_lengths():
    request=protocol.packet(42,[0.]*20)
    assert len(request)==90 and request[:8]==b'TG\x01\x01*\x00\x00\x00'
    assert binascii.crc_hqx(request[:-2],0xffff)==struct.unpack('<H',request[-2:])[0]
    def frame(status=0):
        body=struct.pack('<2sBBIBIH11f',b'TR',1,status,42,9,12345,256,*([0.]*11))
        return body+struct.pack('<H',binascii.crc_hqx(body,0xffff))
    result=protocol.response(frame(),42)
    assert result['prediction']==9 and result['inference_us']==12345 and result['untouched_gap_bytes']==256
    with pytest.raises(ValueError,match='identity'): protocol.response(frame(),43)
    with pytest.raises(ValueError,match='status'): protocol.response(frame(3),42)
    corrupt=bytearray(frame());corrupt[20]^=1
    with pytest.raises(ValueError,match='CRC'): protocol.response(corrupt,42)
    with pytest.raises(ValueError,match='Truncated'): protocol.response(frame()[:-1],42)
