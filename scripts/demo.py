#!/usr/bin/env python3
"""Replay bundled preprocessed engineering vectors; no raw-data accuracy claim."""
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.artifacts import HeaderReference, json_hash, load_ternary_bundle


class Workspace(ct.Structure):
    _fields_ = [('a', ct.c_float * 64), ('b', ct.c_float * 64),
                ('fixed', ct.c_int64 * 64)]


def main():
    torch.set_num_threads(1)
    active = json.loads((ROOT / 'model/active_model.json').read_text())
    manifest = json.loads((ROOT / active['manifest']).read_text())
    for kind in ('checkpoint', 'header'):
        actual = hashlib.sha256((ROOT / active[kind]).read_bytes()).hexdigest()
        if actual != manifest[kind + '_sha256']:
            raise ValueError(kind + ' identity mismatch')
    config = json.loads((ROOT / active['preprocessing']).read_text())
    if json_hash(config) != manifest['preprocessing_sha256']:
        raise ValueError('Preprocessing identity mismatch')
    model, bundle = load_ternary_bundle(ROOT / active['checkpoint'])
    if bundle['preprocessing_sha256'] != json_hash(config):
        raise ValueError('Checkpoint preprocessing mismatch')
    if active['architecture'] != bundle['architecture']:
        raise ValueError('Architecture mismatch')
    header = HeaderReference(ROOT / active['header'])
    spec = importlib.util.spec_from_file_location('tg_c_build', ROOT / 'engine-software/build.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    with np.load(ROOT / active['golden_vectors'], allow_pickle=False) as z:
        inputs = np.ascontiguousarray(z['inputs'], dtype=np.float32)
        recorded = z['logits'].copy()
    with torch.no_grad():
        expected = model(torch.from_numpy(inputs)).numpy()
    header_logits = header(inputs)
    with tempfile.TemporaryDirectory(prefix='ternaryguard-demo-') as temporary:
        output = Path(temporary)
        builder.build(output)
        library = output / 'engine.so'
        subprocess.run(['cc', '-std=c11', '-O2', '-ffp-contract=off', '-fno-fast-math',
                        '-shared', '-fPIC', '-I' + str(output), '-I' + str(ROOT),
                        str(ROOT / 'engine-software/ternary_infer.c'), '-lm', '-o', str(library)], check=True)
        engine = ct.CDLL(str(library))
        engine.tg_infer_preprocessed.argtypes = [ct.POINTER(Workspace),
                                               ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)]
        engine.tg_infer_preprocessed.restype = ct.c_int
        work = Workspace()
        c_logits = []
        for row in inputs:
            logits = (ct.c_float * 11)()
            status = engine.tg_infer_preprocessed(ct.byref(work),
                row.ctypes.data_as(ct.POINTER(ct.c_float)), logits)
            if status:
                raise RuntimeError('C inference returned status ' + str(status))
            c_logits.append(list(logits))
    c_logits = np.asarray(c_logits, dtype=np.float32)
    report = dict(scope='264 bundled preprocessed engineering vectors; not dataset accuracy or live traffic',
                  samples=len(inputs), checkpoint_sha256=manifest['checkpoint_sha256'],
                  atol=2e-5, rtol=2e-5)
    for name, actual in [('recorded_golden', recorded), ('existing_header', header_logits), ('c_engine', c_logits)]:
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        disagreements = int(np.count_nonzero(actual.argmax(1) != expected.argmax(1)))
        if disagreements:
            raise AssertionError(name + ' prediction disagreements: ' + str(disagreements))
        report[name] = dict(prediction_matches=len(inputs), prediction_disagreements=disagreements,
                            max_abs_logit_error=float(np.max(np.abs(actual - expected))))
    report['known_dataset_limitation'] = 'BASHLITE TCP recall 1/5555 (0.018%); not re-evaluated by this demo'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
