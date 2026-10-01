#!/usr/bin/env python3
"""Build C against verified frozen weights and JSON-derived preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.artifacts import json_hash


def build(output, cc='cc', sanitize=False):
    output = Path(output).resolve()
    active = json.loads((ROOT/'model/active_model.json').read_text())
    if active['header'] != 'model_weights.h':
        raise ValueError('Engine includes the exact root header')
    if active['architecture'] != dict(input_dim=20, hidden_dims=[64,32], output_dim=11, use_rmsnorm=True, ternary_output=True):
        raise ValueError('Engine requires the frozen 20/64/32/11 architecture')
    manifest = json.loads((ROOT/active['manifest']).read_text())
    config = json.loads((ROOT/active['preprocessing']).read_text())
    for key in ('checkpoint', 'header'):
        if hashlib.sha256((ROOT/active[key]).read_bytes()).hexdigest() != manifest[key+'_sha256']:
            raise ValueError(key+' hash mismatch')
    if json_hash(config) != manifest['preprocessing_sha256']:
        raise ValueError('Preprocessing hash mismatch')
    if config['transform'] != 'signed_log1p' or len(config['selected_features']) != 20:
        raise ValueError('Expected 20 signed-log1p features')
    output.mkdir(parents=True, exist_ok=True)
    arrays = [('uint8_t', 'tg_feature_indices', config['selected_features']),
              ('double', 'tg_feature_mean', config['scaler']['mean']),
              ('double', 'tg_feature_scale', config['scaler']['scale'])]
    header = ['/* Generated from frozen model/feature_config.json. Do not edit. */',
              '#ifndef TG_PREPROCESSING_H', '#define TG_PREPROCESSING_H',
              '#include "model_weights.h"']
    for typ, name, values in arrays:
        tokens = [str(v) if typ == 'uint8_t' else float(v).hex() for v in values]
        header.append(f'static const {typ} TGPROGMEM {name}[] = {{{", ".join(tokens)}}};')
    header.append('#endif\n')
    (output/'preprocessing.h').write_text('\n'.join(header))
    flags = ['-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-ffp-contract=off', '-fno-fast-math', '-fstack-usage', '-I'+str(output), '-I'+str(ROOT)]
    if sanitize:
        flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    command = [cc, *flags, str(ROOT/'engine-software/ternary_infer.c'), str(ROOT/'engine-software/runner.c'), '-lm', '-o', str(output/'ternary_infer')]
    print('$ '+shlex.join(command), flush=True)
    subprocess.run(command, check=True)
    return output/'ternary_infer'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'engine-software/build')
    parser.add_argument('--cc', default='cc')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    build(args.output, args.cc, args.sanitize)
