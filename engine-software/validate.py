#!/usr/bin/env python3
"""Replay every frozen test identity from raw CSV through C and PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.artifacts import load_ternary_bundle, json_hash
from model.data_pipeline import get_feature_names
from model.validate_export import frozen_test_loader, verify_active
from build import build


def validate(output, data_dir, sanitize=False):
    torch.set_num_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    active = json.loads((ROOT/'model/active_model.json').read_text())
    # Guard the exact active artifacts before compiling or running the engine.
    verification = verify_active(ROOT/'model/active_model.json', data_dir)
    print('ACTIVE ARTIFACT VERIFICATION '+json.dumps(verification), flush=True)
    executable = build(output/'build', sanitize=sanitize)
    checkpoint = ROOT/active['checkpoint']
    model, bundle = load_ternary_bundle(checkpoint)
    manifest = json.loads((checkpoint.parent/'data_manifest.json').read_text())
    if json_hash(manifest) != bundle['provenance']['data_manifest_sha256']:
        raise ValueError('Data manifest changed')
    with np.load(checkpoint.parent/'split_rows.npz', allow_pickle=False) as ids:
        mask = ids['split'] == 'test'
        sources, rows = ids['source_id'][mask], ids['row'][mask]
    raw, labels, metadata = [], [], []
    for source_id in np.unique(sources):
        source = manifest['sources'][int(source_id)]
        path = Path(data_dir)/source['path']
        with path.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != source['sha256']:
                raise ValueError('Raw source changed: '+str(path))
        peek = pd.read_csv(path, nrows=1, header=None)
        numeric = pd.to_numeric(peek.iloc[0], errors='coerce').notna().all()
        frame = pd.read_csv(path, header=None if numeric else 0)
        if frame.shape[1] != 115 or (not numeric and list(frame.columns) != get_feature_names()):
            raise ValueError('CSV schema changed')
        take = rows[sources == source_id]
        raw.append(frame.iloc[take].to_numpy(dtype=np.float64))
        labels.extend([source['label']]*len(take))
        metadata.append(pd.DataFrame(dict(source_id=int(source_id), device=source['device'], row=take)))
    identities = pd.concat(metadata, ignore_index=True)
    digest = hashlib.sha256(identities.to_csv(index=False).encode()).hexdigest()
    if digest != manifest['partitions']['test']['row_identity_sha256']:
        raise ValueError('Test row identity mismatch')
    raw = np.concatenate(raw)
    labels = np.asarray(labels)
    if len(raw) != 69040:
        raise ValueError('Expected all 69,040 frozen test rows')
    raw_path = output/'test_raw_f64.bin'
    raw.tofile(raw_path)
    raw[np.linspace(0, len(raw)-1, 256, dtype=int)].tofile(output/'benchmark_raw_f64.bin')
    # Separate C preprocessing output proves we did not feed Python-scaled inputs.
    for mode, name in [('--preprocess', 'preprocessed_f32.bin'), ('--run', 'logits_f32.bin')]:
        command = [str(executable), mode, str(raw_path), str(output/name)]
        print('$ '+' '.join(command), flush=True)
        subprocess.run(command, check=True)
    loader = frozen_test_loader(checkpoint, data_dir)
    x, y = loader.dataset.tensors
    np.testing.assert_array_equal(labels, y.numpy())
    expected = []
    with torch.no_grad():
        for batch, _ in loader:
            expected.append(model(batch).numpy())
    expected = np.concatenate(expected)
    c_pre = np.fromfile(output/'preprocessed_f32.bin', dtype=np.float32).reshape(-1,20)
    c_logits = np.fromfile(output/'logits_f32.bin', dtype=np.float32).reshape(-1,11)
    np.testing.assert_allclose(c_pre, x.numpy(), atol=2e-6, rtol=2e-6)
    predicted, reference = c_logits.argmax(1), expected.argmax(1)
    mismatch = np.flatnonzero(predicted != reference)
    details = identities.iloc[mismatch].copy()
    details['test_index'] = mismatch
    details['c_prediction'] = predicted[mismatch]
    details['torch_prediction'] = reference[mismatch]
    details.to_csv(output/'mismatches.csv', index=False)
    tcp_id = next(int(k) for k,v in bundle['preprocessing']['labels'].items() if v == 'bashlite_tcp')
    tcp = labels == tcp_id
    report = dict(samples=len(raw), prediction_matches=int(len(raw)-len(mismatch)),
        prediction_disagreements=int(len(mismatch)),
        max_abs_preprocessing_error=float(np.max(np.abs(c_pre-x.numpy()))),
        max_abs_logit_error=float(np.max(np.abs(c_logits-expected))),
        logits_within_export_tolerance=bool(np.allclose(c_logits, expected, atol=2e-5, rtol=2e-5)),
        c_accuracy=float(np.mean(predicted==labels)), torch_accuracy=float(np.mean(reference==labels)),
        c_macro_f1=float(f1_score(labels,predicted,average='macro')),
        torch_macro_f1=float(f1_score(labels,reference,average='macro')),
        bashlite_tcp=dict(total=int(tcp.sum()), c_true_positives=int(np.sum(predicted[tcp]==tcp_id)),
                         torch_true_positives=int(np.sum(reference[tcp]==tcp_id))),
        row_identity_sha256=digest, frozen_artifact_hashes=verification['hashes'])
    (output/'parity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    np.testing.assert_allclose(c_logits, expected, atol=2e-5, rtol=2e-5)
    if len(mismatch) or report['bashlite_tcp'] != dict(total=5555,c_true_positives=1,torch_true_positives=1):
        raise AssertionError('Frozen model prediction/TCP parity failed')
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'engine-software/build/validation')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/raw/nbaiot')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    validate(args.output, args.data_dir, args.sanitize)
