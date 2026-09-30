#!/usr/bin/env python3
"""Rebuild and validate an export against frozen, source-verified test row identities.

Usage: python3 model/validate_export.py --checkpoint checkpoints/RUN/ternary_2bit.pt
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.artifacts import export_bundle, load_ternary_bundle, json_hash
from model.data_pipeline import get_feature_names, transform_features
from model.train import verify_export


def frozen_test_loader(checkpoint, data_dir, batch_size=1024, partition="test"):
    if partition not in ("train", "val", "test"):
        raise ValueError("Unknown frozen partition")
    _, bundle = load_ternary_bundle(checkpoint)
    run = Path(checkpoint).parent
    manifest = json.loads((run/'data_manifest.json').read_text())
    if json_hash(manifest) != bundle['provenance']['data_manifest_sha256']:
        raise ValueError('Data manifest fingerprint mismatch')
    identities = np.load(run/'split_rows.npz', allow_pickle=False)
    mask = identities['split'] == partition
    source_ids, rows = identities['source_id'][mask], identities['row'][mask]
    config = bundle['preprocessing']
    features, labels, metadata = [], [], []
    for source_id in np.unique(source_ids):
        source = manifest['sources'][int(source_id)]
        path = Path(data_dir)/source['path']
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):
                digest.update(chunk)
        if digest.hexdigest() != source['sha256']:
            raise ValueError(f'Dataset source changed: {path}')
        peek = pd.read_csv(path,nrows=1,header=None)
        numeric = pd.to_numeric(peek.iloc[0],errors='coerce').notna().all()
        frame = pd.read_csv(path,header=None if numeric else 0)
        if not numeric and list(frame.columns) != get_feature_names():
            raise ValueError('Dataset schema changed')
        take = rows[source_ids == source_id]
        selected = frame.iloc[take,config['selected_features']].to_numpy()
        selected = transform_features(selected, config.get('transform', 'identity'))
        features.append(((selected-np.array(config['scaler']['mean'])) /
                          np.array(config['scaler']['scale'])).astype(np.float32))
        labels.extend([source['label']]*len(take))
        metadata.append(pd.DataFrame(dict(source_id=int(source_id),device=source['device'],row=take)))
    identity_hash = hashlib.sha256(pd.concat(metadata,ignore_index=True).to_csv(index=False).encode()).hexdigest()
    if identity_hash != manifest['partitions'][partition]['row_identity_sha256']:
        raise ValueError('Frozen test row identities changed')
    return DataLoader(TensorDataset(torch.from_numpy(np.concatenate(features)),
                                    torch.tensor(labels,dtype=torch.long)),batch_size=batch_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/nbaiot')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=args.output_dir or args.checkpoint.parent/'export'
    loader=frozen_test_loader(args.checkpoint,args.data_dir)
    model,_=export_bundle(args.checkpoint,output)
    print(json.dumps(verify_export(model,output/'model_weights.h',loader,output),indent=2))


if __name__=='__main__':
    main()
