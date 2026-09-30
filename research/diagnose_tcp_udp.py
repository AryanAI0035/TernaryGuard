#!/usr/bin/env python3
"""Training/validation-only diagnostic of GAFGYT TCP versus UDP separability.

This classifier is a diagnostic, not a deployment model. Device 9 is never read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.data_pipeline import get_feature_names


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,default=Path('data/raw/nbaiot'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    frames,labels={},{}
    for split,devices in [('train',range(1,8)),('val',[8])]:
        xs,ys=[],[]
        for device in devices:
            for label,attack in enumerate(['tcp','udp']):
                path=args.data_dir/f'{device}.gafgyt.{attack}.csv'
                peek=pd.read_csv(path,nrows=1,header=None)
                numeric=pd.to_numeric(peek.iloc[0],errors='coerce').notna().all()
                x=pd.read_csv(path,header=None if numeric else 0).sample(n=1000,random_state=42)
                xs.append(x.to_numpy(dtype=float));ys.extend([label]*len(x))
        frames[split]=np.concatenate(xs);labels[split]=np.array(ys)
    report=dict(seed=42,train_devices=list(range(1,8)),validation_device=8,
                rows_per_device_per_class=1000,config=str(args.config),experiments={})
    names=get_feature_names()
    for kind,indices in [('selected',cfg['selected_features']),('all',list(range(115)))]:
        model=RandomForestClassifier(n_estimators=40,max_depth=12,random_state=42,n_jobs=1)
        model.fit(frames['train'][:,indices],labels['train'])
        report['experiments'][kind]=dict(
            training_accuracy=model.score(frames['train'][:,indices],labels['train']),
            validation_balanced_accuracy=balanced_accuracy_score(labels['val'],model.predict(frames['val'][:,indices])),
            top_features=[dict(name=names[indices[i]],importance=float(model.feature_importances_[i]))
                          for i in np.argsort(model.feature_importances_)[-8:][::-1]])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
