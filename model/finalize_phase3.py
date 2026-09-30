#!/usr/bin/env python3
"""Select a model using validation only, then train architecture-matched baselines.

A nonzero recall for every validation class is a minimal eligibility check, not
proof of deployment readiness. The final report retains all class-level metrics.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.artifacts import load_ternary_bundle, export_bundle, save_bundle
from model.data_pipeline import get_label_names
from model.train import (FP32MLP, evaluate, train_model, quantize_int8, seed_everything,
                         record_experiment, verify_export)
from model.validate_export import frozen_test_loader
from sklearn.metrics import recall_score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--data-dir',type=Path,default=Path('data/raw/nbaiot'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=False)
    candidates=[]
    for run in args.runs:
        loader=frozen_test_loader(run/'ternary_2bit.pt',args.data_dir,partition='val')
        for name in ['ternary_2bit']+[f'ablation_{a}_{b}' for a,b in [(16,8),(24,12),(32,16),(48,24),(64,32)]]:
            path=run/(name+'.pt')
            model,bundle=load_ternary_bundle(path)
            metrics=evaluate(model,loader,nn.CrossEntropyLoss(),torch.device('cpu'))
            recall=recall_score(metrics['labels'],metrics['preds'],labels=list(range(11)),average=None,zero_division=0)
            candidates.append(dict(checkpoint=str(path),validation_macro_f1=metrics['macro_f1'],
                validation_recall={get_label_names()[i]:float(v) for i,v in enumerate(recall)},
                eligible=bool((recall>0).all())))
    eligible=[c for c in candidates if c['eligible']]
    if not eligible:
        (output/'selection.json').write_text(json.dumps(dict(candidates=candidates,status='No eligible candidate'),indent=2)+'\n')
        raise RuntimeError('Every candidate misses a validation class; retain results for further model work')
    chosen=max(eligible,key=lambda c:c['validation_macro_f1'])
    selection=dict(policy='Require nonzero recall for every validation class, then maximize validation macro-F1; test metrics unused',
                   chosen=chosen,candidates=candidates)
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('SELECTED',json.dumps(chosen),flush=True)
    checkpoint=Path(chosen['checkpoint'])
    model,bundle=load_ternary_bundle(checkpoint)
    # Preserve exactly the selected split/transform and its source identities.
    for filename in ['data_manifest.json','split_rows.npz','feature_config.json']:
        shutil.copyfile(checkpoint.parent/filename,output/filename)
    train=frozen_test_loader(checkpoint,args.data_dir,partition='train')
    loaders=dict(train=DataLoader(train.dataset,batch_size=1024,shuffle=True,
                     generator=torch.Generator().manual_seed(bundle['provenance']['seed'])),
                 val=frozen_test_loader(checkpoint,args.data_dir,partition='val'),
                 test=frozen_test_loader(checkpoint,args.data_dir,partition='test'))
    seed=bundle['provenance']['seed']
    original_args=bundle['provenance']['arguments']
    epochs=original_args['ablation_epochs'] if checkpoint.stem.startswith('ablation_') else original_args['epochs']
    provenance=dict(bundle['provenance'],run_id=output.name,selection=selection,
                    matched_baseline_epochs=epochs,selected_checkpoint=str(checkpoint))
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    # Re-evaluate the selected weights without any test-driven retraining.
    metrics=evaluate(model,loaders['test'],nn.CrossEntropyLoss(),torch.device('cpu'))
    metrics.update(val_macro_f1=chosen['validation_macro_f1'],history=bundle['metrics']['history'],
                   best_epoch=bundle['metrics']['best_epoch'])
    record_experiment('ternary_2bit',model,metrics,bundle['preprocessing'],provenance,output,output/'results.csv',seed)
    seed_everything(seed)
    fp32=FP32MLP(model.input_dim,model.hidden_dims,model.output_dim)
    metrics=train_model(fp32,loaders,epochs=epochs,model_name='Matched FP32',lr=original_args['lr'])
    record_experiment('baseline_fp32',fp32,metrics,bundle['preprocessing'],provenance,output,output/'results.csv',seed)
    int8,_=quantize_int8(fp32)
    metrics=evaluate(int8,loaders['test'],nn.CrossEntropyLoss(),torch.device('cpu'))
    metrics['val_macro_f1']=evaluate(int8,loaders['val'],nn.CrossEntropyLoss(),torch.device('cpu'))['macro_f1']
    record_experiment('quantized_int8',int8,metrics,bundle['preprocessing'],provenance,output,output/'results.csv',seed)
    model,_=export_bundle(output/'ternary_2bit.pt',output/'export')
    print('EXPORT',verify_export(model,output/'export/model_weights.h',loaders['test'],output/'export'),flush=True)


if __name__=='__main__':
    main()
