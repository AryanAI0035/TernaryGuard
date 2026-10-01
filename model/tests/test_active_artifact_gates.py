"""Regression tests for active-file verification and invalidated-result plots."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from model.artifacts import export_bundle, json_hash, save_bundle
from model.data_pipeline import get_feature_names
from model.ternary_linear import TernaryMLP
from model.validate_export import verify_active

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def active_files(tmp_path):
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    data = tmp_path / 'data'
    data.mkdir()
    run = tmp_path / 'checkpoints' / 'run'
    run.mkdir(parents=True)
    frame = pd.DataFrame(np.arange(5*115).reshape(5,115), columns=get_feature_names())
    source = data / '9.benign.csv'
    frame.to_csv(source, index=False)
    metadata = pd.DataFrame(dict(source_id=[0]*5, device=['9']*5, row=list(range(5))))
    manifest = dict(sources=[dict(path=source.name, device='9', label=0,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest())],
        partitions={'test':dict(row_identity_sha256=hashlib.sha256(
            metadata.to_csv(index=False).encode()).hexdigest())})
    (run/'data_manifest.json').write_text(json.dumps(manifest))
    np.savez(run/'split_rows.npz',source_id=[0]*5,row=np.arange(5),split=['test']*5)
    torch.manual_seed(12)
    model = TernaryMLP(3,[4],2)
    cfg = dict(transform='identity', selected_features=[0,1,2],
               feature_names=get_feature_names()[:3],
               scaler=dict(mean=[0.]*3, scale=[1.]*3))
    checkpoint = run/'model.pt'
    save_bundle(checkpoint,model,cfg,dict(data_manifest_sha256=json_hash(manifest)))
    export_bundle(checkpoint,tmp_path)
    (model_dir/'active_model.json').write_text(json.dumps(dict(
        checkpoint='checkpoints/run/model.pt',header='model_weights.h',
        preprocessing='feature_config.json',manifest='manifest.json',
        architecture=dict(input_dim=3,hidden_dims=[4],output_dim=2,
                          use_rmsnorm=True,ternary_output=True))))
    return tmp_path, model_dir/'active_model.json', data


def corrupt_bias(path):
    before=path.read_text()
    after,count=re.subn(r'(tg_bias_1\[\]\s*=\s*\{)[^,]+',r'\g<1>1000.0f',before)
    assert count == 1 and before != after
    path.write_text(after)


def test_verify_active_cli_catches_corrupted_header_and_never_rewrites(active_files):
    root,active,data=active_files
    cmd=[sys.executable,str(ROOT/'model/validate_export.py'),'--verify-active',
         '--active-model',str(active),'--data-dir',str(data)]
    watched=[p for p in root.rglob('*') if p.is_file()]
    before={p:p.read_bytes() for p in watched}
    clean=subprocess.run(cmd,capture_output=True,text=True)
    assert clean.returncode == 0, clean.stderr
    report=json.loads(clean.stdout)
    assert report['samples']==5 and report['prediction_disagreements']==0
    assert report['artifacts_rewritten'] is False
    assert all(p.read_bytes()==value for p,value in before.items())
    header=root/'model_weights.h'
    corrupt_bias(header)
    corrupted=header.read_bytes()
    bad=subprocess.run(cmd,capture_output=True,text=True)
    assert bad.returncode != 0
    assert 'header_sha256 mismatch' in bad.stderr
    assert header.read_bytes()==corrupted
    header.write_bytes(before[header])
    assert subprocess.run(cmd,capture_output=True,text=True).returncode==0


@pytest.mark.parametrize('kind',['checkpoint','preprocessing'])
def test_verify_active_rejects_other_hash_mismatches(active_files,kind):
    root,active,data=active_files
    if kind=='checkpoint':
        p=root/'checkpoints/run/model.pt'
        p.write_bytes(p.read_bytes()+b'tampered')
    else:
        p=root/'feature_config.json'
        cfg=json.loads(p.read_text());cfg['scaler']['mean'][0]=1.
        p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match=kind+'_sha256 mismatch'):
        verify_active(active,data)


def test_verify_active_evaluates_existing_header_even_if_hash_is_updated(active_files):
    root,active,data=active_files
    header=root/'model_weights.h'
    corrupt_bias(header)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['header_sha256']=hashlib.sha256(header.read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest))
    before=header.read_bytes()
    with pytest.raises(AssertionError,match='Not equal'):
        verify_active(active,data)
    assert header.read_bytes()==before


def plot_command(tmp_path,*extra):
    output=tmp_path/'cli'
    env=dict(os.environ,MPLCONFIGDIR=str(tmp_path/'mpl'))
    result=subprocess.run([sys.executable,str(ROOT/'plot_results.py'),
        '--results',str(ROOT/'docs/benchmarks/legacy_phase3_invalidated.csv'),
        '--output',str(output),*extra],env=env,capture_output=True,text=True)
    return result,output


def test_legacy_plot_cli_refuses_invalidated_csv(tmp_path):
    result,output=plot_command(tmp_path)
    assert result.returncode != 0
    assert 'Refusing INVALIDATED results' in result.stderr
    assert 'legacy_phase3_invalidated.csv' in result.stderr
    assert not list(output.glob('*.png'))


def test_legacy_plot_cli_flag_renders_warning_into_pngs(tmp_path,monkeypatch):
    result,output=plot_command(tmp_path,'--allow-invalidated')
    assert result.returncode==0,result.stderr
    import plot_results
    from matplotlib.figure import Figure
    expected=tmp_path/'expected';expected.mkdir()
    df=plot_results.load_results(ROOT/'docs/benchmarks/legacy_phase3_invalidated.csv',
                                 allow_invalidated=True)
    observed=[]
    original=Figure.savefig
    def inspect_save(fig,path,**kwargs):
        marks=[t for t in fig.texts if t.get_text()=='INVALIDATED — DO NOT USE']
        assert len(marks)==1 and marks[0].get_color()=='red'
        observed.append(Path(path).name)
        return original(fig,path,**kwargs)
    monkeypatch.setattr(Figure,'savefig',inspect_save)
    plot_results.plot_accuracy_vs_quantization(df,expected)
    plot_results.plot_ablation_sweep(df,expected)
    assert set(observed)=={'accuracy_vs_quantization.png','ablation_sweep.png'}
    for name in observed:
        # The CLI's actual PNG pixels must equal a render whose visible warning
        # was inspected immediately before saving, not merely a metadata flag.
        np.testing.assert_array_equal(np.array(Image.open(output/name)),
                                      np.array(Image.open(expected/name)))
