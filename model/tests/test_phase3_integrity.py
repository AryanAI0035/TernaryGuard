"""Regression gates for the phase-3 audit findings."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model.artifacts import HeaderReference, export_bundle, load_ternary_bundle, save_bundle
from model.data_pipeline import NBaIoTDataset, get_feature_names
from model.ternary_linear import TernaryMLP
from model.train import RESULT_FIELDS, append_result, prepare_data, train_model


def config(dim):
    return dict(selected_features=list(range(dim)), feature_names=get_feature_names()[:dim],
                scaler=dict(mean=[0.]*dim, scale=[1.]*dim))


def test_schema_known_positions():
    names = get_feature_names()
    assert names[28] == 'H_L0.01_mean'
    assert names[30] == 'HH_L5_weight'
    assert names[65] == 'HH_jit_L5_weight'
    assert names[80] == 'HpHp_L5_weight'
    assert names[114] == 'HpHp_L0.01_pcc'


def test_sampling_covers_late_devices_and_rejects_bad_headers(tmp_path):
    for device in [1, 8, 9]:
        pd.DataFrame(np.full((30,115), device), columns=get_feature_names()).to_csv(
            tmp_path/f'{device}.benign.csv', index=False)
    ds = NBaIoTDataset(str(tmp_path), max_samples_per_class=15)
    X,y = ds.load_data()
    assert len(X) == 15
    assert ds.metadata.groupby('device').size().to_dict() == {'1':5,'8':5,'9':5}
    assert set(y) == {0}
    bad = pd.DataFrame(np.ones((2,115)), columns=list(reversed(get_feature_names())))
    bad.to_csv(tmp_path/'9.benign.csv', index=False)
    with pytest.raises(ValueError, match='schema'):
        ds.load_data()


@pytest.mark.parametrize("transform", ["identity", "signed_log1p"])
def test_preprocessing_never_fits_heldout_rows(monkeypatch, transform):
    n = 11*20
    X = pd.DataFrame(np.random.default_rng(4).normal(size=(n*3,115)),columns=get_feature_names())
    X.iloc[n:] += 1000
    y = pd.Series(np.tile(np.repeat(np.arange(11),20),3))
    def fake_load(self):
        self.metadata = pd.DataFrame(dict(device=np.repeat(['1','8','9'],n),
            source_id=np.repeat([0,1,2],n), row=np.tile(np.arange(n),3)))
        return X,y
    seen = []
    def select(self, features, labels, **kwargs):
        seen.extend(features.index.tolist())
        return features.iloc[:,:3].to_numpy(), [0,1,2]
    monkeypatch.setattr(NBaIoTDataset,'load_data',fake_load)
    monkeypatch.setattr(NBaIoTDataset,'select_features',select)
    loaders, cfg, manifest, ds, parts = prepare_data('',3,1000,32,42,['8'],['9'],feature_transform=transform)
    assert seen == list(range(n))
    from model.data_pipeline import transform_features
    np.testing.assert_allclose(cfg['scaler']['mean'],transform_features(X.iloc[:n,:3],transform).mean())
    assert cfg['transform'] == transform
    assert loaders['test'].dataset.tensors[0].mean() > 3
    assert not set(parts['train']) & set(parts['test'])
    assert manifest['partitions']['test']['devices'] == ['9']


@pytest.mark.parametrize('norm',[False,True])
def test_export_preserves_noninteger_bias_logits_and_exact_float_constants(tmp_path,norm):
    torch.manual_seed(19)
    model = TernaryMLP(3,[5,3],2,use_rmsnorm=norm)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'bias' in name:
                p.copy_(torch.linspace(-.49,.49,p.numel()))
    path = tmp_path/'weights.h'
    model.export_weights_header(str(path))
    x = torch.randn(100,3)
    expected = model(x).detach().numpy()
    actual = HeaderReference(path)(x.numpy())
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
    np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
    text = path.read_text()
    assert 'float TGPROGMEM tg_scale_0' in text
    assert 'float TGPROGMEM tg_bias_0' in text


def test_export_only_loads_frozen_bundle_without_dataset(tmp_path):
    model = TernaryMLP(3,[5],2)
    checkpoint = tmp_path/'model.pt'
    cfg = config(3)
    save_bundle(checkpoint,model,cfg,dict(run_id='test'))
    destination=tmp_path/'export'
    subprocess.run([sys.executable,'model/train.py','--export-only','--checkpoint',str(checkpoint),
                    '--output-dir',str(destination),'--data-dir',str(tmp_path/'absent')],check=True)
    assert json.loads((destination/'feature_config.json').read_text()) == cfg
    bundle=torch.load(checkpoint,weights_only=True)
    bundle['preprocessing']['selected_features']=[1,0,2]
    torch.save(bundle,checkpoint)
    with pytest.raises(ValueError,match='fingerprint'):
        load_ternary_bundle(checkpoint)
    torch.save(model.state_dict(),checkpoint)
    with pytest.raises(ValueError,match='version-2'):
        export_bundle(checkpoint,destination)


def test_results_quotes_and_schema(tmp_path):
    path=tmp_path/'results.csv'
    append_result(path,dict(hidden_dims='[32, 16]',notes='a, b'))
    rows=list(csv.DictReader(path.open()))
    assert rows[0]['hidden_dims']=='[32, 16]' and rows[0]['notes']=='a, b'
    assert list(rows[0])==RESULT_FIELDS and None not in rows[0]
    path.write_text('wrong,header\n')
    with pytest.raises(ValueError,match='schema'):
        append_result(path,{})


def test_per_layer_padding_and_fp_classifier_budget():
    model=TernaryMLP(3,[3],3,use_rmsnorm=False)
    assert model.estimate_packed_size()['packed_weight_bytes']==6  # ceil(9/4) twice
    assert model.estimate_packed_size()['bias_bytes']==24
    fp=TernaryMLP(3,[3],3,ternary_output=False,use_rmsnorm=False)
    assert fp.estimate_packed_size()['fp_weight_bytes']==36
    with pytest.raises(ValueError,match='ternary classifier'):
        fp.export_weights_header('/unused.h')


def test_heldout_devices_require_disjoint_sets_and_class_coverage():
    ds=NBaIoTDataset()
    ds.metadata=pd.DataFrame({'device':['1','8','9']})
    with pytest.raises(ValueError,match='disjoint'):
        ds.device_split_indices([0,0,0],['9'],['9'])
    with pytest.raises(ValueError,match='all 11'):
        ds.device_split_indices([0,0,0])


def test_feature_selection_does_not_spend_slots_on_duplicates():
    rng=np.random.default_rng(41)
    labels=pd.Series(np.tile([0,1],100))
    signal=labels.to_numpy()+rng.normal(0,.05,200)
    frame=pd.DataFrame({'signal':signal,'duplicate':signal.copy(),
                        'other':rng.normal(size=200)})
    _,indices=NBaIoTDataset().select_features(frame,labels,n_features=2,correlation_limit=.98)
    assert not {0,1}.issubset(indices)
    assert 2 in indices


def test_golden_vectors_include_every_class(tmp_path):
    from model.train import verify_export
    from torch.utils.data import DataLoader,TensorDataset
    model=TernaryMLP(3,[4],2)
    x=torch.randn(80,3); y=torch.tensor([0]*40+[1]*40)
    header=tmp_path/'weights.h'; model.export_weights_header(str(header))
    report=verify_export(model,header,DataLoader(TensorDataset(x,y),batch_size=8),tmp_path)
    golden=np.load(tmp_path/'golden_vectors.npz')
    assert np.bincount(golden['labels']).tolist()==[24,24]
    assert report['samples']==80 and report['prediction_disagreements']==0


def test_header_compiles_as_c_and_constants_have_expected_sizes(tmp_path):
    import shutil
    cc=shutil.which('cc')
    if not cc:
        pytest.skip('C compiler unavailable')
    model=TernaryMLP(20,[32,16],11)
    model.export_weights_header(str(tmp_path/'model_weights.h'))
    source=tmp_path/'check.c'
    source.write_text('''#include "model_weights.h"
_Static_assert(sizeof(tg_bias_0) == 32*4, "FP32 biases required");
_Static_assert(sizeof(tg_weights_0) == 160, "Packed weights required");
_Static_assert(sizeof(TG_DIMS) == 8, "16-bit dimensions required");
int main(void) { return TG_FORMAT_VERSION != 2; }
''')
    subprocess.run([cc,'-std=c11','-Wall','-Werror',str(source),'-o',str(tmp_path/'check')],check=True)
    subprocess.run([str(tmp_path/'check')],check=True)


def test_signed_log_transform_handles_negative_covariances_and_large_values():
    from model.data_pipeline import transform_features
    x=np.array([-1e12,-1.,0.,1e-12,1.,1e12])
    y=transform_features(x,'signed_log1p')
    assert np.isfinite(y).all()
    np.testing.assert_array_equal(np.sign(y),np.sign(x))
    np.testing.assert_allclose(np.sign(y)*np.expm1(np.abs(y)),x,rtol=1e-12)
    with pytest.raises(ValueError,match='Unknown feature transform'):
        transform_features(x,'unrecognized')


def test_scaler_config_rejects_feature_dimension_mismatch(tmp_path):
    ds=NBaIoTDataset()
    ds.preprocess(np.ones((5,3)))
    with pytest.raises(ValueError,match='dimensions'):
        ds.save_feature_config([0,1],str(tmp_path/'config.json'))


def test_preprocessing_hash_survives_json_integer_label_keys():
    from model.artifacts import json_hash
    value=config(3)
    value['labels']={i:str(i) for i in range(11)}
    assert json_hash(value)==json_hash(json.loads(json.dumps(value)))


def test_frozen_rows_and_source_content_are_verified(tmp_path):
    import hashlib
    from model.artifacts import json_hash
    from model.validate_export import frozen_test_loader
    source=tmp_path/'9.benign.csv'
    frame=pd.DataFrame(np.arange(5*115,dtype=float).reshape(5,115),columns=get_feature_names())
    frame.to_csv(source,index=False)
    metadata=pd.DataFrame(dict(source_id=[0,0],device=['9','9'],row=[1,3]))
    manifest=dict(sources=[dict(path=source.name,device='9',label=0,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest())],
        partitions={'test':dict(row_identity_sha256=hashlib.sha256(metadata.to_csv(index=False).encode()).hexdigest())})
    (tmp_path/'data_manifest.json').write_text(json.dumps(manifest))
    np.savez_compressed(tmp_path/'split_rows.npz',source_id=[0,0],row=[1,3],split=['test','test'])
    checkpoint=tmp_path/'model.pt'
    cfg=config(3);cfg['transform']='signed_log1p'
    save_bundle(checkpoint,TernaryMLP(3,[4],2),cfg,dict(data_manifest_sha256=json_hash(manifest)))
    loader=frozen_test_loader(checkpoint,tmp_path)
    np.testing.assert_allclose(loader.dataset.tensors[0].numpy(),np.log1p(frame.iloc[[1,3],:3]),rtol=1e-6)
    np.savez_compressed(tmp_path/'split_rows.npz',source_id=[0,0],row=[1,2],split=['test','test'])
    with pytest.raises(ValueError,match='row identities'):
        frozen_test_loader(checkpoint,tmp_path)
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='source changed'):
        frozen_test_loader(checkpoint,tmp_path)
