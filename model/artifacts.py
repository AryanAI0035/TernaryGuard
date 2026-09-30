"""Versioned training bundles and an independent reader for exported C constants."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from model.ternary_linear import TernaryMLP
from model.data_pipeline import get_feature_names


def json_hash(value):
    # JSON object keys are strings on disk; normalize before sorting so integer
    # label-map keys hash identically before and after a JSON round trip.
    normalized = json.loads(json.dumps(value))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def save_bundle(path, model, preprocessing, provenance, metrics=None):
    architecture = dict(input_dim=model.input_dim, hidden_dims=list(model.hidden_dims),
                        output_dim=model.output_dim)
    if isinstance(model, TernaryMLP):
        architecture.update(use_rmsnorm=model.use_rmsnorm, ternary_output=model.ternary_output)
    bundle = dict(format_version=2, model_type=type(model).__name__,
                  architecture=architecture, state_dict=model.cpu().state_dict(),
                  preprocessing=preprocessing, preprocessing_sha256=json_hash(preprocessing),
                  provenance=provenance, metrics=metrics or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    return bundle


def load_ternary_bundle(path):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('format_version') != 2 or bundle.get('model_type') != 'TernaryMLP':
        raise ValueError('A version-2 ternary training bundle is required; legacy weights cannot be exported safely')
    config = bundle['preprocessing']
    if json_hash(config) != bundle['preprocessing_sha256']:
        raise ValueError('Preprocessing fingerprint mismatch')
    if config.get('transform', 'identity') not in ('identity', 'signed_log1p'):
        raise ValueError('Unsupported feature transform')
    dim = bundle['architecture']['input_dim']
    indices = config['selected_features']
    if len(indices) != dim or len(set(indices)) != dim:
        raise ValueError('Invalid selected feature indices')
    if config['feature_names'] != [get_feature_names()[i] for i in indices]:
        raise ValueError('Feature schema mismatch')
    for key in ('mean', 'scale'):
        values = np.asarray(config['scaler'][key])
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError('Invalid scaler parameters')
    if np.any(np.asarray(config['scaler']['scale']) <= 0):
        raise ValueError('Scaler must have positive scales')
    model = TernaryMLP(**bundle['architecture'])
    model.load_state_dict(bundle['state_dict'])
    return model.eval(), bundle


def export_bundle(checkpoint, destination):
    model, bundle = load_ternary_bundle(checkpoint)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model.export_weights_header(str(destination / 'model_weights.h'))
    (destination / 'feature_config.json').write_text(json.dumps(bundle['preprocessing'], indent=2)+'\n')
    manifest = dict(format_version=2, architecture=bundle['architecture'],
                    preprocessing_sha256=bundle['preprocessing_sha256'],
                    checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    header_sha256=hashlib.sha256((destination/'model_weights.h').read_bytes()).hexdigest(),
                    provenance=bundle['provenance'])
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return model, bundle


class HeaderReference:
    """NumPy inference from actual header text, without access to training weights.

    This is an export-validation oracle, not the phase-4 optimized C engine.
    Inputs are selected, standardized float32 vectors; output is raw logits.
    """
    def __init__(self, path):
        text = Path(path).read_text()
        def array(name, dtype=np.float32):
            match = re.search(r'\b'+name+r'\[\]\s*=\s*\{(.*?)\};', text, re.S)
            if not match:
                raise ValueError(f'Missing exported array: {name}')
            tokens = [v.strip().rstrip('f') for v in match.group(1).split(',') if v.strip()]
            return np.array([int(v, 0) if dtype == np.uint8 else float(v) for v in tokens], dtype=dtype)
        def scalar(name):
            match = re.search(r'\b'+name+r'\s*=\s*([^;]+);', text)
            if not match:
                raise ValueError(f'Missing exported scalar: {name}')
            return np.float32(match.group(1).rstrip('f'))
        self.dims = array('TG_DIMS', np.int64)
        self.norm = bool(int(re.search(r'#define TG_USE_RMSNORM (\d+)', text).group(1)))
        self.layers = []
        for i, (ins, outs) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            packed = array(f'tg_weights_{i}', np.uint8)
            codes = ((packed[:, None] >> (2*np.arange(4))) & 3).flatten()[:ins*outs]
            if np.any(codes == 3):
                raise ValueError('Reserved weight code in exported artifact')
            weights = np.choose(codes, [0., 1., -1.]).astype(np.float32).reshape(outs, ins)
            bias = array(f'tg_bias_{i}')
            gamma = array(f'tg_rmsnorm_gamma_{i}') if self.norm and i < len(self.dims)-2 else None
            eps = scalar(f'tg_rmsnorm_eps_{i}') if gamma is not None else None
            self.layers.append((weights, bias, scalar(f'tg_scale_{i}'), gamma, eps))

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        for i, (weights, bias, scale, gamma, eps) in enumerate(self.layers):
            if gamma is not None:
                x = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+eps)*gamma
            x = (x @ weights.T)*scale + bias
            if i < len(self.layers)-1:
                x = np.maximum(x, 0)
        return x
