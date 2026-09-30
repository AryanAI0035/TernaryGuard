#!/usr/bin/env python3
"""
TernaryGuard — Phase 3 Training & Ablation Script

Trains and evaluates three model variants on N-BaIoT:
    1. FP32 baseline  (standard nn.Linear MLP)
    2. INT8 quantized  (simulated symmetric weight quantization of the FP32 model)
    3. Ternary 1.58-bit (TernaryMLP with STE, 2-bit packed storage)

Also runs hidden-width ablation sweeps and exports the final ternary
model's weights to model_weights.h.

Usage:
    python3 model/train.py                          # Full pipeline
    python3 model/train.py --ablation-only          # Only ablation sweep
    python3 model/train.py --export-only            # Only export weights
"""

import argparse
import csv
import copy
import datetime
import json
import logging
import math
import os
import sys
import time
import random
import hashlib
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.ternary_linear import TernaryMLP, TernaryLinear, RMSNorm
from model.data_pipeline import NBaIoTDataset, get_feature_names, get_label_names, transform_features
from model.artifacts import save_bundle, export_bundle, HeaderReference, json_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = PROJECT_ROOT / "results.csv"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


# ─────────────────────── FP32 Baseline MLP ───────────────────────────

class FP32MLP(nn.Module):
    """
    Full-precision MLP baseline with RMSNorm — architecturally identical
    to TernaryMLP so the comparison isolates precision, not normalization.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 use_rmsnorm: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            if use_rmsnorm:
                layers.append(RMSNorm(prev_dim))
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "ternary": 0}

    def model_size_bytes(self) -> int:
        """Total bytes for FP32 storage (all parameters as float32)."""
        return sum(p.numel() * 4 for p in self.parameters())


# ─────────────────────── Training Loop ───────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model. Returns dict with loss, accuracy, precision, recall, f1."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = len(all_labels)

    return {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "preds": all_preds,
        "labels": all_labels,
    }


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    model_name: str = "model",
    patience: int = 10,
) -> Dict[str, float]:
    """
    Train model with early stopping on validation F1.
    Returns test metrics dict for the best-validation-F1 checkpoint.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    if epochs < 1:
        raise ValueError("epochs must be positive")
    history = []
    best_epoch = 0
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    logger.info(f"Training {model_name} for up to {epochs} epochs (patience={patience})")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(val_metrics["macro_f1"])
        history.append(dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                            val_loss=val_metrics["loss"], val_macro_f1=val_metrics["macro_f1"]))

        # Log every 5 epochs or first/last
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

        # Early stopping on validation F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (best val_f1={best_val_f1:.4f})")
                break

    # Restore best checkpoint and evaluate on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    logger.info(
        f"  {model_name} TEST: acc={test_metrics['accuracy']:.4f} "
        f"prec={test_metrics['precision']:.4f} "
        f"rec={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )

    test_metrics.update(history=history, best_epoch=best_epoch, val_macro_f1=best_val_f1)
    return test_metrics


# ─────────────────────── INT8 Quantization ───────────────────────────

class INT8MLP(nn.Module):
    """
    Simulated INT8 MLP: weights are quantized to int8 per-tensor
    (scale + zero_point), dequantized to float32 for inference.
    This gives an honest model-size estimate without depending on
    PyTorch's deprecated quantization engine.
    """

    def __init__(self, fp32_model: FP32MLP):
        super().__init__()
        self.input_dim = fp32_model.input_dim
        self.hidden_dims = fp32_model.hidden_dims
        self.output_dim = fp32_model.output_dim

        # Deep copy the network structure
        self.network = copy.deepcopy(fp32_model.network)

        # Quantize each Linear layer's weights to int8
        self._quantize_weights()

    def _quantize_weights(self):
        """Per-tensor symmetric INT8 quantization of all Linear layers."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                # Symmetric quantization: scale = max(|w|) / 127
                scale = w.abs().max() / 127.0
                if scale == 0:
                    scale = torch.tensor(1.0)
                # Quantize to int8 range, then dequantize back
                w_int8 = (w / scale).round().clamp(-128, 127)
                module.weight.data = w_int8 * scale
                # Store scale for size estimation
                module.register_buffer('_int8_scale', scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def model_size_bytes(self) -> int:
        """INT8 weights (1B) + FP32 biases (4B) + scales + RMSNorm gamma (4B)."""
        size = 0
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                size += module.weight.numel() * 1    # int8: 1 byte
                if module.bias is not None:
                    size += module.bias.numel() * 4  # float32 bias
                size += 4  # float32 scale per layer
            elif isinstance(module, RMSNorm):
                size += module.gamma.numel() * 4     # float32 gamma
        return size


def quantize_int8(fp32_model: FP32MLP) -> Tuple[nn.Module, int]:
    """
    Post-training INT8 quantization of an FP32 model.
    Returns (quantized_model, model_size_bytes).
    """
    int8_model = INT8MLP(fp32_model)
    return int8_model, int8_model.model_size_bytes()


# ─────────────────────── Reproducible experiments ────────────────────

RESULT_FIELDS = ["phase", "experiment", "model_type", "architecture", "hidden_dims",
    "num_params", "accuracy", "precision", "recall", "f1_score", "model_size_bytes",
    "inference_latency_us", "flash_bytes", "ram_bytes", "lut_count", "ff_count",
    "dsp_count", "bram_count", "power_mw", "clock_mhz", "notes", "run_id", "seed",
    "macro_f1", "val_macro_f1", "estimated_ram_bytes", "ternary_weight_terms",
    "nonzero_ternary_terms", "dense_weight_multiplications", "output_scale_multiplications"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def append_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline='') as f:
            reader = csv.reader(f)
            if next(reader) != RESULT_FIELDS or any(len(r) != len(RESULT_FIELDS) for r in reader):
                raise ValueError(f"Invalid results schema: {path}")
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prepare_data(data_dir, n_features, cap, batch_size, seed, val_devices, test_devices,
                 mi_samples=20000, correlation_limit=0.98, feature_transform="identity"):
    dataset = NBaIoTDataset(data_dir=data_dir, max_samples_per_class=cap, seed=seed)
    X, y = dataset.load_data()
    if X.empty:
        raise ValueError('No data found; run research/download_nbaiot.py first')
    partitions = dataset.device_split_indices(y, val_devices, test_devices)
    X = transform_features(X, feature_transform)
    train = partitions['train']
    selection_rows = train
    if len(train) > mi_samples:
        selection_rows, _ = train_test_split(train, train_size=mi_samples,
                                            stratify=y.iloc[train], random_state=seed)
    logger.info('Selecting features on %d training-only rows', len(selection_rows))
    _, indices = dataset.select_features(X.iloc[selection_rows], y.iloc[selection_rows],
                                         n_features=n_features, correlation_limit=correlation_limit)
    splits = {}
    for name, rows in partitions.items():
        selected = X.iloc[rows, indices].to_numpy()
        splits['X_'+name] = dataset.preprocess(selected, fit=(name == 'train')).astype(np.float32)
        splits['y_'+name] = y.iloc[rows].to_numpy()
    config = dict(transform=feature_transform, selected_features=[int(i) for i in indices],
                  feature_names=[get_feature_names()[i] for i in indices],
                  scaler=dict(mean=dataset.scaler.mean_.tolist(), scale=dataset.scaler.scale_.tolist()),
                  labels=get_label_names())
    manifest = dict(protocol='device_holdout', sampling='equal quota per class/capture; no shortfall redistribution',
                    seed=seed, cap_per_class=cap, mi_training_samples=len(selection_rows),
                    feature_correlation_limit=correlation_limit, feature_transform=feature_transform,
                    sources=dataset.sources, partitions={})
    for name, rows in partitions.items():
        metadata = dataset.metadata.iloc[rows]
        manifest['partitions'][name] = dict(samples=len(rows), devices=sorted(set(metadata.device)),
            class_counts={str(k): int(v) for k,v in y.iloc[rows].value_counts().sort_index().items()},
            row_identity_sha256=hashlib.sha256(metadata.to_csv(index=False).encode()).hexdigest())
    return dataset.to_torch_loaders(splits, batch_size), config, manifest, dataset, partitions


def metric_report(metrics):
    labels, preds = metrics['labels'], metrics['preds']
    return dict(accuracy=float(metrics['accuracy']), macro_f1=float(metrics['macro_f1']),
        weighted_f1=float(metrics['f1']), val_macro_f1=metrics.get('val_macro_f1'),
        best_epoch=metrics.get('best_epoch'), history=metrics.get('history', []),
        per_class=classification_report(labels, preds, labels=list(range(11)),
            target_names=list(get_label_names().values()), output_dict=True, zero_division=0),
        confusion_matrix=confusion_matrix(labels, preds, labels=list(range(11))).tolist(),
        binary_detection=classification_report(labels != 0, preds != 0, labels=[False, True],
            target_names=['benign', 'attack'], output_dict=True, zero_division=0))


def record_experiment(name, model, metrics, config, provenance, output, results_path, seed):
    report = metric_report(metrics)
    failed_classes = [name for name in get_label_names().values()
                      if report['per_class'][name]['recall'] == 0]
    report['readiness'] = dict(zero_recall_classes=failed_classes,
        all_classes_have_nonzero_recall=not failed_classes,
        note='Nonzero recall is only a minimum sanity check, not a deployment acceptance threshold')
    if failed_classes:
        logger.warning('%s has zero recall for %s; do not claim complete 11-class detection', name, failed_classes)
    (output/(name+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    save_bundle(output/(name+'.pt'), model, config, provenance, report)
    ternary = isinstance(model, TernaryMLP)
    layers = [m for m in model.modules() if isinstance(m, TernaryLinear)]
    size = model.estimate_packed_size()['total_bytes'] if ternary else model.model_size_bytes()
    row = dict(phase=3, experiment=name, model_type='ternary' if ternary else ('int8' if isinstance(model, INT8MLP) else 'fp32'),
        architecture=type(model).__name__, hidden_dims=str(model.hidden_dims),
        num_params=sum(p.numel() for p in model.parameters()), accuracy=metrics['accuracy'],
        precision=metrics['precision'], recall=metrics['recall'], f1_score=metrics['f1'],
        macro_f1=metrics['macro_f1'], val_macro_f1=metrics.get('val_macro_f1', ''),
        model_size_bytes=size, run_id=provenance['run_id'], seed=seed,
        estimated_ram_bytes=model.estimate_ram_usage()['total_bytes'] if ternary else '',
        ternary_weight_terms=sum(m.weight.numel() for m in layers),
        nonzero_ternary_terms=sum(int((m.get_ternary_weights()[0] != 0).sum()) for m in layers),
        dense_weight_multiplications=0 if ternary else sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear)),
        output_scale_multiplications=sum(m.out_features for m in layers),
        notes='device-held-out test; model bytes are constant-storage estimate; hardware unmeasured; '+
            ('INT8 weights simulated with FP32 execution' if isinstance(model, INT8MLP) else 'FP32 biases and normalization'))
    append_result(results_path, row)
    append_result(output/'accuracy_ablation.csv', row)
    return report


def verify_export(model, header, loader, output):
    reference = HeaderReference(header)
    max_error, disagreements, total = 0., 0, 0
    golden_x, golden_y, golden_logits = [], [], []
    golden_counts = np.zeros(model.output_dim, dtype=int)
    with torch.no_grad():
        for x, y in loader:
            expected = model(x).numpy()
            actual = reference(x.numpy())
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            max_error = max(max_error, float(np.max(np.abs(actual-expected))))
            disagreements += int(np.sum(actual.argmax(1) != expected.argmax(1)))
            total += len(x)
            for label in range(model.output_dim):
                take = np.flatnonzero(y.numpy() == label)[:max(0, 24-golden_counts[label])]
                if len(take):
                    golden_x.append(x.numpy()[take]); golden_y.append(y.numpy()[take])
                    golden_logits.append(expected[take]); golden_counts[label] += len(take)
    if disagreements:
        raise ValueError(f'Export changes {disagreements}/{total} classifications')
    np.savez_compressed(output/'golden_vectors.npz', inputs=np.concatenate(golden_x), labels=np.concatenate(golden_y),
                        logits=np.concatenate(golden_logits))
    report = dict(samples=total, prediction_disagreements=disagreements, max_abs_logit_error=max_error,
                  atol=2e-5, rtol=2e-5, golden_samples_per_class=golden_counts.tolist(),
                  oracle='NumPy reader of exported C header; float32 arithmetic')
    (output/'export_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description='TernaryGuard reproducible phase 3')
    parser.add_argument('--data-dir', default=str(PROJECT_ROOT/'data/raw/nbaiot'))
    parser.add_argument('--n-features', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--ablation-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-samples-per-class', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-devices', nargs='+', default=['8'])
    parser.add_argument('--test-devices', nargs='+', default=['9'])
    parser.add_argument('--mi-samples', type=int, default=20000)
    parser.add_argument('--feature-correlation-limit', type=float, default=0.98)
    parser.add_argument('--feature-transform', choices=['identity','signed_log1p'], default='signed_log1p')
    parser.add_argument('--run-id', default=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--results', type=Path, default=RESULTS_CSV)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--export-only', action='store_true')
    modes.add_argument('--ablation-only', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--no-ablation', action='store_true')
    args = parser.parse_args()
    if args.ablation_only and args.no_ablation:
        parser.error('--ablation-only cannot be combined with --no-ablation')
    if min(args.epochs, args.ablation_epochs, args.batch_size, args.mi_samples) < 1:
        parser.error('Epochs, batch size, and MI sample count must be positive')
    torch.set_num_threads(1)
    seed_everything(args.seed)
    output = args.output_dir or CHECKPOINTS_DIR/args.run_id
    if args.export_only:
        if args.checkpoint is None:
            parser.error('--export-only requires --checkpoint pointing to a version-2 bundle')
        export_bundle(args.checkpoint, output)
        logger.info('Exported frozen bundle to %s without reading training data', output)
        return
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/'source'
    snapshot.mkdir()
    for source in (PROJECT_ROOT/'model').glob('*.py'):
        (snapshot/source.name).write_bytes(source.read_bytes())
    (snapshot/'feature_schema.json').write_bytes((PROJECT_ROOT/'model/feature_schema.json').read_bytes())
    loaders, config, manifest, dataset, partitions = prepare_data(args.data_dir, args.n_features,
        args.max_samples_per_class, args.batch_size, args.seed, args.val_devices, args.test_devices, args.mi_samples, args.feature_correlation_limit, args.feature_transform)
    (output/'data_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    split_names = np.empty(len(dataset.metadata), dtype='U5')
    for name, rows in partitions.items():
        split_names[rows] = name
    np.savez_compressed(output/'split_rows.npz', source_id=dataset.metadata.source_id.to_numpy(),
        row=dataset.metadata.row.to_numpy(), split=split_names)
    (output/'feature_config.json').write_text(json.dumps(config, indent=2)+'\n')
    provenance = dict(run_id=args.run_id, seed=args.seed, data_manifest_sha256=json_hash(manifest),
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        versions=dict(torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, python=sys.version),
        source_sha256={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((PROJECT_ROOT/'model').glob('*.py'))})
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    candidates = []
    if not args.ablation_only:
        seed_everything(args.seed)
        loaders['train'].generator.manual_seed(args.seed)
        fp32 = FP32MLP(args.n_features, [32,16], 11)
        metrics = train_model(fp32, loaders, epochs=args.epochs, lr=args.lr, model_name='FP32')
        record_experiment('baseline_fp32', fp32, metrics, config, provenance, output, args.results, args.seed)
        int8, _ = quantize_int8(fp32)
        metrics = evaluate(int8, loaders['test'], nn.CrossEntropyLoss(), torch.device('cpu'))
        metrics['val_macro_f1'] = evaluate(int8, loaders['val'], nn.CrossEntropyLoss(), torch.device('cpu'))['macro_f1']
        record_experiment('quantized_int8', int8, metrics, config, provenance, output, args.results, args.seed)
    configs = [] if args.ablation_only else [('ternary_2bit', [32,16], args.epochs)]
    if not args.no_ablation:
        configs += [(f'ablation_{a}_{b}', [a,b], args.ablation_epochs) for a,b in [(16,8),(24,12),(32,16),(48,24),(64,32)]]
    for name, dims, epochs in configs:
        seed_everything(args.seed)
        loaders['train'].generator.manual_seed(args.seed)
        model = TernaryMLP(args.n_features, dims, 11)
        metrics = train_model(model, loaders, epochs=epochs, lr=args.lr, model_name=name)
        report = record_experiment(name, model, metrics, config, provenance, output, args.results, args.seed)
        candidates.append((report['val_macro_f1'], name))
    if args.ablation_only:
        logger.info('Ablations saved; production export unchanged')
        return
    # Production architecture remains prespecified for comparison with FP32/INT8.
    # Report width selection separately using validation only, never test scores.
    selection = max(candidates)
    (output/'selection.json').write_text(json.dumps(dict(production='ternary_2bit',
        best_validation_candidate=selection[1], best_validation_macro_f1=selection[0],
        policy='production width fixed at 32/16; ablation ranking uses validation only'), indent=2)+'\n')
    model, _ = export_bundle(output/'ternary_2bit.pt', output/'export')
    validation = verify_export(model, output/'export/model_weights.h', loaders['test'], output/'export')
    logger.info('Export parity passed: %s', validation)
    logger.info('Run saved to %s. Assess per-class held-out metrics before deployment.', output)


if __name__ == '__main__':
    main()
