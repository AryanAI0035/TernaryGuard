#!/usr/bin/env python3
"""
TernaryGuard — Phase 3 Training & Ablation Script

Trains and evaluates three model variants on N-BaIoT:
    1. FP32 baseline  (standard nn.Linear MLP)
    2. INT8 quantized  (post-training dynamic quantization of the FP32 model)
    3. Ternary 1.58-bit (TernaryMLP with STE, 2-bit packed storage)

Also runs hidden-width ablation sweeps and exports the final ternary
model's weights to model_weights.h.

Usage:
    python3 model/train.py                          # Full pipeline
    python3 model/train.py --skip-download          # Skip dataset download
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
)

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.ternary_linear import TernaryMLP, TernaryLinear, RMSNorm
from model.data_pipeline import NBaIoTDataset, get_feature_names, get_label_names

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
    """Standard full-precision MLP for baseline comparison."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
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
        """Total bytes for FP32 storage (weights + biases as float32)."""
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

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0

    logger.info(f"Training {model_name} for up to {epochs} epochs (patience={patience})")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(val_metrics["f1"])

        # Log every 5 epochs or first/last
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )

        # Early stopping on validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
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
        """INT8 weights (1 byte each) + FP32 biases (4 bytes each) + scales."""
        size = 0
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                size += module.weight.numel() * 1    # int8: 1 byte
                if module.bias is not None:
                    size += module.bias.numel() * 4  # float32 bias
                size += 4  # float32 scale per layer
        return size


def quantize_int8(fp32_model: FP32MLP) -> Tuple[nn.Module, int]:
    """
    Post-training INT8 quantization of an FP32 model.
    Returns (quantized_model, model_size_bytes).
    """
    int8_model = INT8MLP(fp32_model)
    return int8_model, int8_model.model_size_bytes()


# ─────────────────────── Results Logging ─────────────────────────────

def log_result(
    phase: int,
    experiment: str,
    model_type: str,
    architecture: str,
    hidden_dims: str,
    num_params: int,
    metrics: Dict[str, float],
    model_size_bytes: int,
    flash_bytes: str = "",
    ram_bytes: str = "",
    notes: str = "",
):
    """Append a row to results.csv."""
    row = {
        "phase": phase,
        "experiment": experiment,
        "model_type": model_type,
        "architecture": architecture,
        "hidden_dims": hidden_dims,
        "num_params": num_params,
        "accuracy": f"{metrics['accuracy']:.6f}",
        "precision": f"{metrics['precision']:.6f}",
        "recall": f"{metrics['recall']:.6f}",
        "f1_score": f"{metrics['f1']:.6f}",
        "model_size_bytes": model_size_bytes,
        "inference_latency_us": "",
        "flash_bytes": flash_bytes,
        "ram_bytes": ram_bytes,
        "lut_count": "",
        "ff_count": "",
        "dsp_count": "",
        "bram_count": "",
        "power_mw": "",
        "clock_mhz": "",
        "notes": notes,
    }

    fieldnames = [
        "phase", "experiment", "model_type", "architecture", "hidden_dims",
        "num_params", "accuracy", "precision", "recall", "f1_score",
        "model_size_bytes", "inference_latency_us", "flash_bytes", "ram_bytes",
        "lut_count", "ff_count", "dsp_count", "bram_count", "power_mw",
        "clock_mhz", "notes",
    ]

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

    logger.info(f"  Logged to results.csv: {experiment}")


# ─────────────────────── Data Loading ────────────────────────────────

def load_and_prepare_data(
    data_dir: str = "data/raw/nbaiot",
    n_features: int = 20,
    max_samples_per_class: int = 50000,
    batch_size: int = 256,
) -> Tuple[Dict[str, DataLoader], List[int], NBaIoTDataset]:
    """
    Load N-BaIoT, run feature selection, preprocess, split, create loaders.
    Returns (loaders, selected_feature_indices, dataset_obj).
    """
    logger.info("Loading N-BaIoT dataset...")
    dataset = NBaIoTDataset(data_dir=data_dir, max_samples_per_class=max_samples_per_class)
    X_df, y_s = dataset.load_data()
    logger.info(f"  Loaded {len(X_df)} samples, {X_df.shape[1]} features, "
                f"{y_s.nunique()} classes")

    # Class distribution
    class_counts = y_s.value_counts().sort_index()
    label_names = get_label_names()
    logger.info("  Class distribution:")
    for cls_id, count in class_counts.items():
        logger.info(f"    {cls_id:2d} ({label_names.get(cls_id, '?'):20s}): {count:,}")

    # Feature selection
    logger.info(f"  Running mutual_info feature selection (n={n_features})...")
    X_sel, feature_indices = dataset.select_features(
        X_df, y_s, method="mutual_info", n_features=n_features
    )
    feature_names = get_feature_names()
    logger.info(f"  Selected {len(feature_indices)} features:")
    for rank, idx in enumerate(feature_indices):
        logger.info(f"    [{rank+1:2d}] index={idx:3d}: {feature_names[idx]}")

    # Preprocess (StandardScaler)
    logger.info("  Preprocessing (StandardScaler)...")
    X_scaled = dataset.preprocess(X_sel, fit=True)

    # Save feature config
    config_path = str(PROJECT_ROOT / "model" / "feature_config.json")
    dataset.save_feature_config(feature_indices, path=config_path)
    logger.info(f"  Saved feature config to {config_path}")

    # Split
    splits = dataset.get_splits(X_scaled, y_s, test_size=0.2, val_size=0.1)
    logger.info(f"  Splits: train={len(splits['X_train'])}, "
                f"val={len(splits['X_val'])}, test={len(splits['X_test'])}")

    # Create loaders
    loaders = dataset.to_torch_loaders(splits, batch_size=batch_size)

    return loaders, feature_indices, dataset


# ─────────────────────── Main Pipeline ───────────────────────────────

def run_training_pipeline(
    loaders: Dict[str, DataLoader],
    input_dim: int = 20,
    hidden_dims: List[int] = None,
    output_dim: int = 11,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, nn.Module]:
    """
    Train all three model variants. Returns dict of trained models.
    """
    if hidden_dims is None:
        hidden_dims = [32, 16]

    trained_models = {}
    hd_str = str(hidden_dims)

    # ── 1. FP32 Baseline ──
    logger.info("=" * 60)
    logger.info("TRAINING: FP32 Baseline")
    logger.info("=" * 60)

    fp32_model = FP32MLP(input_dim, hidden_dims, output_dim)
    fp32_metrics = train_model(
        fp32_model, loaders, epochs=epochs, lr=lr, device=device,
        model_name="FP32"
    )
    fp32_size = fp32_model.model_size_bytes()
    log_result(
        phase=3, experiment="baseline_fp32", model_type="fp32",
        architecture="MLP", hidden_dims=hd_str,
        num_params=fp32_model.count_parameters()["total"],
        metrics=fp32_metrics, model_size_bytes=fp32_size,
        notes="Standard FP32 MLP baseline"
    )
    trained_models["fp32"] = fp32_model

    # ── 2. INT8 Quantized ──
    logger.info("=" * 60)
    logger.info("QUANTIZING: INT8 (post-training dynamic from FP32)")
    logger.info("=" * 60)

    int8_model, int8_size = quantize_int8(fp32_model)
    int8_metrics = evaluate(
        int8_model, loaders["test"],
        nn.CrossEntropyLoss(), device
    )
    logger.info(
        f"  INT8 TEST: acc={int8_metrics['accuracy']:.4f} "
        f"prec={int8_metrics['precision']:.4f} "
        f"rec={int8_metrics['recall']:.4f} "
        f"f1={int8_metrics['f1']:.4f}"
    )
    log_result(
        phase=3, experiment="quantized_int8", model_type="int8",
        architecture="MLP", hidden_dims=hd_str,
        num_params=fp32_model.count_parameters()["total"],
        metrics=int8_metrics, model_size_bytes=int8_size,
        notes="Post-training dynamic INT8 from FP32 baseline"
    )
    trained_models["int8"] = int8_model

    # ── 3. Ternary ──
    logger.info("=" * 60)
    logger.info("TRAINING: Ternary (1.58-bit, 2-bit packed)")
    logger.info("=" * 60)

    ternary_model = TernaryMLP(
        input_dim, hidden_dims, output_dim,
        ternary_output=True, use_rmsnorm=True
    )
    ternary_metrics = train_model(
        ternary_model, loaders, epochs=epochs, lr=lr, device=device,
        model_name="Ternary"
    )
    ternary_size = ternary_model.estimate_packed_size()["total_bytes"]
    ternary_flash = ternary_size + 8000 + 2048 + 1500  # model + engine + bootloader + serial
    ternary_ram = ternary_model.estimate_ram_usage()["total_bytes"] + 64  # + serial buffer
    log_result(
        phase=3, experiment="ternary_2bit", model_type="ternary",
        architecture="TernaryMLP", hidden_dims=hd_str,
        num_params=ternary_model.count_parameters()["total"],
        metrics=ternary_metrics, model_size_bytes=ternary_size,
        flash_bytes=str(ternary_flash), ram_bytes=str(ternary_ram),
        notes="Ternary with RMSNorm, 2-bit packing, unified PROGMEM"
    )
    trained_models["ternary"] = ternary_model

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'Model':<12} {'Accuracy':>10} {'F1':>10} {'Size (B)':>10}")
    logger.info(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    logger.info(f"  {'FP32':<12} {fp32_metrics['accuracy']:>10.4f} {fp32_metrics['f1']:>10.4f} {fp32_size:>10,}")
    logger.info(f"  {'INT8':<12} {int8_metrics['accuracy']:>10.4f} {int8_metrics['f1']:>10.4f} {int8_size:>10,}")
    logger.info(f"  {'Ternary':<12} {ternary_metrics['accuracy']:>10.4f} {ternary_metrics['f1']:>10.4f} {ternary_size:>10,}")

    return trained_models


# ─────────────────────── Ablation Sweep ──────────────────────────────

def run_ablation_sweep(
    loaders: Dict[str, DataLoader],
    input_dim: int = 20,
    output_dim: int = 11,
    epochs: int = 30,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> List[Dict]:
    """
    Hidden-width ablation: vary hidden dimensions, measure accuracy vs size.
    Tests multiple width configurations for the ternary model.
    """
    configs = [
        [16, 8],
        [24, 12],
        [32, 16],     # production
        [48, 24],
        [64, 32],
    ]

    logger.info("=" * 60)
    logger.info("ABLATION SWEEP: Hidden Width vs Performance")
    logger.info("=" * 60)

    results = []
    for hidden_dims in configs:
        hd_str = str(hidden_dims)
        logger.info(f"\n--- Ablation: hidden_dims={hd_str} ---")

        model = TernaryMLP(
            input_dim, hidden_dims, output_dim,
            ternary_output=True, use_rmsnorm=True
        )
        metrics = train_model(
            model, loaders, epochs=epochs, lr=lr, device=device,
            model_name=f"Ternary-{hd_str}", patience=8
        )
        packed = model.estimate_packed_size()
        ram = model.estimate_ram_usage()

        result = {
            "hidden_dims": hidden_dims,
            "metrics": metrics,
            "packed_size": packed,
            "ram": ram,
        }
        results.append(result)

        log_result(
            phase=3,
            experiment=f"ablation_{'_'.join(str(h) for h in hidden_dims)}",
            model_type="ternary",
            architecture="TernaryMLP",
            hidden_dims=hd_str,
            num_params=model.count_parameters()["total"],
            metrics=metrics,
            model_size_bytes=packed["total_bytes"],
            flash_bytes=str(packed["total_bytes"] + 8000 + 2048 + 1500),
            ram_bytes=str(ram["total_bytes"] + 64),
            notes=f"Ablation sweep: {hd_str}"
        )

    # Print ablation summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'Hidden Dims':<14} {'Acc':>8} {'F1':>8} {'Flash(B)':>10} {'RAM(B)':>8} {'Fits?':>6}")
    logger.info(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")
    for r in results:
        hd = str(r["hidden_dims"])
        m = r["metrics"]
        p = r["packed_size"]
        ram = r["ram"]
        fits = "✅" if ram["fits"] else "❌"
        logger.info(
            f"  {hd:<14} {m['accuracy']:>8.4f} {m['f1']:>8.4f} "
            f"{p['total_bytes']:>10,} {ram['total_bytes']:>8,} {fits:>6}"
        )

    return results


# ─────────────────────── Main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TernaryGuard Phase 3 Training")
    parser.add_argument("--data-dir", type=str, default="data/raw/nbaiot")
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--ablation-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples-per-class", type=int, default=50000)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")  # ATmega target = CPU-only training
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch: {torch.__version__}")

    # Load data
    loaders, feature_indices, dataset = load_and_prepare_data(
        data_dir=args.data_dir,
        n_features=args.n_features,
        max_samples_per_class=args.max_samples_per_class,
        batch_size=args.batch_size,
    )

    input_dim = args.n_features
    output_dim = 11  # N-BaIoT: 10 attacks + benign
    hidden_dims = [32, 16]  # Production architecture

    if not args.ablation_only and not args.export_only:
        # Train all three variants
        trained = run_training_pipeline(
            loaders, input_dim=input_dim, hidden_dims=hidden_dims,
            output_dim=output_dim, epochs=args.epochs, lr=args.lr,
            device=device,
        )

        # Save ternary checkpoint
        CHECKPOINTS_DIR.mkdir(exist_ok=True)
        ckpt_path = CHECKPOINTS_DIR / "ternary_best.pt"
        torch.save(trained["ternary"].state_dict(), ckpt_path)
        logger.info(f"Saved ternary checkpoint: {ckpt_path}")

    if not args.no_ablation and not args.export_only:
        # Ablation sweep
        ablation_results = run_ablation_sweep(
            loaders, input_dim=input_dim, output_dim=output_dim,
            epochs=args.ablation_epochs, lr=args.lr, device=device,
        )

    # Export weights for best model (production architecture)
    logger.info("=" * 60)
    logger.info("EXPORTING: model_weights.h")
    logger.info("=" * 60)

    # Load or train production ternary model
    ternary_model = TernaryMLP(
        input_dim, hidden_dims, output_dim,
        ternary_output=True, use_rmsnorm=True
    )
    ckpt_path = CHECKPOINTS_DIR / "ternary_best.pt"
    if ckpt_path.exists():
        ternary_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        logger.info(f"  Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning("  No checkpoint found, exporting untrained weights!")

    export_path = str(PROJECT_ROOT / "model_weights.h")
    ternary_model.export_weights_header(path=export_path)
    logger.info(f"  Exported to: {export_path}")

    # Verify consistency with feature_config.json
    config_path = str(PROJECT_ROOT / "model" / "feature_config.json")
    if os.path.exists(config_path):
        config = NBaIoTDataset.load_feature_config(config_path)
        n_config_features = len(config["selected_features"])
        logger.info(f"  feature_config.json: {n_config_features} features")
        assert n_config_features == input_dim, (
            f"Feature config has {n_config_features} features but model expects {input_dim}"
        )
        logger.info("  ✅ model_weights.h input dim matches feature_config.json")
    else:
        logger.warning(f"  feature_config.json not found at {config_path}")

    # Print final flash/RAM budget
    packed = ternary_model.estimate_packed_size()
    ram = ternary_model.estimate_ram_usage()
    logger.info(f"  Flash: {packed['total_bytes']}B model → ~{packed['total_bytes']+11548}B total")
    logger.info(f"  RAM: {ram['total_bytes']}B model + 64B serial = {ram['total_bytes']+64}B total")
    logger.info(f"  Fits ATmega328P: flash={'✅' if packed['total_bytes'] < 32768 else '❌'} "
                f"ram={'✅' if ram['fits'] else '❌'}")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 3 complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
