#!/usr/bin/env python3
"""
TernaryGuard — Sanity Check: Weight Distribution Histograms

Generates plots showing weight distributions before and after ternary
quantization for visual verification that the quantization is working
as expected.

Produces:
    research/weight_histograms.png  — 2×2 grid:
        [pre-quant histogram]  [post-quant bar chart]
        [pre-quant per-layer]  [weight distribution pie]

Usage:
    python research/sanity_check_weights.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.ternary_linear import TernaryLinear, TernaryMLP

# ──────────────── Configuration ────────────────

OUTPUT_DIR = Path(__file__).parent
SEED = 42
ARCHITECTURE = {
    "input_dim": 20,
    "hidden_dims": [16, 8],
    "output_dim": 2,
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.facecolor": "white",
})


def main():
    torch.manual_seed(SEED)

    # Build model
    model = TernaryMLP(**ARCHITECTURE)
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Parameter counts: {model.count_parameters()}")
    print(f"Packed size estimate: {model.estimate_packed_size()}")
    print()

    # Collect weights from all TernaryLinear layers
    layer_names = []
    pre_quant_weights = []
    post_quant_weights = []
    distributions = []

    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            w_fp = module.weight.detach().clone().numpy().flatten()
            w_q, scale = module.get_ternary_weights()
            w_q_np = w_q.numpy().flatten()
            dist = module.weight_distribution()

            layer_names.append(name or "root")
            pre_quant_weights.append(w_fp)
            post_quant_weights.append(w_q_np)
            distributions.append(dist)

            print(f"Layer: {name}")
            print(f"  Scale (absmean): {scale:.6f}")
            print(f"  Distribution: -1={dist[-1]}, 0={dist[0]}, +1={dist[1]}")
            total = dist[-1] + dist[0] + dist[1]
            print(f"  Fractions: -1={dist[-1]/total:.1%}, 0={dist[0]/total:.1%}, +1={dist[1]/total:.1%}")
            print()

    # ──────────────── Plot ────────────────

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0) Pre-quantization histogram — all layers combined
    ax = axes[0, 0]
    all_pre = np.concatenate(pre_quant_weights)
    ax.hist(all_pre, bins=50, color="#2196F3", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, label="zero")
    mean_abs = np.mean(np.abs(all_pre))
    ax.axvline(x=0.5 * mean_abs, color="green", linestyle="--", linewidth=1.5, label=f"threshold (+0.5α = {0.5*mean_abs:.3f})")
    ax.axvline(x=-0.5 * mean_abs, color="green", linestyle="--", linewidth=1.5, label=f"threshold (-0.5α = {-0.5*mean_abs:.3f})")
    ax.set_title("Pre-Quantization Weight Distribution (FP32)")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # (0,1) Post-quantization bar chart — all layers combined
    ax = axes[0, 1]
    all_post = np.concatenate(post_quant_weights)
    counts = {-1: np.sum(all_post == -1), 0: np.sum(all_post == 0), 1: np.sum(all_post == 1)}
    total = sum(counts.values())
    bars = ax.bar(
        ["-1", "0", "+1"],
        [counts[-1], counts[0], counts[1]],
        color=["#F44336", "#9E9E9E", "#4CAF50"],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, val in zip(bars, [counts[-1], counts[0], counts[1]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{val}\n({val/total:.1%})", ha="center", va="bottom", fontsize=10)
    ax.set_title("Post-Quantization Weight Distribution (Ternary)")
    ax.set_xlabel("Quantized Weight Value")
    ax.set_ylabel("Count")

    # (1,0) Per-layer pre-quant histograms overlaid
    ax = axes[1, 0]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
    for i, (name, weights) in enumerate(zip(layer_names, pre_quant_weights)):
        ax.hist(weights, bins=30, alpha=0.5, color=colors[i % len(colors)],
                label=name, edgecolor="black", linewidth=0.3)
    ax.set_title("Per-Layer FP32 Weight Distributions")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # (1,1) Per-layer ternary distribution stacked bar
    ax = axes[1, 1]
    x_pos = np.arange(len(layer_names))
    bar_width = 0.25
    for i, (name, dist) in enumerate(zip(layer_names, distributions)):
        total_l = dist[-1] + dist[0] + dist[1]
        ax.bar(x_pos[i] - bar_width, dist[-1] / total_l * 100, bar_width,
               color="#F44336", label="-1" if i == 0 else None)
        ax.bar(x_pos[i], dist[0] / total_l * 100, bar_width,
               color="#9E9E9E", label="0" if i == 0 else None)
        ax.bar(x_pos[i] + bar_width, dist[1] / total_l * 100, bar_width,
               color="#4CAF50", label="+1" if i == 0 else None)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Per-Layer Ternary Distribution (%)")
    ax.legend()

    fig.suptitle("TernaryGuard — Weight Quantization Sanity Check", fontsize=14, fontweight="bold")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "weight_histograms.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
