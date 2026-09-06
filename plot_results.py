#!/usr/bin/env python3
"""
TernaryGuard — Experiment Results Plotting Script

Reads results.csv and generates publication-quality comparison plots:
  1. Accuracy vs Quantization (FP32 / INT8 / Ternary)
  2. Latency comparison across engines (Software / Arduino / FPGA)
  3. Resource usage (Flash, RAM, LUTs) across engines
  4. Power consumption comparison

Usage:
    python plot_results.py                    # plots all available data
    python plot_results.py --phase 3          # plots only Phase 3 data
    python plot_results.py --output docs/benchmarks/  # save to specific dir
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────── Configuration ────────────────

RESULTS_FILE = Path(__file__).parent / "results.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "docs" / "benchmarks"

COLORS = {
    "fp32": "#2196F3",
    "int8": "#FF9800",
    "ternary": "#4CAF50",
    "software": "#2196F3",
    "arduino": "#FF9800",
    "fpga": "#F44336",
}

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

sns.set_style("whitegrid")


def load_results(path: Path = RESULTS_FILE) -> pd.DataFrame:
    """Load and validate results CSV."""
    if not path.exists():
        print(f"Error: {path} not found. Run experiments first.")
        sys.exit(1)
    df = pd.read_csv(path)
    if df.empty:
        print("Warning: results.csv is empty. No plots to generate.")
        sys.exit(0)
    return df


def plot_accuracy_vs_quantization(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: accuracy/F1 for FP32 vs INT8 vs Ternary (Phase 3 data)."""
    phase3 = df[df["phase"] == 3]
    if phase3.empty:
        print("  Skipping accuracy plot — no Phase 3 data yet.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["accuracy", "f1_score"]):
        for model_type in ["fp32", "int8", "ternary"]:
            subset = phase3[phase3["model_type"] == model_type]
            if not subset.empty:
                ax.bar(
                    model_type.upper(),
                    subset[metric].values[0],
                    color=COLORS.get(model_type, "#999"),
                    edgecolor="black",
                    linewidth=0.5,
                )
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Quantization Level")

    fig.suptitle("TernaryGuard — Accuracy vs Quantization", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_quantization.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ accuracy_vs_quantization.png")


def plot_latency_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: inference latency across Software / Arduino / FPGA."""
    engines = df[df["phase"].isin([4, 5, 6]) & df["inference_latency_us"].notna()]
    if engines.empty:
        print("  Skipping latency plot — no engine data yet.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    engine_map = {4: "Software", 5: "Arduino", 6: "FPGA"}
    for phase_num, label in engine_map.items():
        subset = engines[engines["phase"] == phase_num]
        if not subset.empty:
            color_key = label.lower()
            ax.bar(
                label,
                subset["inference_latency_us"].values[0],
                color=COLORS.get(color_key, "#999"),
                edgecolor="black",
                linewidth=0.5,
            )

    ax.set_ylabel("Inference Latency (μs)")
    ax.set_title("TernaryGuard — Inference Latency by Engine", fontweight="bold")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_dir / "latency_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ latency_comparison.png")


def plot_resource_usage(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: flash/RAM (Arduino) and LUTs/FFs (FPGA)."""
    engines = df[df["phase"].isin([5, 6])]
    if engines.empty:
        print("  Skipping resource plot — no hardware engine data yet.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Arduino resources
    arduino = engines[engines["phase"] == 5]
    if not arduino.empty:
        resources = {
            "Flash": arduino["flash_bytes"].values[0],
            "RAM": arduino["ram_bytes"].values[0],
        }
        axes[0].bar(
            resources.keys(),
            resources.values(),
            color=[COLORS["arduino"]] * 2,
            edgecolor="black",
            linewidth=0.5,
        )
        axes[0].set_title("Arduino Nano — Resource Usage")
        axes[0].set_ylabel("Bytes")
        # Add limit lines
        axes[0].axhline(y=32768, color="red", linestyle="--", label="Flash Limit (32KB)")
        axes[0].axhline(y=2048, color="darkred", linestyle="--", label="RAM Limit (2KB)")
        axes[0].legend()

    # FPGA resources
    fpga = engines[engines["phase"] == 6]
    if not fpga.empty:
        resources = {
            "LUTs": fpga["lut_count"].values[0],
            "FFs": fpga["ff_count"].values[0],
            "DSPs": fpga["dsp_count"].values[0],
        }
        resources = {k: v for k, v in resources.items() if pd.notna(v)}
        if resources:
            axes[1].bar(
                resources.keys(),
                resources.values(),
                color=[COLORS["fpga"]] * len(resources),
                edgecolor="black",
                linewidth=0.5,
            )
            axes[1].set_title("FPGA (Basys 3) — Resource Utilization")
            axes[1].set_ylabel("Count")

    fig.suptitle("TernaryGuard — Hardware Resource Usage", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "resource_usage.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ resource_usage.png")


def plot_power_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: estimated power across engines."""
    engines = df[df["phase"].isin([4, 5, 6]) & df["power_mw"].notna()]
    if engines.empty:
        print("  Skipping power plot — no power data yet.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    engine_map = {4: "Software", 5: "Arduino", 6: "FPGA"}
    for phase_num, label in engine_map.items():
        subset = engines[engines["phase"] == phase_num]
        if not subset.empty:
            ax.bar(
                label,
                subset["power_mw"].values[0],
                color=COLORS.get(label.lower(), "#999"),
                edgecolor="black",
                linewidth=0.5,
            )

    ax.set_ylabel("Estimated Power (mW)")
    ax.set_title("TernaryGuard — Power Consumption by Engine", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "power_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ power_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Plot TernaryGuard experiment results")
    parser.add_argument("--phase", type=int, help="Filter to a specific phase")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("TernaryGuard — Generating benchmark plots")
    print(f"  Source: {RESULTS_FILE}")
    print(f"  Output: {output_dir}")
    print()

    df = load_results()

    if args.phase is not None:
        df = df[df["phase"] == args.phase]

    plot_accuracy_vs_quantization(df, output_dir)
    plot_latency_comparison(df, output_dir)
    plot_resource_usage(df, output_dir)
    plot_power_comparison(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
