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

import matplotlib
matplotlib.use("Agg")  # File-only plotting also works in headless environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})


def invalidated_rows(df: pd.DataFrame) -> pd.Series:
    """The notes marker INVALIDATED is case-insensitive and applies to the whole input."""
    return df.get("notes", pd.Series("", index=df.index)).fillna("").astype(str).str.contains(
        "INVALIDATED", case=False, regex=False)


def save_figure(fig, path, df):
    """Every plot sourced from invalidated data carries a visible warning."""
    if df.attrs.get("invalidated_source", False) or invalidated_rows(df).any():
        fig.text(0.5, 0.5, "INVALIDATED — DO NOT USE", ha="center", va="center",
                 rotation=25, fontsize=28, weight="bold", color="red", alpha=0.7,
                 zorder=1000)
    fig.savefig(path, bbox_inches="tight")


def load_results(path: Path = RESULTS_FILE, allow_invalidated: bool = False) -> pd.DataFrame:
    """Load and validate results CSV."""
    if not path.exists():
        print(f"Error: {path} not found. Run experiments first.")
        sys.exit(1)
    df = pd.read_csv(path)
    if df.empty:
        print("Warning: results.csv is empty. No plots to generate.")
        sys.exit(0)
    invalid = invalidated_rows(df)
    if invalid.any() and not allow_invalidated:
        runs = (df.loc[invalid, "run_id"].dropna().astype(str).unique().tolist()
                if "run_id" in df else [])
        identity = ", ".join(runs) or str(path)
        raise ValueError(f"Refusing INVALIDATED results: {identity}. "
                         "Use --allow-invalidated only for watermarked historical plots.")
    df.attrs["invalidated_source"] = bool(invalid.any())
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
            subset = phase3[(phase3["model_type"] == model_type) & ~phase3["experiment"].str.startswith("ablation_")]
            if not subset.empty:
                ax.bar(
                    ("INT8 (sim.)" if model_type == "int8" else model_type.upper()),
                    subset[metric].values[0],
                    color=COLORS.get(model_type, "#999"),
                    edgecolor="black",
                    linewidth=0.5,
                )
        metric_label = "Weighted F1" if metric == "f1_score" else "Accuracy"
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{metric_label} vs Weight Quantization")

    fig.suptitle("TernaryGuard — Accuracy vs Quantization", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "accuracy_vs_quantization.png", df)
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
    save_figure(fig, output_dir / "latency_comparison.png", df)
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
    save_figure(fig, output_dir / "resource_usage.png", df)
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
    save_figure(fig, output_dir / "power_comparison.png", df)
    plt.close(fig)
    print("  ✓ power_comparison.png")


def plot_ablation_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot: F1 score vs hidden layer width for each quantization type (Phase 3.3)."""
    phase3 = df[df["phase"] == 3]
    if phase3.empty or "hidden_dims" not in phase3.columns:
        print("  Skipping ablation plot — no Phase 3 data yet.")
        return

    # Need multiple hidden_dims entries to make a sweep plot
    ablation = phase3[phase3["experiment"].str.startswith("ablation_") & phase3["f1_score"].notna()]
    if len(ablation) < 2:
        print("  Skipping ablation plot — need ≥2 hidden_dims entries for a sweep.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_type in ["fp32", "int8", "ternary"]:
        subset = ablation[ablation["model_type"] == model_type].sort_values("hidden_dims")
        if not subset.empty:
            ax.plot(
                subset["hidden_dims"].astype(str),
                subset["f1_score"],
                marker="o",
                linewidth=2,
                markersize=8,
                label=model_type.upper(),
                color=COLORS.get(model_type, "#999"),
            )

    ax.set_xlabel("Hidden Layer Width")
    ax.set_ylabel("Weighted F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("TernaryGuard — Architecture Ablation (Hidden Width Sweep)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "ablation_sweep.png", df)
    plt.close(fig)
    print("  ✓ ablation_sweep.png")


def main():
    parser = argparse.ArgumentParser(description="Plot TernaryGuard experiment results")
    parser.add_argument("--results", type=Path, default=RESULTS_FILE, help="Results CSV to plot")
    parser.add_argument("--run-id", help="Run to plot; defaults to the latest recorded run")
    parser.add_argument("--phase", type=int, help="Filter to a specific phase")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for plots",
    )
    parser.add_argument("--allow-invalidated", action="store_true",
                        help="Allow historical INVALIDATED notes only with a warning on every plot")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("TernaryGuard — Generating benchmark plots")
    print(f"  Source: {args.results}")
    print(f"  Output: {output_dir}")
    print()

    try:
        df = load_results(args.results, allow_invalidated=args.allow_invalidated)
    except ValueError as exc:
        parser.error(str(exc))

    if "run_id" in df.columns:
        run_id = args.run_id or df["run_id"].dropna().iloc[-1]
        df = df[df["run_id"] == run_id]
        if df.empty:
            parser.error(f"No results for run {run_id}")
        print(f"  Run: {run_id}")

    if args.phase is not None:
        df = df[df["phase"] == args.phase]

    plot_accuracy_vs_quantization(df, output_dir)
    plot_ablation_sweep(df, output_dir)
    plot_latency_comparison(df, output_dir)
    plot_resource_usage(df, output_dir)
    plot_power_comparison(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
