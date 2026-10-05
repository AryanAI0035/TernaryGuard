#!/usr/bin/env python3
"""Explore N-BaIoT feature distributions and estimated model memory.

Writes class counts, mutual-information rankings, correlations, histograms
and t-SNE plots. When CSVs are absent, it uses synthetic demonstration data;
those plots cannot support measured dataset or classifier claims. Memory
figures are analytical budgets, not measured Nano flash or peak SRAM.

Usage:
    python research/explore_nbaiot.py --data-dir data/raw/nbaiot --n-features 20

The saved images in research/eda/ are historical exploratory outputs. See
that directory's README for their provenance limits.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path for model package imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model.data_pipeline import NBaIoTDataset, get_feature_names, get_label_names
from model.ternary_linear import TernaryMLP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("explore_nbaiot")

# Plot styling constants
DPI = 150
plt.rcParams.update({
    "figure.dpi": DPI,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Data Fallback
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_nbaiot_data(
    n_samples_per_class: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic N-BaIoT data with 115 features and 11 classes.

    Used when the raw CSV dataset directory is missing or empty, allowing the
    entire exploratory pipeline, visualizations, and feasibility checks to run.
    Simulates right-skewed, positive network packet and jitter statistics with
    distinct distribution shifts for benign traffic and individual attack types.

    Args:
        n_samples_per_class: Number of samples to generate per class.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (features_df, labels_series)
    """
    logger.info("Generating synthetic N-BaIoT dataset for demonstration...")
    rng = np.random.default_rng(random_state)
    feature_names = get_feature_names()
    label_map = get_label_names()
    n_classes = len(label_map)
    n_features = len(feature_names)

    total_samples = n_samples_per_class * n_classes
    data = np.zeros((total_samples, n_features), dtype=np.float32)
    labels = np.zeros(total_samples, dtype=np.int64)

    # Base feature properties: scale and mean depending on stream type
    base_scales = []
    for feat in feature_names:
        if "weight" in feat:
            base_scales.append(1.5)
        elif "mean" in feat:
            base_scales.append(3.0)
        elif "std" in feat:
            base_scales.append(1.0)
        elif "radius" in feat or "magnitude" in feat:
            base_scales.append(2.0)
        elif "jit" in feat:
            base_scales.append(0.5)
        else:
            base_scales.append(1.0)
    base_scales = np.array(base_scales, dtype=np.float32)

    for c in range(n_classes):
        start_idx = c * n_samples_per_class
        end_idx = start_idx + n_samples_per_class
        labels[start_idx:end_idx] = c

        if c == 0:
            # Benign: lower traffic rates, low packet variance, modest means
            locs = np.full(n_features, 0.5, dtype=np.float32)
            scales = base_scales * 0.4
        else:
            # Attack types: specific signature surges in short/long windows
            locs = np.full(n_features, 1.2, dtype=np.float32)
            # Add class-specific signature shifts across different feature blocks
            shift_block = (c * 10) % n_features
            locs[shift_block : min(shift_block + 15, n_features)] += 2.0
            scales = base_scales * (0.8 + 0.1 * (c % 4))

        # Generate strictly non-negative log-normal traffic samples
        class_samples = rng.lognormal(mean=locs, sigma=scales * 0.5, size=(n_samples_per_class, n_features))
        # Add realistic noise and clip
        data[start_idx:end_idx] = np.clip(class_samples, 0.0, 1e6)

    features_df = pd.DataFrame(data, columns=feature_names)
    labels_series = pd.Series(labels, name="label")

    logger.info(
        "Generated synthetic dataset: %d samples, %d features across %d classes.",
        len(features_df),
        n_features,
        n_classes,
    )
    return features_df, labels_series


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate_dataset(
    data_dir: str,
    max_samples_per_class: int = 50000,
    synthetic_samples_per_class: int = 500,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    """
    Load data using NBaIoTDataset or fall back to synthetic data if missing.

    Args:
        data_dir: Path to raw N-BaIoT CSV directory.
        max_samples_per_class: Cap for loading from disk per class.
        synthetic_samples_per_class: Samples per class if synthetic generation is needed.
        random_state: Seed for synthetic generation.

    Returns:
        Tuple[pd.DataFrame, pd.Series, bool]: (features_df, labels_series, is_synthetic)
    """
    data_path = Path(data_dir)
    is_synthetic = False

    if data_path.exists():
        logger.info("Loading N-BaIoT dataset from '%s'...", data_dir)
        dataset_loader = NBaIoTDataset(
            data_dir=str(data_path),
            max_samples_per_class=max_samples_per_class,
        )
        features_df, labels_series = dataset_loader.load_data()
    else:
        features_df, labels_series = pd.DataFrame(), pd.Series(dtype=int)

    # If missing directory or no CSV files were parsed
    if len(features_df) == 0:
        logger.warning(
            "Raw N-BaIoT data directory '%s' does not exist or contains no valid CSVs.",
            data_dir,
        )
        features_df, labels_series = generate_synthetic_nbaiot_data(
            n_samples_per_class=synthetic_samples_per_class,
            random_state=random_state,
        )
        is_synthetic = True

    return features_df, labels_series, is_synthetic


# ─────────────────────────────────────────────────────────────────────────────
# Mutual Information Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: int = 20000,
    random_state: int = 42,
) -> pd.Series:
    """
    Compute mutual information scores for all features against target labels.

    Uses a stratified subsample if the dataset exceeds max_samples to ensure
    fast execution while maintaining statistical significance.

    Args:
        X: Feature DataFrame (115 columns).
        y: Class labels Series.
        max_samples: Max samples to use for MI computation.
        random_state: Random state for subsampling.

    Returns:
        pd.Series: Sorted mutual information scores indexed by feature name.
    """
    logger.info("Computing mutual information across all %d features...", X.shape[1])
    n_samples = len(X)

    if n_samples > max_samples:
        logger.info(
            "Subsampling from %d to %d samples for efficient mutual information calculation...",
            n_samples,
            max_samples,
        )
        X_sub, _, y_sub, _ = train_test_split(
            X,
            y,
            train_size=max_samples,
            stratify=y if len(np.unique(y)) > 1 else None,
            random_state=random_state,
        )
    else:
        X_sub, y_sub = X, y

    mi_array = mutual_info_classif(
        X_sub.values,
        y_sub.values,
        discrete_features=False,
        random_state=random_state,
    )
    mi_scores = pd.Series(mi_array, index=X.columns, name="mutual_info")
    return mi_scores.sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    y: pd.Series,
    output_path: Path,
    label_map: Dict[int, str],
) -> None:
    """
    Plot 1: Class distribution bar chart.

    Displays sample count per class (benign + 10 attack types).
    Benign is highlighted in green, while attack classes are rendered in
    distinct red/coral shades.

    Args:
        y: Class labels Series.
        output_path: Destination path for saving PNG.
        label_map: Mapping from integer label to human-readable name.
    """
    logger.info("Generating Plot 1: Class distribution -> %s", output_path)
    class_counts = y.value_counts().sort_index()

    # Build ordered list of all classes defined in label_map
    all_class_ids = sorted(label_map.keys())
    counts = [int(class_counts.get(cid, 0)) for cid in all_class_ids]
    names = [label_map[cid] for cid in all_class_ids]

    # Color mapping: Benign is green (#2ca02c), 10 attack types in red/coral shades
    attack_shades = [
        "#ff6b6b", "#ee5253", "#ff7675", "#d63031", "#e17055",
        "#e84118", "#c23616", "#b71540", "#9c0000", "#7a0000",
    ]
    colors = ["#2ca02c"] + attack_shades[: len(all_class_ids) - 1]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(names, counts, color=colors, edgecolor="black", linewidth=0.6, width=0.65)

    # Annotate counts and percentages on top of bars
    total_samples = max(sum(counts), 1)
    max_count = max(counts) if counts else 1
    for bar, count in zip(bars, counts):
        pct = (count / total_samples) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_count * 0.015),
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="medium",
        )

    ax.set_ylim(0, max_count * 1.18)
    ax.set_title("N-BaIoT Dataset Class Distribution (Benign vs 10 Attack Types)", fontweight="bold", pad=12)
    ax.set_xlabel("Class Label", labelpad=8)
    ax.set_ylabel("Number of Samples", labelpad=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")

    # Legend for category interpretation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="Benign Traffic"),
        Patch(facecolor="#d63031", edgecolor="black", label="Botnet Attacks (Mirai / Bashlite)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    mi_scores: pd.Series,
    output_path: Path,
    top_k: int = 30,
) -> List[str]:
    """
    Plot 2: Horizontal bar chart of top features ranked by mutual information.

    Args:
        mi_scores: Mutual information scores sorted descending.
        output_path: Destination path for saving PNG.
        top_k: Number of top features to plot (default: 30).

    Returns:
        List[str]: Names of the top_k features.
    """
    logger.info("Generating Plot 2: Feature importance (top %d) -> %s", top_k, output_path)
    top_features = mi_scores.head(top_k)
    feature_names = top_features.index.tolist()
    scores = top_features.values

    # Invert for horizontal bar chart (highest score on top)
    y_pos = np.arange(len(top_features))[::-1]

    fig, ax = plt.subplots(figsize=(10, 10))
    # Palette from dark blue to lighter teal
    palette = sns.color_palette("mako_r", n_colors=top_k)
    bars = ax.barh(y_pos, scores, color=palette, edgecolor="black", linewidth=0.5, height=0.7)

    # Value labels beside each bar
    max_score = max(scores) if len(scores) > 0 else 1.0
    for bar, score in zip(bars, scores):
        ax.text(
            score + (max_score * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=8.5)
    ax.set_xlim(0, max_score * 1.15)
    ax.set_xlabel("Mutual Information Score (nats)", labelpad=8)
    ax.set_ylabel("N-BaIoT Feature Name", labelpad=8)
    ax.set_title(
        f"Top {top_k} Features Ranked by Mutual Information with Class Labels",
        fontweight="bold",
        pad=12,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return feature_names


def plot_correlation_matrix(
    X: pd.DataFrame,
    top_features: List[str],
    output_path: Path,
) -> None:
    """
    Plot 3: Correlation matrix heatmap of top 30 features.

    Annotations are turned off for clarity as requested.

    Args:
        X: Feature DataFrame.
        top_features: List of feature names to correlate (top 30).
        output_path: Destination path for saving PNG.
    """
    logger.info("Generating Plot 3: Correlation matrix heatmap -> %s", output_path)
    corr_df = X[top_features].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_df,
        annot=False,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        cbar_kws={"label": "Pearson Correlation Coefficient", "shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        f"Feature Correlation Matrix (Top {len(top_features)} MI-Selected Features)",
        fontweight="bold",
        pad=12,
    )
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: List[str],
    output_path: Path,
    grid_rows: int = 4,
    grid_cols: int = 5,
) -> None:
    """
    Plot 4: 4x5 grid of histograms for the top 20 selected features.

    Features are colored by class, simplified to binary (benign vs attack).
    Benign is green, attack is red.

    Args:
        X: Feature DataFrame.
        y: Class labels Series.
        selected_features: List of features to plot (first 20 used).
        output_path: Destination path for saving PNG.
        grid_rows: Number of grid rows (4).
        grid_cols: Number of grid columns (5).
    """
    features_to_plot = selected_features[: grid_rows * grid_cols]
    n_plots = len(features_to_plot)
    logger.info("Generating Plot 4: Feature distributions grid (%d features) -> %s", n_plots, output_path)

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(18, 13))
    axes_flat = axes.flatten()

    is_benign = (y == 0).values
    is_attack = (y > 0).values

    for i, feat_name in enumerate(features_to_plot):
        ax = axes_flat[i]
        vals = X[feat_name].values
        benign_vals = vals[is_benign]
        attack_vals = vals[is_attack]

        # Use log1p transformation if values span multiple orders of magnitude
        min_v, max_v = vals.min(), vals.max()
        use_log = (max_v > 50) and (min_v >= 0)

        b_plot = np.log1p(benign_vals) if use_log else benign_vals
        a_plot = np.log1p(attack_vals) if use_log else attack_vals

        # Determine common bin edges
        all_plot = np.concatenate([b_plot, a_plot]) if len(b_plot) and len(a_plot) else vals
        bins = np.linspace(np.percentile(all_plot, 1), np.percentile(all_plot, 99), 30)

        if len(benign_vals) > 0:
            ax.hist(
                b_plot,
                bins=bins,
                density=True,
                alpha=0.6,
                color="#2ca02c",
                label="Benign" if i == 0 else None,
                edgecolor="black",
                linewidth=0.3,
            )
        if len(attack_vals) > 0:
            ax.hist(
                a_plot,
                bins=bins,
                density=True,
                alpha=0.5,
                color="#d62728",
                label="Attack" if i == 0 else None,
                edgecolor="black",
                linewidth=0.3,
            )

        # Truncate title cleanly if needed
        clean_title = feat_name
        if len(clean_title) > 22:
            clean_title = clean_title[:20] + ".."
        if use_log:
            clean_title += " [log1p]"

        ax.set_title(clean_title, fontsize=8.5, fontweight="semibold", pad=4)
        ax.set_ylabel("Density", fontsize=7.5)
        ax.tick_params(axis="both", which="major", labelsize=7)

    # Hide unused subplots if fewer than 20
    for j in range(n_plots, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Single global figure legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#2ca02c", edgecolor="black", alpha=0.65, label="Benign Traffic"),
        Patch(facecolor="#d62728", edgecolor="black", alpha=0.55, label="Attack Traffic (All Types)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        fontsize=10,
        frameon=True,
    )

    fig.suptitle(
        "Top 20 Feature Distributions: Benign vs Attack Traffic (Density Normalized)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_visualization(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    label_map: Dict[int, str],
    max_samples: int = 5000,
    random_state: int = 42,
) -> None:
    """
    Plot 5: t-SNE 2D projection of a random subsample (5,000 points max).

    Features are standardized prior to t-SNE projection. Points are colored
    by class label with Benign prominently distinct.

    Args:
        X: Feature DataFrame.
        y: Class labels Series.
        output_path: Destination path for saving PNG.
        label_map: Mapping from integer label to human-readable name.
        max_samples: Maximum number of points to sample (default 5000).
        random_state: Random state for reproducibility.
    """
    n_total = len(X)
    n_sample = min(max_samples, n_total)
    logger.info("Generating Plot 5: t-SNE 2D projection (%d points) -> %s", n_sample, output_path)

    # Subsample data stratified by class
    if n_sample < n_total:
        X_sub, _, y_sub, _ = train_test_split(
            X,
            y,
            train_size=n_sample,
            stratify=y if len(np.unique(y)) > 1 else None,
            random_state=random_state,
        )
    else:
        X_sub, y_sub = X, y

    # Standardize features before t-SNE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    # t-SNE projection
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    embeddings = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(11, 8))

    # Palette: distinct green for benign, qualitative colors for attacks
    unique_classes = sorted(np.unique(y_sub))
    palette_attacks = sns.color_palette("tab10", n_colors=10)
    class_colors: Dict[int, Any] = {0: "#2ca02c"}
    for idx, c in enumerate([c for c in unique_classes if c != 0]):
        class_colors[c] = palette_attacks[idx % len(palette_attacks)]

    for c in unique_classes:
        mask = (y_sub == c).values
        c_name = label_map.get(c, f"Class {c}")
        is_benign = (c == 0)
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[class_colors[c]],
            label=f"{c_name} (n={mask.sum()})",
            s=22 if is_benign else 14,
            alpha=0.75 if is_benign else 0.55,
            edgecolor="black" if is_benign else "none",
            linewidth=0.3 if is_benign else 0.0,
        )

    ax.set_title(
        f"t-SNE 2D Manifold Projection of N-BaIoT Traffic Patterns (n={n_sample:,})",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("t-SNE Dimension 1", labelpad=8)
    ax.set_ylabel("t-SNE Dimension 2", labelpad=8)
    ax.legend(
        title="Class Label",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=True,
        fontsize=8.5,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Embedded Memory Feasibility Estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_embedded_memory_budget(
    n_features: int,
) -> Dict[str, Any]:
    """
    Calculate theoretical and concrete embedded memory footprint for the
    selected feature set on the ATmega328P (32KB Flash, 2KB RAM).

    Evaluates:
    - Raw feature vector storage (FP32, Int16 fixed-point, Int8).
    - Model Flash and RAM usage using TernaryMLP with architectures:
      - Binary classification: [n_features -> 16 -> 8 -> 2]
      - Multi-class classification: [n_features -> 16 -> 8 -> 11]

    Args:
        n_features: Number of selected input features.

    Returns:
        Dict[str, Any]: Detailed memory breakdown dictionary.
    """
    # Raw feature vector sizes
    fp32_bytes = n_features * 4
    int16_bytes = n_features * 2
    int8_bytes = n_features * 1

    # Instantiate TernaryMLP models to pull exact parameter counts and budgets
    model_binary = TernaryMLP(input_dim=n_features, hidden_dims=[16, 8], output_dim=2)
    model_multiclass = TernaryMLP(input_dim=n_features, hidden_dims=[16, 8], output_dim=11)

    flash_bin = model_binary.estimate_packed_size()
    ram_bin = model_binary.estimate_ram_usage(activation_dtype_bytes=2, stack_reserve=128)

    flash_multi = model_multiclass.estimate_packed_size()
    ram_multi = model_multiclass.estimate_ram_usage(activation_dtype_bytes=2, stack_reserve=128)

    return {
        "n_features": n_features,
        "raw_vector": {
            "fp32_bytes": fp32_bytes,
            "int16_bytes": int16_bytes,
            "int8_bytes": int8_bytes,
        },
        "binary_model": {
            "architecture": f"[{n_features} -> 16 -> 8 -> 2]",
            "flash": flash_bin,
            "ram": ram_bin,
        },
        "multiclass_model": {
            "architecture": f"[{n_features} -> 16 -> 8 -> 11]",
            "flash": flash_multi,
            "ram": ram_multi,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary Table Output
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(
    total_samples: int,
    total_features: int,
    y: pd.Series,
    label_map: Dict[int, str],
    mi_scores: pd.Series,
    n_selected_features: int,
    memory_info: Dict[str, Any],
    is_synthetic: bool,
) -> None:
    """
    Print a structured, publication-grade summary table to stdout.

    Args:
        total_samples: Total number of records in dataset.
        total_features: Total raw features (115 in N-BaIoT).
        y: Target label Series.
        label_map: Label ID to string name mapping.
        mi_scores: Ranked mutual information scores.
        n_selected_features: Number of selected features.
        memory_info: Memory estimation dictionary.
        is_synthetic: Flag indicating whether synthetic fallback was used.
    """
    class_counts = y.value_counts().sort_index()
    benign_count = int(class_counts.get(0, 0))
    attack_count = total_samples - benign_count

    print()
    print("=" * 78)
    print("                     TERNARYGUARD: N-BaIoT EDA SUMMARY")
    if is_synthetic:
        print("          *** NOTE: Synthetic Sample Data (Raw CSVs Not Found) ***")
    print("=" * 78)
    print(f"Total Dataset Samples:   {total_samples:,}")
    print(f"Total Raw Features:      {total_features} (N-BaIoT multi-timeframe statistics)")
    print(f"Selected Feature Subset: {n_selected_features} features")
    print("-" * 78)

    # Class Breakdown Table
    print("1. Class Distribution Summary:")
    print(f"{'ID':<4} {'Class Name':<20} {'Category':<10} {'Samples':>10} {'Percentage':>12}")
    print("-" * 78)
    for cid in sorted(label_map.keys()):
        count = int(class_counts.get(cid, 0))
        pct = (count / total_samples * 100) if total_samples > 0 else 0.0
        cat = "Benign" if cid == 0 else "Attack"
        print(f"{cid:<4} {label_map[cid]:<20} {cat:<10} {count:>10,} {pct:>11.2f}%")
    print("-" * 78)
    print(f"Aggregate Benign:  {benign_count:>10,} ({benign_count / total_samples * 100:>5.2f}%)")
    print(f"Aggregate Attacks: {attack_count:>10,} ({attack_count / total_samples * 100:>5.2f}%)")
    print("-" * 78)

    # Selected Features Table
    print(f"2. Top {n_selected_features} Selected Features (Mutual Information Ranking):")
    print(f"{'Rank':<6} {'Feature Name':<38} {'MI Score (nats)':>15}")
    print("-" * 78)
    top_selected = mi_scores.head(n_selected_features)
    for rank, (feat_name, score) in enumerate(top_selected.items(), start=1):
        print(f"{rank:<6} {feat_name:<38} {score:>15.4f}")
    print("-" * 78)

    # Embedded Memory Budget
    raw_vec = memory_info["raw_vector"]
    bin_m = memory_info["binary_model"]
    multi_m = memory_info["multiclass_model"]

    print("3. Embedded Feasibility & Memory Estimate (Target: ATmega328P)")
    print("   Constraints: 32 KB Flash (32,768 bytes), 2 KB SRAM (2,048 bytes) @ 16 MHz")
    print("-" * 78)
    print("Feature Vector Buffer in RAM:")
    print(f"  - FP32 representation (4B/val):    {raw_vec['fp32_bytes']:>5} bytes ({raw_vec['fp32_bytes']/2048*100:>4.1f}% of RAM)")
    print(f"  - Int16 fixed-point (2B/val):      {raw_vec['int16_bytes']:>5} bytes ({raw_vec['int16_bytes']/2048*100:>4.1f}% of RAM)")
    print(f"  - Int8 quantized (1B/val):         {raw_vec['int8_bytes']:>5} bytes ({raw_vec['int8_bytes']/2048*100:>4.1f}% of RAM)")
    print()

    for label, m_dict in [("Binary Classifier", bin_m), ("11-Class Classifier", multi_m)]:
        fl = m_dict["flash"]
        rm = m_dict["ram"]
        print(f"Model Architecture: {m_dict['architecture']} ({label})")
        print(f"  - Ternary weights:                 {fl['ternary_weights']:>5} weights")
        print(f"  - Packed weight storage (2b):      {fl['packed_weight_bytes']:>5} bytes (in Flash via PROGMEM)")
        print(f"  - Total Flash required:            {fl['total_bytes']:>5} bytes ({fl['total_bytes']/32768*100:>4.2f}% of 32KB Flash)")
        print(f"  - Peak runtime RAM required:       {rm['total_bytes']:>5} bytes ({rm['total_bytes']/2048*100:>4.2f}% of 2KB RAM)")
        print(f"    * Ping-pong activation buffers:  {rm['activation_bytes']:>5} bytes")
        print(f"    * Input UART feature buffer:     {rm['input_buffer_bytes']:>5} bytes")
        print(f"    * Layer scaling factors (float): {rm['scale_bytes']:>5} bytes")
        print(f"    * RMSNorm gamma parameters:      {rm['rmsnorm_ram_bytes']:>5} bytes")
        print(f"    * Reserved stack & UART buffer:  {rm['stack_reserve']:>5} bytes")
        print(f"  - RAM Headroom:                    {rm['headroom_bytes']:>5} bytes remaining")
        print(f"  - Embedded Feasibility Verdict:    {'[FEASIBLE - FITS ON ATMEGA328P]' if rm['fits'] else '[EXCEEDS 2KB RAM]'}")
        print()

    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point & CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main CLI execution flow."""
    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis (EDA) for N-BaIoT dataset in TernaryGuard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/nbaiot",
        help="Path to directory containing N-BaIoT CSV files",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Number of top features to select for embedded model (target: 20-40)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/eda",
        help="Directory where generated plots will be saved",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=50000,
        help="Maximum samples loaded per class when loading raw CSVs",
    )
    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=5000,
        help="Maximum number of points used for t-SNE 2D visualization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing TernaryGuard N-BaIoT EDA...")
    logger.info("Output directory: %s", output_dir)

    # 1. Ingest dataset with graceful synthetic fallback
    features_df, labels_series, is_synthetic = load_or_generate_dataset(
        data_dir=args.data_dir,
        max_samples_per_class=args.max_samples_per_class,
        random_state=args.seed,
    )

    label_map = get_label_names()

    # 2. Compute mutual information for feature ranking
    mi_scores = compute_mutual_information(
        features_df,
        labels_series,
        max_samples=20000,
        random_state=args.seed,
    )

    # 3. Generate all 5 plots
    # Plot 1: Class distribution
    plot_class_distribution(
        y=labels_series,
        output_path=output_dir / "class_distribution.png",
        label_map=label_map,
    )

    # Plot 2: Feature importance (top 30)
    top_30_features = plot_feature_importance(
        mi_scores=mi_scores,
        output_path=output_dir / "feature_importance.png",
        top_k=30,
    )

    # Plot 3: Feature correlation matrix (top 30)
    plot_correlation_matrix(
        X=features_df,
        top_features=top_30_features,
        output_path=output_dir / "correlation_matrix.png",
    )

    # Plot 4: Feature distribution histograms (top 20)
    plot_feature_distributions(
        X=features_df,
        y=labels_series,
        selected_features=top_30_features[:20],
        output_path=output_dir / "feature_distributions.png",
        grid_rows=4,
        grid_cols=5,
    )

    # Plot 5: t-SNE 2D projection
    plot_tsne_visualization(
        X=features_df,
        y=labels_series,
        output_path=output_dir / "tsne_visualization.png",
        label_map=label_map,
        max_samples=args.tsne_samples,
        random_state=args.seed,
    )

    # 4. Embedded feasibility & memory estimation
    memory_info = estimate_embedded_memory_budget(n_features=args.n_features)

    # 5. Summary Table
    print_summary_table(
        total_samples=len(features_df),
        total_features=features_df.shape[1],
        y=labels_series,
        label_map=label_map,
        mi_scores=mi_scores,
        n_selected_features=args.n_features,
        memory_info=memory_info,
        is_synthetic=is_synthetic,
    )

    logger.info("EDA completed successfully! All plots saved to: %s", output_dir)


if __name__ == "__main__":
    main()
