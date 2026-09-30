"""
TernaryGuard — Unit Tests for Data Pipeline

Tests cover:
    1. Feature name generation (115 names)
    2. Label mapping completeness (11 classes)
    3. Label detection from file paths
    4. Dataset loading with empty/missing directory
    5. Preprocessing (scaler fit/transform)
    6. Stratified splits (proportions, label preservation)
    7. PyTorch DataLoader creation
    8. Feature config save/load round-trip
    9. Feature selection methods
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model.data_pipeline import (
    NBaIoTDataset,
    get_feature_names,
    get_label_names,
)


# ──────────────── Fixtures ────────────────


@pytest.fixture
def seed():
    """Fix random seed for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def synthetic_data(seed):
    """Create synthetic data mimicking N-BaIoT structure."""
    n_samples = 500
    n_features = 115
    n_classes = 11

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


@pytest.fixture
def synthetic_dataframe(synthetic_data):
    """Create synthetic pandas DataFrame with feature names."""
    X, y = synthetic_data
    feature_names = get_feature_names()
    df = pd.DataFrame(X, columns=feature_names)
    labels = pd.Series(y)
    return df, labels


# ──────────────── Feature Names ────────────────


class TestFeatureNames:

    def test_count(self):
        """Must produce exactly 115 feature names."""
        names = get_feature_names()
        assert len(names) == 115, f"Expected 115 features, got {len(names)}"

    def test_unique(self):
        """All feature names must be unique."""
        names = get_feature_names()
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_format(self):
        """Each name should follow stream_timeframe_stat convention."""
        names = get_feature_names()
        for name in names:
            parts = name.split("_")
            assert len(parts) >= 3, f"Feature name '{name}' doesn't follow convention"


# ──────────────── Label Names ────────────────


class TestLabelNames:

    def test_count(self):
        """Must have exactly 11 labels (benign + 10 attacks)."""
        labels = get_label_names()
        assert len(labels) == 11

    def test_benign_is_zero(self):
        """Benign must be label 0."""
        labels = get_label_names()
        assert labels[0] == "benign"

    def test_all_keys_present(self):
        """Labels 0-10 must all be present."""
        labels = get_label_names()
        for i in range(11):
            assert i in labels, f"Label {i} missing"


# ──────────────── Label Detection ────────────────


class TestLabelDetection:

    def test_benign(self):
        ds = NBaIoTDataset()
        assert ds._detect_label("danmini/benign_traffic.csv") == 0

    def test_mirai_ack(self):
        ds = NBaIoTDataset()
        assert ds._detect_label("danmini/mirai_attacks/ack.csv") == 1

    def test_mirai_udpplain(self):
        """udpplain must match before udp."""
        ds = NBaIoTDataset()
        assert ds._detect_label("danmini/mirai_attacks/udpplain.csv") == 5

    def test_mirai_udp(self):
        ds = NBaIoTDataset()
        assert ds._detect_label("device/mirai_attacks/udp.csv") == 4

    def test_bashlite_combo(self):
        ds = NBaIoTDataset()
        assert ds._detect_label("device/bashlite_attacks/combo.csv") == 6

    def test_gafgyt_alias(self):
        """gafgyt should be treated as bashlite."""
        ds = NBaIoTDataset()
        assert ds._detect_label("device/gafgyt_attacks/junk.csv") == 7

    def test_unknown(self):
        ds = NBaIoTDataset()
        assert ds._detect_label("device/unknown_file.csv") == -1


# ──────────────── Dataset Loading ────────────────


class TestDatasetLoading:

    def test_missing_directory_returns_empty(self):
        """Non-existent data dir should return empty DataFrame."""
        ds = NBaIoTDataset(data_dir="/nonexistent/path")
        X, y = ds.load_data()
        assert len(X) == 0
        assert len(y) == 0

    def test_load_synthetic_csvs(self, tmp_path, seed):
        """Create synthetic CSVs and verify they load correctly."""
        feature_names = get_feature_names()

        # Create benign CSV
        device_dir = tmp_path / "test_device"
        device_dir.mkdir()
        benign_data = np.random.randn(100, 115)
        df = pd.DataFrame(benign_data, columns=feature_names)
        df.to_csv(device_dir / "benign_traffic.csv", index=False)

        # Create attack CSV
        attack_dir = device_dir / "mirai_attacks"
        attack_dir.mkdir()
        attack_data = np.random.randn(50, 115)
        df_attack = pd.DataFrame(attack_data, columns=feature_names)
        df_attack.to_csv(attack_dir / "ack.csv", index=False)

        ds = NBaIoTDataset(data_dir=str(tmp_path))
        X, y = ds.load_data()

        assert len(X) == 150  # 100 benign + 50 attack
        assert (y == 0).sum() == 100
        assert (y == 1).sum() == 50


# ──────────────── Preprocessing ────────────────


class TestPreprocessing:

    def test_scaler_output_shape(self, synthetic_data):
        X, y = synthetic_data
        ds = NBaIoTDataset()
        X_scaled = ds.preprocess(X, fit=True)
        assert X_scaled.shape == X.shape

    def test_scaler_zero_mean(self, synthetic_data):
        """After fit+transform, mean should be ~0."""
        X, y = synthetic_data
        ds = NBaIoTDataset()
        X_scaled = ds.preprocess(X, fit=True)
        means = X_scaled.mean(axis=0)
        assert np.allclose(means, 0, atol=1e-6)

    def test_scaler_unit_variance(self, synthetic_data):
        """After fit+transform, std should be ~1."""
        X, y = synthetic_data
        ds = NBaIoTDataset()
        X_scaled = ds.preprocess(X, fit=True)
        stds = X_scaled.std(axis=0)
        assert np.allclose(stds, 1, atol=0.1)

    def test_transform_without_fit(self, synthetic_data):
        """Transform-only should use previously fitted scaler."""
        X, y = synthetic_data
        ds = NBaIoTDataset()
        ds.preprocess(X[:200], fit=True)
        X_test = ds.preprocess(X[200:], fit=False)
        assert X_test.shape == (300, 115)


# ──────────────── Splits ────────────────


class TestSplits:

    def test_split_proportions(self, synthetic_data):
        X, y = synthetic_data
        ds = NBaIoTDataset()
        splits = ds.get_splits(X, y, test_size=0.2, val_size=0.1)

        total = len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test'])
        assert total == len(X)

        # Test set should be ~20%
        test_frac = len(splits['X_test']) / len(X)
        assert 0.15 < test_frac < 0.25

    def test_labels_preserved(self, synthetic_data):
        """All original labels should appear in train split."""
        X, y = synthetic_data
        ds = NBaIoTDataset()
        splits = ds.get_splits(X, y)

        train_labels = set(splits['y_train'])
        original_labels = set(y)
        assert train_labels == original_labels

    def test_split_types(self, synthetic_data):
        """Splits should be numpy arrays."""
        X, y = synthetic_data
        ds = NBaIoTDataset()
        splits = ds.get_splits(X, y)

        for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
            assert isinstance(splits[key], np.ndarray), f"{key} is not ndarray"


# ──────────────── DataLoaders ────────────────


class TestDataLoaders:

    def test_loader_creation(self, synthetic_data):
        X, y = synthetic_data
        ds = NBaIoTDataset()
        splits = ds.get_splits(X, y)
        loaders = ds.to_torch_loaders(splits, batch_size=32)

        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders

    def test_loader_batch_shape(self, synthetic_data):
        X, y = synthetic_data
        ds = NBaIoTDataset()
        splits = ds.get_splits(X, y)
        loaders = ds.to_torch_loaders(splits, batch_size=32)

        batch_X, batch_y = next(iter(loaders['train']))
        assert batch_X.shape[1] == 115
        assert batch_X.dtype == torch.float32
        assert batch_y.dtype == torch.long


# ──────────────── Feature Config ────────────────


class TestFeatureConfig:

    def test_save_load_roundtrip(self, synthetic_data, tmp_path):
        """Save and load feature config, verify round-trip."""
        X, y = synthetic_data
        ds = NBaIoTDataset()
        feature_indices = [0, 5, 10, 42, 100]
        ds.preprocess(X[:, feature_indices], fit=True)

        config_path = str(tmp_path / "test_config.json")
        ds.save_feature_config(feature_indices, path=config_path)

        loaded = NBaIoTDataset.load_feature_config(path=config_path)

        assert loaded['selected_features'] == feature_indices
        assert 'scaler' in loaded
        assert len(loaded['scaler']['mean']) == 5
        assert len(loaded['scaler']['scale']) == 5


# ──────────────── Feature Selection ────────────────


class TestFeatureSelection:

    def test_mutual_info_count(self, synthetic_dataframe):
        """Must return exactly n_features indices."""
        df, labels = synthetic_dataframe
        ds = NBaIoTDataset()
        X_sel, indices = ds.select_features(df, labels, method='mutual_info', n_features=20)
        assert X_sel.shape[1] == 20
        assert len(indices) == 20

    def test_variance_count(self, synthetic_dataframe):
        df, labels = synthetic_dataframe
        ds = NBaIoTDataset()
        X_sel, indices = ds.select_features(df, labels, method='variance', n_features=15)
        assert X_sel.shape[1] == 15

    def test_correlation_count(self, synthetic_dataframe):
        df, labels = synthetic_dataframe
        ds = NBaIoTDataset()
        X_sel, indices = ds.select_features(df, labels, method='correlation', n_features=20)
        assert X_sel.shape[1] == 20

    def test_invalid_method_raises(self, synthetic_dataframe):
        df, labels = synthetic_dataframe
        ds = NBaIoTDataset()
        with pytest.raises(ValueError, match="Unknown feature selection method"):
            ds.select_features(df, labels, method='bogus')
