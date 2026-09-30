import os
import json
import logging
from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold

logger = logging.getLogger(__name__)


def get_feature_names() -> List[str]:
    """
    Return the 115 N-BaIoT feature names in order.
    Uses the naming convention: {stream}_{timeframe}_{stat}
    """
    streams_stats = {
        'MI_dir': ['weight', 'mean', 'std'],
        'H': ['weight', 'mean', 'std', 'radius', 'magnitude'],
        'HH': ['weight', 'mean', 'std', 'radius', 'magnitude', 'cov', 'pcc'],
        'HH_jit': ['weight', 'mean', 'std'],
        'HpHp': ['weight', 'mean', 'std', 'radius', 'magnitude']
    }
    timeframes = ['L5', 'L3', 'L1', 'L0.1', 'L0.01']
    
    features = []
    for stream, stats in streams_stats.items():
        for timeframe in timeframes:
            for stat in stats:
                features.append(f"{stream}_{timeframe}_{stat}")
    return features


def get_label_names() -> Dict[int, str]:
    """
    Return mapping from int label to human-readable name.
    10 attack types + benign = 11 classes total.
    """
    return {
        0: "benign",
        1: "mirai_ack",
        2: "mirai_scan",
        3: "mirai_syn",
        4: "mirai_udp",
        5: "mirai_udpplain",
        6: "bashlite_combo",
        7: "bashlite_junk",
        8: "bashlite_scan",
        9: "bashlite_tcp",
        10: "bashlite_udp"
    }


class NBaIoTDataset:
    """
    Core data preprocessing and feature selection pipeline for TernaryGuard.
    """

    def __init__(self, data_dir: str = 'data/raw/nbaiot', 
                 selected_features: Optional[List[int]] = None, 
                 max_samples_per_class: int = 50000):
        """
        Initialize the dataset pipeline.
        
        Args:
            data_dir (str): Path to extracted CSVs.
            selected_features (list, optional): List of feature indices to use (None = all 115).
            max_samples_per_class (int): Cap per class to balance the dataset.
        """
        self.data_dir = data_dir
        self.selected_features = selected_features
        self.max_samples_per_class = max_samples_per_class
        self.scaler = StandardScaler()
        self.label_map = get_label_names()
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Walk the data_dir and load all CSVs.
        Assign labels: 0=benign, 1-10 for each attack type.
        Cap samples per class at max_samples_per_class.
        
        Handles both UCI (no header) and Kaggle (with header) CSV formats
        by auto-detecting whether the first row is numeric.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: (features_df, labels_series)
        """
        all_features = []
        all_labels = []
        class_counts = {k: 0 for k in self.label_map.keys()}
        
        feature_names = get_feature_names()
        
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist. Returning empty dataset.")
            return pd.DataFrame(columns=feature_names), pd.Series(dtype=int)

        for root, dirs, files in os.walk(self.data_dir):
            for file in sorted(files):
                if not file.endswith('.csv'):
                    continue
                
                # Determine label from file path (includes parent directory)
                rel_path = os.path.relpath(os.path.join(root, file), self.data_dir).lower()
                label = self._detect_label(rel_path)
                
                if label == -1:
                    logger.warning(f"Could not determine label for {rel_path}, skipping.")
                    continue
                    
                if class_counts[label] >= self.max_samples_per_class:
                    continue
                    
                path = os.path.join(root, file)
                try:
                    # Auto-detect header: try reading first row and check if numeric
                    peek = pd.read_csv(path, nrows=1, header=None)
                    first_row_numeric = all(
                        pd.to_numeric(peek.iloc[0], errors='coerce').notna()
                    )
                    
                    if first_row_numeric:
                        # No header row — raw numeric data (UCI format)
                        df = pd.read_csv(path, header=None)
                        if len(df.columns) == len(feature_names):
                            df.columns = feature_names
                    else:
                        # Has header row (Kaggle format)
                        df = pd.read_csv(path)
                    
                    # Compute how many samples we can take
                    remaining = self.max_samples_per_class - class_counts[label]
                    if len(df) > remaining:
                        df = df.sample(n=remaining, random_state=42)
                        
                    all_features.append(df)
                    all_labels.extend([label] * len(df))
                    class_counts[label] += len(df)
                    logger.info(f"Loaded {len(df)} samples from {rel_path} (label={self.label_map[label]})")
                    
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
                    
        if not all_features:
            return pd.DataFrame(columns=feature_names), pd.Series(dtype=int)
            
        features_df = pd.concat(all_features, ignore_index=True)
        labels_series = pd.Series(all_labels)
        
        logger.info(f"Loaded {len(features_df)} total samples across {sum(1 for v in class_counts.values() if v > 0)} classes")
        
        if self.selected_features is not None:
            features_df = features_df.iloc[:, self.selected_features]
            
        return features_df, labels_series

    def _detect_label(self, rel_path: str) -> int:
        """
        Detect the class label from a relative file path.
        
        Works with both UCI structure (device/attack_type.csv) and 
        Kaggle structure (device/mirai_attacks/ack.csv).
        
        Args:
            rel_path: Relative path from data_dir, lowercased.
            
        Returns:
            Integer label, or -1 if unrecognized.
        """
        if 'benign' in rel_path:
            return 0
        
        # Extract the filename (without extension) for attack type matching
        filename = os.path.splitext(os.path.basename(rel_path))[0].lower()
        # Check if the path contains mirai or bashlite/gafgyt (in directory names)
        path_lower = rel_path.replace('gafgyt', 'bashlite')
        is_mirai = 'mirai' in path_lower
        is_bashlite = 'bashlite' in path_lower
        
        if is_mirai:
            mirai_map = {
                'ack': 1, 'scan': 2, 'syn': 3, 'udp': 4, 'udpplain': 5,
            }
            # Check most specific first (udpplain before udp)
            if 'udpplain' in filename:
                return 5
            for attack, label in mirai_map.items():
                if filename == attack or filename.endswith(attack):
                    return label
        
        if is_bashlite:
            bashlite_map = {
                'combo': 6, 'junk': 7, 'scan': 8, 'tcp': 9, 'udp': 10,
            }
            for attack, label in bashlite_map.items():
                if filename == attack or filename.endswith(attack):
                    return label
        
        return -1

    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', n_features: int = 20) -> Tuple[np.ndarray, List[int]]:
        """
        Perform feature selection.
        
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Labels.
            method (str): 'mutual_info', 'variance', or 'correlation'.
            n_features (int): Number of top features to keep.
            
        Returns:
            Tuple[np.ndarray, List[int]]: (selected_X, selected_feature_indices)
        """
        logger.info(f"Selecting top {n_features} features using method: {method}")
        X_arr = X.values
        y_arr = y.values
        
        if method == 'mutual_info':
            mi = mutual_info_classif(X_arr, y_arr)
            top_indices = np.argsort(mi)[-n_features:][::-1]
            
        elif method == 'variance':
            variances = np.var(X_arr, axis=0)
            top_indices = np.argsort(variances)[-n_features:][::-1]
            
        elif method == 'correlation':
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            
            X_filtered = X.drop(columns=to_drop)
            # fallback to MI for top k of the remaining
            mi = mutual_info_classif(X_filtered.values, y_arr)
            top_filtered_idx = np.argsort(mi)[-n_features:][::-1]
            top_columns = X_filtered.columns[top_filtered_idx]
            
            # map back to original indices
            top_indices = [X.columns.get_loc(c) for c in top_columns]
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
            
        top_indices = list(top_indices)
        selected_X = X_arr[:, top_indices]
        
        print(f"--- Top {n_features} Selected Features ({method}) ---")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {X.columns[idx]} (Index: {idx})")
            
        return selected_X, top_indices

    def preprocess(self, X: Union[pd.DataFrame, np.ndarray], fit: bool = True) -> np.ndarray:
        """
        StandardScaler normalization.
        
        Args:
            X: Features to scale.
            fit (bool): If True, fits the scaler on X before transforming.
            
        Returns:
            np.ndarray: Scaled features.
        """
        if fit:
            self.scaler.fit(X)
        return self.scaler.transform(X)

    def get_splits(self, X: np.ndarray, y: Union[pd.Series, np.ndarray], test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Stratified train/val/test split.
        
        Args:
            X (np.ndarray): Features.
            y: Labels.
            test_size (float): Proportion of dataset to include in test split.
            val_size (float): Proportion of dataset to include in val split.
            
        Returns:
            dict: 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        """
        if isinstance(y, pd.Series):
            y = y.values
            
        # First split out the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Adjust val_size to proportion of remaining data
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def to_torch_loaders(self, splits: Dict[str, np.ndarray], batch_size: int = 256) -> Dict[str, DataLoader]:
        """
        Convert splits to PyTorch DataLoaders.
        
        Args:
            splits (dict): Dictionary from get_splits.
            batch_size (int): Batch size for DataLoaders.
            
        Returns:
            dict: 'train', 'val', 'test' DataLoaders.
        """
        loaders = {}
        for split_name in ['train', 'val', 'test']:
            X_split = splits[f'X_{split_name}']
            y_split = splits[f'y_{split_name}']
            
            dataset = TensorDataset(
                torch.tensor(X_split, dtype=torch.float32),
                torch.tensor(y_split, dtype=torch.long)
            )
            loaders[split_name] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=(split_name == 'train')
            )
            
        return loaders

    def save_feature_config(self, feature_indices: List[int], path: str = 'model/feature_config.json'):
        """
        Save the selected feature indices and scaler parameters to JSON.
        
        Args:
            feature_indices (List[int]): Indices of the selected features.
            path (str): Path to save the JSON config.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config = {
            'selected_features': [int(i) for i in feature_indices],
            'scaler': {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved feature config to {path}")

    @classmethod
    def load_feature_config(cls, path: str = 'model/feature_config.json') -> Dict:
        """
        Load feature config from JSON.
        
        Args:
            path (str): Path to load the JSON config from.
            
        Returns:
            Dict: Config dictionary containing 'selected_features' and 'scaler' parameters.
        """
        with open(path, 'r') as f:
            config = json.load(f)
        return config
