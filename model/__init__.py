"""
TernaryGuard — Model Package

Ternary neural network components for IoT intrusion detection.
Core module: TernaryLinear layer with absmean quantization + STE.
Data module: N-BaIoT dataset pipeline with feature selection.
"""

from model.ternary_linear import (
    TernaryLinear,
    RMSNorm,
    TernaryMLP,
    ternary_quantize,
)

from model.data_pipeline import (
    NBaIoTDataset,
    get_feature_names,
    get_label_names,
)

__all__ = [
    "TernaryLinear",
    "RMSNorm",
    "TernaryMLP",
    "ternary_quantize",
    "NBaIoTDataset",
    "get_feature_names",
    "get_label_names",
]
