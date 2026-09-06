"""
TernaryGuard — Model Package

Ternary neural network components for IoT intrusion detection.
Core module: TernaryLinear layer with absmean quantization + STE.
"""

from model.ternary_linear import (
    TernaryLinear,
    RMSNorm,
    TernaryMLP,
    ternary_quantize,
)

__all__ = [
    "TernaryLinear",
    "RMSNorm",
    "TernaryMLP",
    "ternary_quantize",
]
