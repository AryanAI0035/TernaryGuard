"""
TernaryGuard — Ternary Linear Layer & Supporting Modules

Implements the core building blocks for a 1.58-bit (ternary) neural network:

    TernaryLinear   — Drop-in replacement for nn.Linear with ternary weight
                      quantization ({-1, 0, +1}) via absmean thresholding and
                      Straight-Through Estimator (STE) for gradient flow.

    RMSNorm         — Root Mean Square normalization for activation scaling,
                      used before each TernaryLinear layer.

    TernaryMLP      — Complete multi-layer perceptron wired with the above,
                      sized for IoT intrusion detection under extreme memory
                      constraints (target: 2KB RAM / 32KB flash on ATmega328P).

Quantization method (BitNet b1.58 style):
    scale     = mean(|W|)                    # absmean scaling factor
    threshold = 0.5 * scale
    W_q[i,j]  = +1  if W[i,j] >  threshold
                -1  if W[i,j] < -threshold
                 0  otherwise

    y = (W_q @ x) * scale + bias

    Since W_q ∈ {-1, 0, +1}, the matrix-vector product reduces to:
        y_j = scale * ( Σ x_i [w=+1]  -  Σ x_i [w=-1] )
    No multiplication is needed in the inner loop — only add, subtract, skip.

References:
    [1] Ma et al., "The Era of 1-bit LLMs: All Large Language Models are
        in 1.58 Bits", arXiv:2402.17764, 2024.
    [2] Bengio et al., "Estimating or Propagating Gradients Through Stochastic
        Neurons for Conditional Computation", arXiv:1308.3432, 2013 (STE).
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

class _TernaryQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for ternary quantization.

    Forward:  quantize weights to {-1, 0, +1} via absmean thresholding.
    Backward: pass gradients through unchanged (identity Jacobian).

    This is the standard STE trick from Bengio et al. (2013) — the gradient
    of the quantization step (which is zero almost everywhere and undefined
    at the thresholds) is replaced with the identity, allowing the optimizer
    to update the underlying full-precision latent weights.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = weight.abs().mean()
        threshold = 0.5 * scale

        # Ternary quantization: +1 / 0 / -1
        w_ternary = torch.zeros_like(weight)
        w_ternary[weight > threshold] = 1.0
        w_ternary[weight < -threshold] = -1.0

        # Save scale for the backward pass (not strictly needed for STE,
        # but useful if we later add gradient clipping at the threshold).
        ctx.save_for_backward(weight)

        return w_ternary, scale

    @staticmethod
    def backward(ctx, grad_w_ternary: torch.Tensor, grad_scale: torch.Tensor):
        # STE: ∂L/∂W_latent = ∂L/∂W_ternary  (identity pass-through)
        return grad_w_ternary


def ternary_quantize(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to {-1, 0, +1} with STE gradient support.

    Args:
        weight: Full-precision weight tensor of any shape.

    Returns:
        w_ternary: Quantized weight tensor (same shape), values in {-1, 0, +1}.
        scale:     Scalar absmean scaling factor (mean of |W|).

    The returned w_ternary supports autograd via STE — gradients flow through
    as if the quantization were the identity function.
    """
    return _TernaryQuantizeSTE.apply(weight)


# ---------------------------------------------------------------------------
# TernaryLinear
# ---------------------------------------------------------------------------

class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weight quantization.

    Drop-in replacement for ``nn.Linear``. Weights are stored in full
    precision (float32) and updated by the optimizer normally. During the
    forward pass, weights are quantized on-the-fly to {-1, 0, +1} via
    absmean thresholding; the backward pass uses STE.

    The output is scaled by the absmean factor so that activation magnitudes
    remain in a sensible range despite the extreme quantization:

        y = (W_q @ x) * scale + bias

    At inference / export time, ``get_ternary_weights()`` returns the frozen
    quantized weights and scale for use by the C / AVR / Verilog engines.

    Args:
        in_features:  Size of each input sample.
        out_features: Size of each output sample.
        bias:         If True (default), adds a learnable bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision latent weights — these are what the optimizer updates.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Kaiming uniform init — standard for ReLU networks.
        # The fan-in calculation accounts for the weight shape.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # ---- forward / quantize ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, scale = ternary_quantize(self.weight)
        out = F.linear(x, w_q) * scale
        if self.bias is not None:
            out = out + self.bias
        return out

    # ---- inference helpers ----

    @torch.no_grad()
    def get_ternary_weights(self) -> Tuple[torch.Tensor, float]:
        """
        Return frozen (no-grad) quantized weights and scale.

        Returns:
            w_q:   int8 tensor with values in {-1, 0, +1}, shape (out, in).
            scale: float, the absmean scaling factor.
        """
        scale = self.weight.abs().mean()
        threshold = 0.5 * scale

        w_q = torch.zeros_like(self.weight, dtype=torch.int8)
        w_q[self.weight > threshold] = 1
        w_q[self.weight < -threshold] = -1

        return w_q, scale.item()

    @torch.no_grad()
    def weight_distribution(self) -> Dict[int, int]:
        """
        Count how many weights fall into each ternary bucket.

        Returns:
            dict mapping {-1: count, 0: count, +1: count}
        """
        w_q, _ = self.get_ternary_weights()
        return {
            -1: int((w_q == -1).sum()),
            0: int((w_q == 0).sum()),
            +1: int((w_q == 1).sum()),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm does not subtract the mean — it only rescales
    by the root-mean-square of the activations:

        RMSNorm(x) = (x / RMS(x)) * γ
        where RMS(x) = sqrt( mean(x²) + ε )

    This is cheaper to compute (no mean subtraction, no variance) and works
    well with ternary networks because scale-only normalization preserves
    the sign structure that ternary weights exploit.

    Args:
        dim: Feature dimension to normalize over (last axis).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma

    def extra_repr(self) -> str:
        return f"dim={self.gamma.shape[0]}, eps={self.eps}"


# ---------------------------------------------------------------------------
# TernaryMLP
# ---------------------------------------------------------------------------

class TernaryMLP(nn.Module):
    """
    Complete ternary MLP for intrusion detection.

    Architecture::

        input → [RMSNorm → TernaryLinear → ReLU] × len(hidden_dims) → classifier

    All hidden layers use ternary quantization. The output (classifier) layer
    is ternary by default but can be switched to full-precision ``nn.Linear``
    via ``ternary_output=False`` for more accurate decision boundaries.

    Args:
        input_dim:      Number of input features.
        hidden_dims:    List of hidden layer widths, e.g. [32, 16].
        output_dim:     Number of output classes.
        ternary_output: If True (default), the classifier is TernaryLinear.
        use_rmsnorm:    If True (default), insert RMSNorm before each layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        ternary_output: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            if use_rmsnorm:
                layers.append(RMSNorm(prev_dim))
            layers.append(TernaryLinear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.features = nn.Sequential(*layers)

        if ternary_output:
            self.classifier = TernaryLinear(prev_dim, output_dim)
        else:
            self.classifier = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    # ---- model analysis helpers ----

    def count_parameters(self) -> Dict[str, int]:
        """
        Count total trainable parameters and ternary-quantized parameters.

        Returns:
            dict with keys 'total' and 'ternary'.
        """
        total = sum(p.numel() for p in self.parameters())
        ternary = sum(
            m.weight.numel()
            for m in self.modules()
            if isinstance(m, TernaryLinear)
        )
        return {"total": total, "ternary": ternary}

    def estimate_packed_size(self) -> Dict[str, int]:
        """
        Estimate the packed model size in bytes for embedded deployment.

        Ternary weights are packed at 2 bits each (4 trits per byte).
        Encoding: 00 = 0, 01 = +1, 10 = -1, 11 = unused.
        This is slightly less dense than the information-theoretic minimum
        of log₂3 ≈ 1.58 bits, but trivial to encode/decode in C and Verilog.

        Biases are stored as int8 on the embedded target (1 byte each).
        Scale factors are float32 (4 bytes each, one per TernaryLinear layer).
        RMSNorm gammas are float32 (4 bytes per feature dimension element).

        Returns:
            dict with detailed byte breakdown and total.
        """
        ternary_weight_count = 0
        bias_count = 0
        scale_count = 0
        rmsnorm_param_count = 0

        for module in self.modules():
            if isinstance(module, TernaryLinear):
                ternary_weight_count += module.weight.numel()
                scale_count += 1
                if module.bias is not None:
                    bias_count += module.bias.numel()
            elif isinstance(module, RMSNorm):
                rmsnorm_param_count += module.gamma.numel()

        packed_weight_bytes = math.ceil(ternary_weight_count * 2 / 8)
        bias_bytes = bias_count  # int8 on embedded
        scale_bytes = scale_count * 4  # float32
        rmsnorm_bytes = rmsnorm_param_count * 4  # float32 gamma

        total = packed_weight_bytes + bias_bytes + scale_bytes + rmsnorm_bytes

        return {
            "ternary_weights": ternary_weight_count,
            "packed_weight_bytes": packed_weight_bytes,
            "bias_bytes": bias_bytes,
            "scale_bytes": scale_bytes,
            "rmsnorm_bytes": rmsnorm_bytes,
            "total_bytes": total,
        }

    def estimate_ram_usage(self, activation_dtype_bytes: int = 2,
                           stack_reserve: int = 128) -> Dict[str, int]:
        """
        Estimate peak runtime RAM usage for embedded inference.

        On the ATmega328P (2048 bytes total RAM), inference needs:
        - **Input buffer**: input_dim × dtype_bytes
        - **Activation buffers**: two ping-pong buffers sized to the largest
          layer dimension (current + next), so we only need max(all dims) × dtype
        - **Scale storage**: one float32 per TernaryLinear layer
        - **Stack reserve**: function calls, locals, serial buffer

        Weights are NOT counted here — they live in flash via PROGMEM.

        Args:
            activation_dtype_bytes: Bytes per activation element. Default 2
                (int16 fixed-point on AVR, no FPU).
            stack_reserve: Bytes reserved for stack + serial buffer.

        Returns:
            dict with detailed RAM breakdown, total, and headroom vs 2KB.
        """
        all_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        max_dim = max(all_dims)

        # Ping-pong buffers: need two buffers of max_dim to hold
        # current-layer input and next-layer output simultaneously
        activation_bytes = 2 * max_dim * activation_dtype_bytes

        # Input buffer (feature vector from UART)
        input_bytes = self.input_dim * activation_dtype_bytes

        # Scale factors kept in RAM (one float32 per TernaryLinear layer)
        num_ternary_layers = sum(
            1 for m in self.modules() if isinstance(m, TernaryLinear)
        )
        scale_bytes = num_ternary_layers * 4

        # RMSNorm gamma — on Arduino these may also be in PROGMEM,
        # but conservatively assume RAM
        rmsnorm_ram = sum(
            m.gamma.numel() * 4
            for m in self.modules() if isinstance(m, RMSNorm)
        )

        total = activation_bytes + input_bytes + scale_bytes + rmsnorm_ram + stack_reserve
        ram_limit = 2048  # ATmega328P

        return {
            "activation_bytes": activation_bytes,
            "input_buffer_bytes": input_bytes,
            "scale_bytes": scale_bytes,
            "rmsnorm_ram_bytes": rmsnorm_ram,
            "stack_reserve": stack_reserve,
            "total_bytes": total,
            "ram_limit": ram_limit,
            "headroom_bytes": ram_limit - total,
            "fits": total <= ram_limit,
        }

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}"
        )
