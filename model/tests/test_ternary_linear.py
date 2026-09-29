"""
TernaryGuard — Unit Tests for Ternary Quantization Components

Tests cover:
    1. _TernaryQuantizeSTE: forward correctness + STE gradient pass-through
    2. TernaryLinear: shapes, gradients, weight distribution, determinism
    3. RMSNorm: shape, normalization property
    4. TernaryMLP: end-to-end forward, training convergence, parameter counting,
                   packed size estimation, configuration variants
"""

import math

import pytest
import torch
import torch.nn as nn

from model.ternary_linear import (
    TernaryLinear,
    TernaryMLP,
    RMSNorm,
    ternary_quantize,
)


# ──────────────── Fixtures ────────────────


@pytest.fixture
def seed():
    """Fix random seed for reproducible tests."""
    torch.manual_seed(42)


# ──────────────── ternary_quantize (STE primitive) ────────────────


class TestTernaryQuantize:
    """Tests for the standalone ternary_quantize function."""

    def test_output_values_are_ternary(self, seed):
        """Every element of the quantized tensor must be in {-1, 0, +1}."""
        w = torch.randn(64, 32)
        w_q, scale = ternary_quantize(w)
        unique = set(w_q.unique().tolist())
        assert unique.issubset({-1.0, 0.0, 1.0}), f"Unexpected values: {unique}"

    def test_scale_is_absmean(self, seed):
        """Scale should equal mean(|W|)."""
        w = torch.randn(64, 32)
        expected_scale = w.abs().mean()
        _, scale = ternary_quantize(w)
        assert torch.allclose(scale, expected_scale, atol=1e-6)

    def test_threshold_logic(self):
        """Manually verify the threshold-based quantization."""
        # Construct a weight tensor where we know the exact outcome.
        # scale = mean(|w|) = mean([0.2, 0.6, 1.0, 0.2, 0.6, 1.0]) = 0.6
        # threshold = 0.5 * 0.6 = 0.3
        w = torch.tensor([0.2, 0.6, 1.0, -0.2, -0.6, -1.0], dtype=torch.float32)
        w.requires_grad_(True)
        w_q, scale = ternary_quantize(w)

        assert scale.item() == pytest.approx(0.6, abs=1e-6)
        expected = torch.tensor([0.0, 1.0, 1.0, 0.0, -1.0, -1.0])
        assert torch.equal(w_q, expected), f"Got {w_q}, expected {expected}"

    def test_ste_gradient_identity(self, seed):
        """STE backward must pass gradients through unchanged."""
        w = torch.randn(16, 8, requires_grad=True)
        w_q, scale = ternary_quantize(w)

        # Use a non-trivial upstream gradient
        grad_upstream = torch.randn_like(w_q)
        w_q.backward(grad_upstream)

        # STE: ∂L/∂W_latent == grad_upstream (identity)
        assert w.grad is not None, "No gradient computed"
        assert torch.allclose(w.grad, grad_upstream, atol=1e-7), (
            "STE gradient is not identity"
        )

    def test_gradient_is_nonzero(self, seed):
        """Gradient through STE must be non-zero (otherwise training stalls)."""
        w = torch.randn(32, 16, requires_grad=True)
        w_q, scale = ternary_quantize(w)
        loss = w_q.sum()
        loss.backward()
        assert w.grad is not None
        assert w.grad.abs().sum() > 0

    def test_all_zeros_weight(self):
        """Edge case: all-zero weights should produce all-zero quantized output."""
        w = torch.zeros(8, 4, requires_grad=True)
        w_q, scale = ternary_quantize(w)
        assert scale.item() == 0.0
        assert torch.all(w_q == 0.0)

    def test_all_same_sign(self):
        """When all weights are positive and large, all should quantize to +1."""
        w = torch.ones(8, 4) * 5.0
        w.requires_grad_(True)
        w_q, scale = ternary_quantize(w)
        assert torch.all(w_q == 1.0)


# ──────────────── TernaryLinear ────────────────


class TestTernaryLinear:
    """Tests for the TernaryLinear layer."""

    def test_output_shape_2d(self, seed):
        """Standard 2D input: (batch, features)."""
        layer = TernaryLinear(20, 16)
        x = torch.randn(8, 20)
        out = layer(x)
        assert out.shape == (8, 16)

    def test_output_shape_single_sample(self, seed):
        """Single sample: (1, features)."""
        layer = TernaryLinear(20, 16)
        x = torch.randn(1, 20)
        out = layer(x)
        assert out.shape == (1, 16)

    def test_output_shape_no_bias(self, seed):
        """Without bias, output shape should be unchanged."""
        layer = TernaryLinear(20, 16, bias=False)
        x = torch.randn(8, 20)
        out = layer(x)
        assert out.shape == (8, 16)

    def test_bias_parameter_registration(self):
        """Bias should be None when disabled, Parameter when enabled."""
        layer_bias = TernaryLinear(10, 5, bias=True)
        layer_nobias = TernaryLinear(10, 5, bias=False)
        assert layer_bias.bias is not None
        assert layer_nobias.bias is None

    def test_gradients_nonzero_weight(self, seed):
        """Weight gradients must be non-zero after a backward pass."""
        layer = TernaryLinear(20, 16)
        x = torch.randn(8, 20)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0

    def test_gradients_nonzero_bias(self, seed):
        """Bias gradients must be non-zero after a backward pass."""
        layer = TernaryLinear(20, 16, bias=True)
        x = torch.randn(8, 20)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.bias.grad is not None
        assert layer.bias.grad.abs().sum() > 0

    def test_weight_distribution_balanced(self, seed):
        """
        With Kaiming initialization, the ternary distribution should not
        be degenerate — each bucket {-1, 0, +1} should contain >5% of weights.
        """
        layer = TernaryLinear(100, 100)
        dist = layer.weight_distribution()
        total = dist[-1] + dist[0] + dist[1]
        assert total == 10000, f"Total weights mismatch: {total}"
        for k in [-1, 0, 1]:
            frac = dist[k] / total
            assert frac > 0.05, (
                f"Bucket {k} has only {frac:.1%} of weights — distribution is degenerate"
            )

    def test_quantized_weights_are_int8_ternary(self, seed):
        """get_ternary_weights() must return int8 values in {-1, 0, +1}."""
        layer = TernaryLinear(20, 16)
        w_q, scale = layer.get_ternary_weights()
        assert w_q.dtype == torch.int8
        unique = set(w_q.unique().tolist())
        assert unique.issubset({-1, 0, 1}), f"Non-ternary values: {unique}"

    def test_scale_is_positive(self, seed):
        """Absmean scale must be strictly positive (non-zero weight init)."""
        layer = TernaryLinear(20, 16)
        _, scale = layer.get_ternary_weights()
        assert scale > 0

    def test_deterministic_forward(self, seed):
        """Two forward passes with the same input must produce identical output."""
        layer = TernaryLinear(20, 16)
        x = torch.randn(4, 20)
        out1 = layer(x)
        out2 = layer(x)
        assert torch.equal(out1, out2)

    def test_repr(self):
        """__repr__ should include in/out features."""
        layer = TernaryLinear(20, 16, bias=False)
        r = repr(layer)
        assert "20" in r and "16" in r


# ──────────────── RMSNorm ────────────────


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self, seed):
        norm = RMSNorm(20)
        x = torch.randn(8, 20)
        out = norm(x)
        assert out.shape == (8, 20)

    def test_rms_near_one(self, seed):
        """After normalization (with gamma=1), RMS of output should be ≈ 1."""
        norm = RMSNorm(100)
        x = torch.randn(32, 100) * 5.0  # large-magnitude input
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.15), (
            f"RMS not near 1: {rms.mean():.3f} ± {rms.std():.3f}"
        )

    def test_gradient_flows(self, seed):
        """Gradients must flow through RMSNorm."""
        norm = RMSNorm(20)
        x = torch.randn(4, 20, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gamma_gradient(self, seed):
        """The learnable gamma parameter must receive gradients."""
        norm = RMSNorm(20)
        x = torch.randn(4, 20)
        out = norm(x)
        out.sum().backward()
        assert norm.gamma.grad is not None
        assert norm.gamma.grad.abs().sum() > 0

    def test_identity_at_unit_rms(self):
        """
        If input already has RMS=1 and gamma=1, output ≈ input.
        """
        norm = RMSNorm(4)
        # Construct input with RMS exactly 1: x = [1, 1, 1, 1] → RMS = 1
        x = torch.ones(1, 4)
        out = norm(x)
        assert torch.allclose(out, x, atol=1e-4)


# ──────────────── TernaryMLP ────────────────


class TestTernaryMLP:
    """Tests for the full ternary MLP model."""

    def test_forward_shape(self, seed):
        """Output shape: (batch, output_dim)."""
        model = TernaryMLP(20, [16, 8], 2)
        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4, 2)

    def test_forward_multiclass(self, seed):
        """Multi-class output (e.g. 5 attack types)."""
        model = TernaryMLP(30, [32, 16], 5)
        x = torch.randn(8, 30)
        out = model(x)
        assert out.shape == (8, 5)

    def test_training_reduces_loss(self, seed):
        """
        Training for 100 steps on synthetic data must reduce the loss.
        This verifies that gradients flow end-to-end through quantization.
        """
        model = TernaryMLP(20, [16, 8], 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(64, 20)
        y = torch.randint(0, 2, (64,))

        # Initial loss
        with torch.no_grad():
            loss_initial = criterion(model(x), y).item()

        # Train
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        loss_final = loss.item()
        assert loss_final < loss_initial, (
            f"Loss did not decrease: {loss_initial:.4f} → {loss_final:.4f}"
        )

    def test_count_parameters(self, seed):
        """Parameter counting must be consistent."""
        model = TernaryMLP(20, [16, 8], 2)
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert counts["ternary"] > 0
        assert counts["ternary"] <= counts["total"]

        # Manual check: ternary weights = sum of TernaryLinear weight numel
        expected_ternary = sum(
            m.weight.numel()
            for m in model.modules()
            if isinstance(m, TernaryLinear)
        )
        assert counts["ternary"] == expected_ternary

    def test_packed_size_below_flash(self, seed):
        """
        A model sized for the Arduino (≤4000 ternary weights) should
        pack into well under 32KB flash.
        """
        # Realistic architecture: input(20) → hidden(16) → hidden(8) → output(2)
        model = TernaryMLP(20, [16, 8], 2)
        size = model.estimate_packed_size()

        assert size["total_bytes"] > 0
        assert size["total_bytes"] < 32768, (
            f"Packed size {size['total_bytes']} exceeds 32KB flash limit"
        )

        # Packed weight bytes should be < raw weight count
        # (since 2 bits < 8 bits per weight)
        assert size["packed_weight_bytes"] < size["ternary_weights"]

    def test_no_rmsnorm(self, seed):
        """Model without RMSNorm should still work."""
        model = TernaryMLP(20, [16, 8], 2, use_rmsnorm=False)
        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4, 2)

        # No RMSNorm modules should be present
        rmsnorm_count = sum(
            1 for m in model.modules() if isinstance(m, RMSNorm)
        )
        assert rmsnorm_count == 0

    def test_fp_output_layer(self, seed):
        """With ternary_output=False, classifier should be nn.Linear."""
        model = TernaryMLP(20, [16, 8], 2, ternary_output=False)
        assert isinstance(model.classifier, nn.Linear)
        assert not isinstance(model.classifier, TernaryLinear)

        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4, 2)

    def test_single_hidden_layer(self, seed):
        """Model with one hidden layer."""
        model = TernaryMLP(20, [16], 2)
        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4, 2)

    def test_deep_model(self, seed):
        """Model with many hidden layers."""
        model = TernaryMLP(20, [32, 16, 8, 4], 2)
        x = torch.randn(4, 20)
        out = model(x)
        assert out.shape == (4, 2)

    def test_estimate_packed_size_includes_rmsnorm(self, seed):
        """Packed size estimate should account for RMSNorm gamma parameters."""
        model_with = TernaryMLP(20, [16, 8], 2, use_rmsnorm=True)
        model_without = TernaryMLP(20, [16, 8], 2, use_rmsnorm=False)

        size_with = model_with.estimate_packed_size()
        size_without = model_without.estimate_packed_size()

        assert size_with["rmsnorm_bytes"] > 0
        assert size_without["rmsnorm_bytes"] == 0
        assert size_with["total_bytes"] > size_without["total_bytes"]

    def test_quantize_paths_agree(self, seed):
        """
        The forward-path quantization (ternary_quantize, returns float32)
        and the inference-path quantization (get_ternary_weights, returns int8)
        must produce identical ternary values for the same weights.

        If someone modifies one path but not the other, this test catches
        the silent divergence before it propagates to cross-engine validation.
        """
        layer = TernaryLinear(32, 16)

        # Forward path (via autograd function)
        w_q_forward, scale_forward = ternary_quantize(layer.weight)
        w_q_forward_int = w_q_forward.detach().to(torch.int8)

        # Inference path (standalone reimplementation)
        w_q_inference, scale_inference = layer.get_ternary_weights()

        # Values must agree exactly
        assert torch.equal(w_q_forward_int, w_q_inference), (
            "Forward and inference quantization paths disagree!"
        )
        assert scale_forward.item() == pytest.approx(scale_inference, abs=1e-6), (
            f"Scales disagree: forward={scale_forward.item()}, inference={scale_inference}"
        )

    def test_ram_fits_2kb(self, seed):
        """
        The target architecture must fit within ATmega328P's 2KB RAM.
        This tests the binding constraint (RAM), not just flash.
        """
        model = TernaryMLP(20, [16, 8], 2)
        ram = model.estimate_ram_usage()

        assert ram["fits"], (
            f"Model needs {ram['total_bytes']} bytes RAM but limit is {ram['ram_limit']}"
        )
        assert ram["headroom_bytes"] > 0
        assert ram["total_bytes"] > 0

    def test_pack_ternary_2bit_roundtrip(self, seed):
        """
        Packing and unpacking must be lossless.
        Encoding: 0b00=0, 0b01=+1, 0b10=-1.
        """
        import numpy as np
        model = TernaryMLP(20, [16, 8], 2)

        for module in model.modules():
            if isinstance(module, TernaryLinear):
                w_ternary, _ = module.get_ternary_weights()
                w_np = w_ternary.numpy()
                packed = TernaryMLP._pack_ternary_2bit(w_np)

                # Unpack and verify round-trip
                flat_original = w_np.flatten()
                unpacked = []
                for byte_val in packed:
                    for j in range(4):
                        code = (byte_val >> (j * 2)) & 0x03
                        if code == 0b01:
                            unpacked.append(1)
                        elif code == 0b10:
                            unpacked.append(-1)
                        else:
                            unpacked.append(0)

                # Trim to original length (last byte may have padding)
                unpacked = unpacked[:len(flat_original)]
                assert np.array_equal(flat_original, np.array(unpacked, dtype=np.int8)), \
                    "2-bit pack/unpack round-trip failed"
                break  # Only need to test one layer

    def test_export_weights_header(self, seed, tmp_path):
        """
        export_weights_header() must produce a valid C header file with
        the canonical encoding spec comment and correct architecture defines.
        """
        from model.ternary_linear import TRIT_ENCODING_SPEC_LINES

        model = TernaryMLP(20, [16, 8], 2)
        out_path = str(tmp_path / "test_weights.h")
        model.export_weights_header(path=out_path)

        with open(out_path, 'r') as f:
            content = f.read()

        # Must contain every line from the SINGLE source of truth constant
        for spec_line in TRIT_ENCODING_SPEC_LINES:
            if spec_line.strip():  # skip empty lines
                assert spec_line.strip() in content, (
                    f"Missing from generated header: '{spec_line}'"
                )

        # Must say LSB-first, NOT MSB-first
        assert "LSB-first" in content
        assert "MSB-first" not in content

        # Must contain architecture defines
        assert "#define TG_INPUT_DIM  20" in content
        assert "#define TG_OUTPUT_DIM 2" in content
        assert "#define TG_NUM_LAYERS 3" in content

        # Must contain include guard
        assert "#ifndef TERNARYGUARD_MODEL_WEIGHTS_H" in content
        assert "#endif" in content

        # Must contain weight arrays for all 3 layers
        assert "tg_weights_0" in content
        assert "tg_weights_1" in content
        assert "tg_weights_2" in content

        # Must contain PROGMEM ifdef
        assert "TGPROGMEM" in content
        assert "__AVR__" in content

