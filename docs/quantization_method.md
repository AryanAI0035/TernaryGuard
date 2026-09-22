# Ternary Quantization Method — TernaryGuard

## 1. Why Ternary?

Standard neural network weights are stored as 32-bit floats (FP32). Each multiply-accumulate (MAC) operation in a linear layer computes:

$$y_j = \sum_{i} W_{ji} \cdot x_i$$

This requires a hardware multiplier for every $W_{ji} \cdot x_i$ term — multipliers are the single most expensive component in inference hardware in terms of area, latency, and power.

**Ternary quantization** constrains every weight to one of three values: $\{-1, 0, +1\}$. When a weight is one of these three values, the multiplication collapses:

| Weight $w$ | Operation $w \cdot x$ | Hardware cost |
|:---:|:---:|:---:|
| $+1$ | $+x$ (add) | 1 adder |
| $0$ | skip | nothing |
| $-1$ | $-x$ (subtract) | 1 adder (inverted) |

**No multiplier is ever needed.** The entire MAC array reduces to a set of adders with multiplexed sign control. This is the core thesis of TernaryGuard — the same architectural simplification enables deployment on a workstation (where it saves cycles), on an ATmega328P (where it fits in 2KB of RAM), and on an FPGA (where it eliminates DSP blocks entirely).

Each ternary weight carries $\log_2 3 \approx 1.58$ bits of information, hence the "1.58-bit" designation from the BitNet b1.58 paper [1].

## 2. Absmean Quantization

We use the absmean thresholding method from BitNet b1.58 to convert full-precision latent weights into ternary values during training.

### 2.1 Scaling Factor

Given a weight matrix $W \in \mathbb{R}^{m \times n}$, compute the **absmean** scaling factor:

$$\alpha = \text{mean}(|W|) = \frac{1}{mn} \sum_{i,j} |W_{ij}|$$

This is the average magnitude of the weights. It serves two purposes:
1. It defines the **threshold** for quantization (Section 2.2).
2. It acts as a **post-quantization scaling factor** that restores activation magnitudes after the ternary dot product (Section 2.3).

### 2.2 Threshold-Based Quantization

The quantization function maps each weight to $\{-1, 0, +1\}$ using a threshold at half the scaling factor:

$$\tilde{W}_{ij} = \begin{cases} +1 & \text{if } W_{ij} > 0.5 \cdot \alpha \\ -1 & \text{if } W_{ij} < -0.5 \cdot \alpha \\ \phantom{+}0 & \text{otherwise} \end{cases}$$

**Why 0.5 × α?** This threshold is a natural midpoint. Weights whose magnitude is less than half the average are "weak" signals — quantizing them to zero (skip) introduces less error than forcing them to ±1. Weights above the threshold are "strong" and carry directional information worth preserving.

### 2.3 Scaled Output

The forward pass of a ternary linear layer computes:

$$y = \alpha \cdot (\tilde{W} \mathbf{x}) + \mathbf{b}$$

The ternary matrix-vector product $\tilde{W}\mathbf{x}$ involves only additions and subtractions. The scalar multiplication by $\alpha$ is applied **once** to the entire output vector — a single scalar multiply per layer, not per weight.

## 3. Straight-Through Estimator (STE)

### 3.1 The Problem

The quantization function $Q(W) = \tilde{W}$ is a step function — its gradient is zero almost everywhere and undefined at the thresholds. Standard backpropagation would produce zero gradients for all weights, making training impossible:

$$\frac{\partial \tilde{W}}{\partial W} = 0 \quad \text{(almost everywhere)}$$

### 3.2 The Solution

The **Straight-Through Estimator** (Bengio et al., 2013 [2]) replaces the true gradient with the identity:

$$\frac{\partial L}{\partial W} \overset{\text{STE}}{=} \frac{\partial L}{\partial \tilde{W}}$$

In words: during the backward pass, we pretend the quantization step didn't happen. Gradients from the loss flow straight through to the full-precision latent weights, which the optimizer updates normally (e.g., via Adam).

### 3.3 Why It Works

The STE is a biased gradient estimator — it ignores the fact that small weight changes may not change the quantized value at all. Despite this bias, it works well in practice because:

1. **The optimizer accumulates many small gradient steps.** Even if a single update doesn't cross a quantization threshold, the accumulated momentum in Adam will eventually push the latent weight across.
2. **The latent weights remain in full precision.** The optimizer sees a smooth loss landscape through the STE, and the quantization acts as a form of regularization.
3. **The gradient direction is correct.** Even though the magnitude is approximate, the sign of the STE gradient correctly indicates whether the weight should increase or decrease.

### 3.4 Training Loop Summary

```
┌─────────────────────────────────────────────────────────┐
│  for each training step:                                 │
│    1. W_ternary = quantize(W_latent)    # forward only   │
│    2. y = W_ternary @ x * scale + bias  # no multiplies  │
│    3. loss = criterion(y, target)                        │
│    4. loss.backward()                   # STE: ∂L/∂W_latent = ∂L/∂W_ternary │
│    5. optimizer.step()                  # updates W_latent (float32) │
└─────────────────────────────────────────────────────────┘
```

The key insight: **W_latent is never ternary**. It's a full-precision tensor that the optimizer freely updates. Only the **forward-pass projection** $Q(W_\text{latent})$ is ternary. This means the model can explore the weight space continuously while the inference path remains integer-only.

## 4. RMSNorm for Activation Scaling

Before each ternary linear layer, we apply **Root Mean Square Normalization** (Zhang & Sennrich, 2019 [3]):

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}$$

where:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

Unlike LayerNorm, RMSNorm does **not** subtract the mean — it only rescales by the root-mean-square. This is computationally cheaper (no mean computation, no variance) and pairs well with ternary quantization: scale-only normalization preserves the sign structure of activations, which is exactly what the ternary weights exploit.

## 5. Memory Budget Calculation

For a model with $N$ ternary weights:

| Component | Bits per element | Bytes |
|:---|:---:|:---:|
| Ternary weights (2-bit packed) | 2 | $\lceil N \times 2 / 8 \rceil$ |
| Bias (int8) | 8 | $B$ (number of biases) |
| Scale factors (float32) | 32 | $4 \times L$ (number of layers) |
| RMSNorm gamma (float32) | 32 | $4 \times D$ (total feature dims) |

**Example** — the planned TernaryGuard architecture:
- Input(20) → Hidden(16) → Hidden(8) → Output(2)
- Ternary weights: $20 \times 16 + 16 \times 8 + 8 \times 2 = 464$
- Packed (2 bits each): $\lceil 464 \times 2 / 8 \rceil = 116$ bytes
- Bias: $16 + 8 + 2 = 26$ bytes (int8)
- Scales: $3 \times 4 = 12$ bytes
- RMSNorm gamma: $(20 + 16) \times 4 = 144$ bytes
- **Total: 298 bytes** — well within the 32KB flash / 2KB RAM constraints of the ATmega328P.

## 6. What the Unit Tests Verify

| Test | What it proves |
|:---|:---|
| Threshold logic with hand-computed values | The quantization math is correct |
| STE gradient == identity | Gradients flow through unchanged |
| Weight distribution balanced across {-1, 0, +1} | Initialization doesn't produce a degenerate quantization |
| Training reduces loss over 100 steps | End-to-end gradient flow works through quantization |
| Packed size < 32KB | Architecture fits the embedded constraint |
| Output shapes for all configurations | Layer wiring is correct |

---

## References

[1] S. Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," arXiv:2402.17764, 2024.

[2] Y. Bengio, N. Léonard, and A. Courville, "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation," arXiv:1308.3432, 2013.

[3] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019.
