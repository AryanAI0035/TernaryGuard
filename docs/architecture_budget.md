# TernaryGuard — Model Architecture & Memory Budget

## Architecture Decision

Based on the N-BaIoT dataset (115 features → reduced to 20 via feature selection)
and the ATmega328P hardware constraints (32KB flash, 2KB RAM, 16MHz, no FPU):

### Final Architecture

```
Input(20) → RMSNorm(20) → TernaryLinear(20, 32) → ReLU
          → RMSNorm(32) → TernaryLinear(32, 16) → ReLU
          → TernaryLinear(16, 11)  [classifier, 11 classes]
```

**Why this shape:**
- **20 input features**: Feature selection reduces 115 → 20 via mutual information ranking
- **32 → 16 hidden**: Wide-then-narrow captures complex attack patterns in first layer, compresses in second
- **11 output classes**: benign + 5 Mirai attacks + 5 BASHLITE attacks
- **RMSNorm before each hidden layer**: Keeps activations scaled for ternary weights
- **No RMSNorm before classifier**: Final layer operates on already-normalized features

### Parameter Count

| Layer | Weights | Bias | RMSNorm γ |
|-------|---------|------|-----------|
| TernaryLinear(20, 32) | 640 | 32 | 20 |
| TernaryLinear(32, 16) | 512 | 16 | 32 |
| TernaryLinear(16, 11) | 176 | 11 | — |
| **Total** | **1328** | **59** | **52** |

### Flash Budget (32,768 bytes)

| Component | Bytes | Calculation |
|-----------|-------|-------------|
| Ternary weights (packed, 2-bit) | 332 | ⌈1328 × 2 / 8⌉ |
| Biases (int8) | 59 | 59 × 1 |
| Scale factors (float32) | 12 | 3 layers × 4 |
| RMSNorm gamma (float32) | 208 | (20 + 32) × 4 |
| **Model total** | **611** | |
| Inference engine code | ~8,000 | Estimated C code + Arduino overhead |
| Arduino bootloader | ~2,048 | |
| Serial/UART driver | ~1,500 | |
| **Grand total** | **~12,159** | **37.1% of 32KB** ✅ |

### RAM Budget (2,048 bytes)

**PROGMEM rule:** all trained constants go in flash via PROGMEM. Only mutable
runtime state (activation buffers, input buffer, stack) lives in RAM.

Access pattern for PROGMEM constants (via `pgm_read_float()`/`pgm_read_byte()`):
- Weights: read in the inner accumulate loop (highest frequency)
- Biases: read once per output neuron
- RMSNorm gamma: read once per input feature during normalization —
  52 reads total (20+32), ~3µs at 16MHz
- Scale factors: read once per layer, broadcast — 3 reads total

For the conservative estimate with all constants in RAM, call
`estimate_ram_usage(constants_in_progmem=False)` → 516B model subtotal.

| Component | Bytes | Calculation | Source |
|-----------|-------|-------------|--------|
| Activation buffers (ping-pong) | 128 | 2 × 32 × 2 (int16, max hidden dim) | `estimate_ram_usage()` |
| Input buffer | 40 | 20 × 2 (int16 features from UART) | `estimate_ram_usage()` |
| Scale factors | 0 | In PROGMEM (12B flash, 3 reads/inference) | `estimate_ram_usage()` |
| RMSNorm gamma | 0 | In PROGMEM (208B flash, 52 reads/inference) | `estimate_ram_usage()` |
| Stack reserve | 128 | Function calls, locals | `estimate_ram_usage()` |
| **Model subtotal** | **296** | | **Code-verified** |
| Serial buffer (Arduino runtime) | 64 | UART RX buffer | Manual estimate |
| **Grand total** | **360** | **17.6% of 2KB** ✅ | |

### Compared to Phase 1 Example (20→16→8→2)

| Metric | Phase 1 (20→16→8→2) | Phase 2 (20→32→16→11) |
|--------|---------------------|----------------------|
| Ternary weights | 464 | 1,328 |
| Flash (model only) | 298 B | 611 B |
| RAM (runtime) | ~184 B | ~360 B |
| Output classes | 2 | 11 |
| Capacity | Minimal | Production-ready |

**Both fit comfortably.** The larger architecture gives us the capacity to distinguish
11 classes while staying well under both constraints.

## Dataset Summary: N-BaIoT

| Property | Value |
|----------|-------|
| Source | UCI ML Repository / Kaggle |
| Total instances | 7,062,606 |
| Features | 115 behavioral (→ reduced to 20) |
| IoT devices | 9 |
| Attack types | 10 (5 Mirai + 5 BASHLITE) |
| Classes | 11 (benign + 10 attacks) |
| Missing values | None |
| Feature type | Real (continuous) |

### Feature Selection Strategy

Using **mutual information** against the multi-class labels to rank all 115 features,
then selecting the top 20. This method:
1. Captures non-linear dependencies (unlike correlation-based methods)
2. Is label-aware (unlike variance-based methods)
3. Produces a fixed, interpretable feature set for embedded deployment

The selected feature indices are saved to `model/feature_config.json` and used by:
- The C software engine (Phase 4)
- The Arduino engine (Phase 5) 
- The FPGA engine (Phase 6) — hardcoded in Verilog

### Preprocessing

- **StandardScaler**: zero-mean, unit-variance normalization
- Scaler parameters (mean, scale per feature) saved to `feature_config.json`
- On Arduino: scaler is applied in int16 fixed-point arithmetic

### Data Splits

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 70% | Model training |
| Validation | 10% | Hyperparameter tuning, early stopping |
| Test | 20% | Final evaluation, cross-engine validation |

All splits are **stratified** to maintain class balance.
