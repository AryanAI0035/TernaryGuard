# TernaryGuard architecture and deployment budget

The initial phase-3 comparison uses the same `20 → 32 → 16 → 11` architecture for FP32,
simulated INT8 weights, and ternary weights. RMSNorm precedes each hidden linear
layer; ReLU follows each hidden layer. The classifier has no additional RMSNorm.


The selected research reference is now **20 → 64 → 32 → 11**, chosen by validation
macro-F1 after requiring nonzero validation recall for every class. Matching
FP32/INT8 baselines were trained at that width. Its TCP recall remains too weak
for deployment; see `phase3_corrections.md`.

| Selected 64/32 component | Bytes |
|---|---:|
| 3,680 ternary weights, 2-bit packed | 920 |
| 107 FP32 biases | 428 |
| 3 FP32 scales | 12 |
| 84 FP32 RMSNorm gammas | 336 |
| Model constants | **1,696** |
| Dimension and epsilon metadata | 16 |
| Exported header constants | **1,712** |
| Estimated buffers plus stack reserve | **720** |
| Including provisional RX/TX buffers | **848** |

The selected model uses 107 output-scale multiplications and normalizes 84
values in two RMSNorm operations. These are separate from its 3,680 ternary
weight terms. The smaller 32/16 reference budget below is retained to explain
the initial comparison; neither budget is a measured AVR footprint.

| Model component | Count | Ternary reference storage |
|---|---:|---:|
| Linear weights | 1,328 | 332 bytes, 2-bit packing |
| Biases | 59 | 236 bytes, FP32 |
| Layer scales | 3 | 12 bytes, FP32 |
| RMSNorm gamma | 52 | 208 bytes, FP32 |
| Model constants | | **788 bytes** |
| RMSNorm epsilon | 2 | 8 bytes, FP32 |
| Dimension metadata | 4 | 8 bytes, uint16 |
| Exported C constants including metadata | | **804 bytes** |

The former 611-byte estimate assumed rounding trained biases to unscaled int8,
which materially changed predictions. Version-2 export preserves FP32 biases.
All exported constants use `TGPROGMEM`; desktop builds define it as empty and
AVR builds use `PROGMEM`. An AVR consumer must use the corresponding flash read
functions for weights, biases, scales, normalization constants, and dimensions.

## Runtime estimates, not measurements

For FP32 activations, a possible portable implementation requires 256 bytes for
two 32-element activation buffers, 80 bytes for a separate input buffer, and a
provisional 128-byte stack reserve: **464 bytes**. Adding a provisional 64-byte
RX and 64-byte TX serial buffer yields **592 bytes**. Actual stack use, runtime
globals, normalization temporaries, flash code size, and serial implementation
must be checked on the built engine. There is no measured hardware footprint yet.

Preprocessing is performed on the host for phase-4 golden vectors. A future
engine consuming raw features must additionally implement and budget the scaler
and feature extraction. Raw selected statistics can be far outside int16 range.
Fixed-point activation, bias, normalization, rounding, and saturation formats
have not been validated; use FP32 as the initial correctness reference.

## Arithmetic scope

Ternary dot products replace 1,328 potential weight multiplications with
add/subtract/skip terms. The actual nonzero count is recorded per experiment.
The three layers also perform **59 output scale multiplications** and 59 bias
additions. RMSNorm additionally squares 52 values, reduces two sums, computes
two square roots, normalizes 52 values, and applies 52 gamma multiplications.
These costs are not included in the reported dot-product operation counts.
No full-network multiplier-free, zero-DSP, or fixed-point claim is established.

## Data and reproducibility

The ordered 115-feature schema is stored in `model/feature_schema.json`, copied
from the local N-BaIoT feature dictionary. The loader checks headers and finite
values. It samples equal quotas per class/capture across the available devices,
retaining short captures without redistributing unused quota.

The default protocol trains on devices 1–7, validates on device 8, and tests on
device 9. Device groups never overlap, and all 11 labels must be present in each
partition. These are device-based proportions, not a 70/10/20 row split.

The default host preprocessing first applies `sign(x) * log1p(abs(x))` to raw
statistics. This deterministic transform needs no fitted statistics and is saved
in the bundle; it handles large dynamic ranges and negative covariance values.
Mutual information is estimated on a seeded, stratified sample of at most 20,000
training rows. A greedy filter excludes features with absolute Pearson correlation
at least 0.98 with an already selected feature, measured only on that training
sample, so the 20 slots contain less redundant information. The scaler fits the complete selected training partition. Both
transforms are frozen for validation and test. Each run stores source file
hashes, sampled row identities and partitions, ordered selected features,
scaling parameters, labels, seeds, source-code hashes, checkpoint bundles,
training histories, confusion matrices, and binary/per-class metrics.

The balanced sample does not represent real deployment prevalence. One held-out
device and one seed are a controlled experiment, not evidence of universal
cross-device generalization. Broader claims require additional held-out devices
and seed repetitions.
