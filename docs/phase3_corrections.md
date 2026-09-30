# Phase 3 corrections and rerun results

The pipeline and export defects from the audit are corrected. The exported
reference is numerically validated. **The 11-class model is still not ready for
deployment: BASHLITE TCP recall is only 1/5,555 (0.0180%) on the held-out test
device.** Do not treat the minimum nonzero-recall selection gate as sufficient
model quality, or claim phase 3's classification objective is fully resolved.

## Corrected workflow

- Use the authoritative 115-feature schema; validate input order and finite values.
- Sample all nine devices with per-class/capture quotas. Train on devices 1–7,
  validate on device 8, and test on device 9. Counts are 411,915 / 69,045 / 69,040.
- Fit mutual-information selection and normalization only on training rows.
  The final preprocessing uses signed-log scaling and a training-only 0.98
  correlation filter. Compare against identity-transform controls.
- Seed selection, initialization, shuffling, and training. Save every model,
  training history, per-class confusion matrix, binary metrics, source hashes,
  split identities, ordered features/scaler, and experiment provenance.
- Bind preprocessing to version-2 checkpoints. Export never refits data and
  rejects legacy bare weights. Canonical JSON fingerprints survive serialization.
- Preserve FP32 biases and normalization constants in the C header, use PROGMEM
  consistently, and validate actual exported constants independently.
- Repair CSV quoting and headers, separate measured metrics from estimates, and
  label INT8 as simulated weight quantization with FP32 execution.

## Model selection and comparable results

Three preprocessing runs each contain FP32, simulated INT8, a ternary 32/16
comparison, and five width ablations (24 recorded experiments). Final selection
examines validation only: require nonzero recall in all 11 classes, then maximize
macro-F1 among eligible candidates. This chooses the signed-log **20 → 64 → 32 → 11**
model. The rule is a minimum sanity check, not a deployment acceptance standard.
The selected checkpoint reached its best validation epoch within the 30-epoch
ablation budget. FP32 and INT8 were then rerun at the same 64/32 architecture,
preprocessing, seed, and maximum 30-epoch training budget. The final comparison
adds three records; all 27 records are retained in `benchmarks/accuracy_ablation.csv`.

| Model | Test accuracy | Macro F1 | Estimated model constants |
|---|---:|---:|---:|
| FP32 | 89.60% | 0.8577 | 15,484 B |
| INT8 weight simulation | 88.75% | 0.8502 | 4,456 B |
| Ternary | 81.08% | 0.7821 | 1,696 B |

The prior 85.5% ternary result used a different, leaking row-split evaluation and
is archived as invalidated. It is not a comparable baseline for these numbers.
Ternary binary benign/attack accuracy is 99.9493%, but this
must not be substituted for its 81.08% 11-class accuracy. The FP32 baseline
also misses an attack class, so class confusion is not exclusively a ternary issue.

A separate training/device-8-only tree diagnostic separates TCP/UDP with the
selected identity-transform features at 100% validation balanced accuracy on
its 2,000-sample diagnostic set. This establishes useful input signal; it does
not establish equivalent neural-network performance. See `research/diagnose_tcp_udp.py`
and `benchmarks/tcp_udp_diagnostic.json` for its narrower protocol.

## Validation and artifacts

- **85 tests pass**, including leakage prevention, device coverage, schema,
  fractional-bias export parity, C compilation, immutable preprocessing,
  checksum round trips, source/row tamper detection, and CSV serialization.
- Full exported-header versus PyTorch comparison: **69,040 test samples, zero
  classification disagreements**, maximum absolute logit difference
  **1.1444091796875e-05**, using float32 NumPy as the independent header reader.
  This is not yet the phase-4 C engine or a hardware measurement.
- Golden vectors contain 24 examples per class (264 total), standardized inputs
  and expected logits. Checkpoint, preprocessing, and header hashes agree.
- The 64/32 model contains 3,680 ternary weights. Its model constants occupy
  **1,696 B**, or **1,712 B** including dimension/epsilon metadata. Estimated
  runtime buffers plus stack reserve are **720 B**; adding provisional 128 B
  serial buffers gives **848 B**. Actual embedded usage remains unmeasured.

The checked-in reference is identified by `model/active_model.json`, with
`model_weights.h`, `model/feature_config.json`, `model/export_manifest.json`, and
`model/golden_vectors.npz`. The full local bundle is
`checkpoints/phase3_final_seed42/ternary_2bit.pt`. Checkpoints/raw data remain
excluded from Git. The malformed legacy results and old checkpoint/header are
preserved under `checkpoints/legacy_phase3/`; the cleaned legacy CSV is explicitly
marked invalidated under `docs/benchmarks/`.

For the next modeling step, resolve the extremely weak TCP recall and agree on
per-class acceptance criteria before declaring complete 11-class detection.
The numerical reference and golden vectors can support phase-4 correctness work,
but stronger model-quality, fixed-point, latency, memory, and hardware claims
still require their respective validation.
