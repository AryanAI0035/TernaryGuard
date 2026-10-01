# Phase 4 software reference

Frozen reference: commit 54d5aa3, model/active_model.json, 20→64→32→11.
No model, checkpoint, selected features, or trained parameters are changed.
BASHLITE TCP remains a known model limitation: 1/5,555 (0.018%) recall.
The C engine must reproduce it, not change it.

From the repository root:

```sh
python3 model/validate_export.py --verify-active
python3 engine-software/validate.py --output /tmp/tg-phase4
/tmp/tg-phase4/build/ternary_infer --benchmark /tmp/tg-phase4/benchmark_raw_f64.bin 100000
python3 -m pytest model/tests/ -v --tb=short
```

Validation verifies active hashes, source CSV hashes and frozen row identities,
then sends all 115 raw columns to C. It compares C preprocessing separately with
frozen_test_loader(), then compares logits/predictions with the frozen PyTorch
checkpoint. Failures exit nonzero. It requires 69,040 matches, the existing
2e-5 absolute/relative logit tolerance, and exactly 1/5,555 TCP recall.
The output includes parity.json, mismatches.csv, raw input and C output binaries.
The binary CLI protocol uses native-endian IEEE float64 raw rows (115 values)
and float32 output rows (11 logits or 20 preprocessed features).

## Arithmetic and portability

ternary_infer.c includes ../model_weights.h directly. build.py checks active
artifact hashes and generates preprocessing.h from the frozen JSON with exact
hexadecimal double constants; no handwritten duplicate scaler is maintained.
The workstation preprocessing uses float64 signed-log1p and StandardScaler,
followed by a float32 cast, matching Python. Hidden layers apply RMSNorm before
the matmul and ReLU afterward; the classifier has neither RMSNorm nor ReLU.

Before each matmul, activations are converted to int64 with a shared power-of-two
scale. If the largest magnitude has frexp exponent e, the integer conversion is
trunc(x * 2^(40-e)). Each magnitude is below 2^40, and at most 64 terms yield a
sum below 2^46. Conversion error per input is less than 2^(e-40). The packed loop
uses only integer shifts/masks, comparisons, add/subtract/skip, and loads. It
rejects reserved code 3. After accumulation, the binary exponent is restored,
then the trained float scale and bias are applied. This changes numerical
summation order, not model parameters; full-data tolerances and prediction
parity are checked explicitly.

The kernel uses fixed-width integers, bounded indices, no dynamic allocation,
and pgm_read_byte on AVR. Constants use the exported PROGMEM convention.
Scratch is caller-owned and reentrant. No AVR cross-compilation or hardware
claim is made in Phase 4. Phase 5 must validate AVR floating-point/library
behavior, preprocessing precision, stack and SRAM fit. In particular, platforms
with 32-bit double cannot assume this workstation preprocessing parity.

## Benchmark interpretation

The benchmark caches 256 evenly spaced rows across the frozen test set, performs
4,096 warmups, then seven trials of the requested iteration count. Each timed
iteration includes feature selection, log/scaler preprocessing, RMSNorm, integer
conversion, all layers, and a checksum; input disk I/O and setup are excluded.
There is no Python/ctypes overhead in the timed loop. Benchmark input caching
(235,520 bytes) belongs to the workstation runner, not the engine.

Report sizeof(tg_workspace), caller input/output buffers, model/preprocessing
constants, compiler stack-usage output, executable bytes and OS peak resident
memory separately. The model constants include dimension and epsilon metadata.
These measurements do not replace the frozen Phase 3 size estimates and do not
prove an Arduino SRAM budget.

To check memory/undefined behavior, build with --sanitize and run the resulting
runner over the same raw test file. Do not use sanitizer builds for latency.
