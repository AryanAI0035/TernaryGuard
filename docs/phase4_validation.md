> Historical phase record: measurements and test counts below describe their recorded acceptance run. The completed core now includes the model, C engine and physically validated Nano; FPGA is optional future scope. See [current project documentation](README.md). Historical budgets do not replace measured Nano resources.

# Phase 4 C engine verification

Frozen reference: commit 54d5aa3 and model/active_model.json. Executed on Apple M3
arm64 with Apple clang 21.0.0, C11, -O2, contraction disabled and no fast math.
All frozen model, checkpoint, preprocessing and Phase 3 results files remain unchanged.

## Result

69,040/69,040 C predictions match PyTorch. Signed-log1p/scaler output matches
Python float32 inputs exactly on all frozen test rows. Maximum logit difference:
1.1444091796875e-05; all logits satisfy atol=rtol=2e-5. Accuracy remains
0.8108198146002318 and macro-F1 remains 0.78209432162603.

Known limitation retained: BASHLITE TCP true positives are exactly 1/5,555 in
both C and PyTorch (0.01800180018% recall). This is not a model improvement.

## Workstation measurement

Seven trials of 100,000 inferences each, after 4,096 warmups, on 256 evenly spaced
frozen test rows. Median: 3.763160 microseconds/inference; range
3.692030–3.851810 microseconds.
Includes C preprocessing, RMSNorm, conversion, all layers and checksum; excludes
input disk I/O. This is a warm-cache workstation measurement, not AVR timing.

Memory quantities are intentionally separate:

- Caller-owned reusable engine workspace: 1,024 bytes.
- Raw 115-column float64 input buffer: 920 bytes; 11-logit output: 44 bytes.
- Declared model constant payload including dimensions/epsilons: 1,712 bytes.
- Declared preprocessing constants: 340 bytes (20 indices + 40 doubles).
- Executable on disk: 35,168 bytes, including the workstation CLI.
- Benchmark process peak RSS: 1,867,776 bytes (1.78125 MiB).
- Benchmark cache alone: 235,520 bytes; this belongs to the runner.
- Compiler reports engine stack frames of 128 bytes (tg_infer), 80 bytes
  (tg_infer_preprocessed), 144 bytes (layer), 112 bytes (tg_preprocess), and
  zero (tg_packed_dot). These are per-function frames, not total process RAM;
  library frames and caller buffers must also be included in a target budget.

The source kernel uses bounded fixed-width integer operations and PROGMEM reads
on AVR, with no heap allocation or floating-point arithmetic in the dot product.
The emitted ARM64 dot-product instructions independently show no floating-point
or multiply instructions. The integer accumulator's maximum magnitude is below
2^46 (64 terms each below 2^40). Restoring the binary exponent, trained scale
multiplication and trained bias addition happen after the integer sum.
RMSNorm and signed-log preprocessing still use floating point outside the dot loop.

No AVR toolchain was available, and no Arduino build, flashing or target RAM
claim has been made. In particular, float64 preprocessing on the workstation
must be revalidated for AVR configurations with 32-bit double. Phase 5 is not
started. See engine-software/README.md for reproducible usage and numerical details.

## Full raw validation output

Command: python3 engine-software/validate.py --output /private/tmp/ternaryguard-phase4

```text
ACTIVE ARTIFACT VERIFICATION {"mode": "verify-active", "hashes": {"checkpoint_sha256": "c335f7d468c932c3855b190061f452ff2b5b0f3fca8276bf1bff6d674ad13774", "header_sha256": "e1d919a0049e2c492d4ce54168cf9cb223bd90b4b1b74554539089ccf4c60ee8", "preprocessing_sha256": "1339886eaa7cde612b32831057afb225b7d5f314f8ebecdc6f8d57ae99c55c7d"}, "preprocessing_hash_format": "canonical JSON SHA-256", "header": "/Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard/model_weights.h", "samples": 69040, "prediction_matches": 69040, "prediction_disagreements": 0, "max_abs_logit_error": 1.1444091796875e-05, "atol": 2e-05, "rtol": 2e-05, "artifacts_rewritten": false}
$ cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math -fstack-usage -I/private/tmp/ternaryguard-phase4/build -I/Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard /Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard/engine-software/ternary_infer.c /Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard/engine-software/runner.c -lm -o /private/tmp/ternaryguard-phase4/build/ternary_infer
$ /private/tmp/ternaryguard-phase4/build/ternary_infer --preprocess /private/tmp/ternaryguard-phase4/test_raw_f64.bin /private/tmp/ternaryguard-phase4/preprocessed_f32.bin
Processed 69040 rows (preprocessing)
$ /private/tmp/ternaryguard-phase4/build/ternary_infer --run /private/tmp/ternaryguard-phase4/test_raw_f64.bin /private/tmp/ternaryguard-phase4/logits_f32.bin
Processed 69040 rows (inference)
{
  "samples": 69040,
  "prediction_matches": 69040,
  "prediction_disagreements": 0,
  "max_abs_preprocessing_error": 0.0,
  "max_abs_logit_error": 1.1444091796875e-05,
  "logits_within_export_tolerance": true,
  "c_accuracy": 0.8108198146002318,
  "torch_accuracy": 0.8108198146002318,
  "c_macro_f1": 0.78209432162603,
  "torch_macro_f1": 0.78209432162603,
  "bashlite_tcp": {
    "total": 5555,
    "c_true_positives": 1,
    "torch_true_positives": 1
  },
  "row_identity_sha256": "b3d5c16fceb383423cdab2982d5e119280bdbed8d72246235f798e56d51be1f1",
  "frozen_artifact_hashes": {
    "checkpoint_sha256": "c335f7d468c932c3855b190061f452ff2b5b0f3fca8276bf1bff6d674ad13774",
    "header_sha256": "e1d919a0049e2c492d4ce54168cf9cb223bd90b4b1b74554539089ccf4c60ee8",
    "preprocessing_sha256": "1339886eaa7cde612b32831057afb225b7d5f314f8ebecdc6f8d57ae99c55c7d"
  }
}
```

## Benchmark raw output

```text
$ /usr/bin/time -l /private/tmp/ternaryguard-phase4/build/ternary_infer --benchmark /private/tmp/ternaryguard-phase4/benchmark_raw_f64.bin 100000
{"iterations_per_trial":100000,"cached_rows":256,"microseconds_per_inference":[3.851810,3.692030,3.789890,3.782410,3.763160,3.740170,3.715680],"workspace_bytes":1024,"raw_input_bytes":920,"output_bytes":44,"model_constant_bytes":1712,"preprocessing_constant_bytes":340,"checksum":-4212146.28}
        2.65 real         2.65 user         0.00 sys
             1867776  maximum resident set size
                   0  average shared memory size
                   0  average unshared data size
                   0  average unshared stack size
                 277  page reclaims
                   6  page faults
                   0  swaps
                   0  block input operations
                   0  block output operations
                   0  messages sent
                   0  messages received
                   0  signals received
                   1  voluntary context switches
                  25  involuntary context switches
         44162223074  instructions retired
         10724363893  cycles elapsed
             1393000  peak memory footprint

```

## Compiler, kernel and frozen-file evidence

```text
$ Commands appear inline below
$ nm -u /private/tmp/ternaryguard-phase4/build/engine.o
_frexpf
_ldexp
_ldexpf
_log1p

$ cat /private/tmp/ternaryguard-phase4/build/engine.su
engine-software/ternary_infer.c:23:tg_preprocess	112	static
engine-software/ternary_infer.c:36:tg_packed_dot	0	static
engine-software/ternary_infer.c:94:tg_infer_preprocessed	80	static
engine-software/ternary_infer.c:69:layer	144	static
engine-software/ternary_infer.c:104:tg_infer	128	static
engine-software/ternary_infer.c:109:tg_model_bytes	0	static
engine-software/ternary_infer.c:116:tg_preprocessing_bytes	0	static

$ wc -c /private/tmp/ternaryguard-phase4/build/ternary_infer
   35168 /private/tmp/ternaryguard-phase4/build/ternary_infer

Compiled packed-dot kernel:
_tg_packed_dot:                         ; @tg_packed_dot
	.cfi_startproc
; %bb.0:
                                        ; kill: def $w1 killed $w1 def $x1
	cmp	w3, #64
	b.ls	LBB1_2
LBB1_1:
	mov	w0, #-1                         ; =0xffffffff
	ret
LBB1_2:
	mov	x8, #0                          ; =0x0
	cbz	w3, LBB1_10
; %bb.3:
	mov	w9, w3
	b	LBB1_7
LBB1_4:                                 ;   in Loop: Header=BB1_7 Depth=1
	cmp	w10, #2
	b.ne	LBB1_1
; %bb.5:                                ;   in Loop: Header=BB1_7 Depth=1
	ldr	x10, [x2]
	sub	x8, x8, x10
LBB1_6:                                 ;   in Loop: Header=BB1_7 Depth=1
	add	w1, w1, #1
	add	x2, x2, #8
	subs	x9, x9, #1
	b.eq	LBB1_10
LBB1_7:                                 ; =>This Inner Loop Header: Depth=1
	ubfx	x10, x1, #2, #14
	ldrb	w10, [x0, x10]
	ubfiz	w11, w1, #1, #2
	lsr	w10, w10, w11
	and	w10, w10, #0x3
	cmp	w10, #1
	b.gt	LBB1_4
; %bb.8:                                ;   in Loop: Header=BB1_7 Depth=1
	cbz	w10, LBB1_6
; %bb.9:                                ;   in Loop: Header=BB1_7 Depth=1
	ldr	x10, [x2]
	add	x8, x10, x8
	b	LBB1_6
LBB1_10:
	mov	w0, #0                          ; =0x0
	str	x8, [x4]
	ret
	
PASS: no floating-point or multiply instructions in packed-dot kernel.

Protected files compared with SHA-256 recorded before Phase 4:

UNCHANGED model_weights.h e1d919a0049e2c492d4ce54168cf9cb223bd90b4b1b74554539089ccf4c60ee8

UNCHANGED model/feature_config.json 9922f3336cc0aa9a268a81954d1c0592e84c92959c99a7c836c66d36ac3856f8

UNCHANGED model/export_manifest.json b104b4f127976dcb21e2c57f47dbe53ce5f3b88af5cae8074d8168beaf53949b

UNCHANGED model/active_model.json 1e0ac583b95827c07ded92621c091a4420c31655d0002755009cb8c6b8e9e994

UNCHANGED model/golden_vectors.npz 57b71bdb4f7db88843c3718487e1935f95d08e00d693a66b7900ca55a49a269a

UNCHANGED checkpoints/phase3_final_seed42/ternary_2bit.pt c335f7d468c932c3855b190061f452ff2b5b0f3fca8276bf1bff6d674ad13774

UNCHANGED model/train.py 0da9dc2cb3308e7c3a1ee8b0bc4e345b434f9bb65524c502f2de85ed3367de66

UNCHANGED model/data_pipeline.py b8dc0eeb35d51d798ff94e6960266e510faa7f19530aa1c36315d884c83b0a61

UNCHANGED model/ternary_linear.py 808340594d84c71995c0fe765f695134484e8a64d8a7c5d663819fb855653c58

UNCHANGED results.csv 9c9c70a5da326e6c405aab683e3631f0313c59a455032476e97b1e8abda4072d

```

## Address/undefined sanitizer run

```text
$ python3 engine-software/build.py --output /private/tmp/ternaryguard-phase4/sanitized --sanitize
$ /private/tmp/ternaryguard-phase4/sanitized/ternary_infer --run /private/tmp/ternaryguard-phase4/test_raw_f64.bin /private/tmp/ternaryguard-phase4/sanitized-logits.bin
Processed 69040 rows (inference)
Exit code: 0
Sanitized logits equal normal-build logits byte for byte: True

```

## Full test suite: 91 before, 96 after

```text
$ python3 -m pytest model/tests/ -v --tb=short
============================= test session starts ==============================
platform darwin -- Python 3.12.7, pytest-7.4.4, pluggy-1.0.0 -- /opt/anaconda3/bin/python3
cachedir: .pytest_cache
rootdir: /Users/aryanshukla/Desktop/MAIN/02_Projects/Active/TernaryGuard
plugins: anyio-4.2.0
collecting ... collected 96 items

model/tests/test_active_artifact_gates.py::test_verify_active_cli_catches_corrupted_header_and_never_rewrites PASSED [  1%]
model/tests/test_active_artifact_gates.py::test_verify_active_rejects_other_hash_mismatches[checkpoint] PASSED [  2%]
model/tests/test_active_artifact_gates.py::test_verify_active_rejects_other_hash_mismatches[preprocessing] PASSED [  3%]
model/tests/test_active_artifact_gates.py::test_verify_active_evaluates_existing_header_even_if_hash_is_updated PASSED [  4%]
model/tests/test_active_artifact_gates.py::test_legacy_plot_cli_refuses_invalidated_csv PASSED [  5%]
model/tests/test_active_artifact_gates.py::test_legacy_plot_cli_flag_renders_warning_into_pngs PASSED [  6%]
model/tests/test_c_engine.py::test_integer_dot_all_codes_unaligned_rows_and_large_signed_values PASSED [  7%]
model/tests/test_c_engine.py::test_integer_dot_rejects_reserved_code_and_oversized_input PASSED [  8%]
model/tests/test_c_engine.py::test_preprocessing_signed_values_and_nonfinite_rejection PASSED [  9%]
model/tests/test_c_engine.py::test_frozen_golden_vectors_match_logits_and_predictions PASSED [ 10%]
model/tests/test_c_engine.py::test_cli_rejects_partial_raw_record PASSED [ 11%]
model/tests/test_data_pipeline.py::TestFeatureNames::test_count PASSED   [ 12%]
model/tests/test_data_pipeline.py::TestFeatureNames::test_unique PASSED  [ 13%]
model/tests/test_data_pipeline.py::TestFeatureNames::test_format PASSED  [ 14%]
model/tests/test_data_pipeline.py::TestLabelNames::test_count PASSED     [ 15%]
model/tests/test_data_pipeline.py::TestLabelNames::test_benign_is_zero PASSED [ 16%]
model/tests/test_data_pipeline.py::TestLabelNames::test_all_keys_present PASSED [ 17%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_benign PASSED [ 18%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_mirai_ack PASSED [ 19%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_mirai_udpplain PASSED [ 20%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_mirai_udp PASSED [ 21%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_bashlite_combo PASSED [ 22%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_gafgyt_alias PASSED [ 23%]
model/tests/test_data_pipeline.py::TestLabelDetection::test_unknown PASSED [ 25%]
model/tests/test_data_pipeline.py::TestDatasetLoading::test_missing_directory_returns_empty PASSED [ 26%]
model/tests/test_data_pipeline.py::TestDatasetLoading::test_load_synthetic_csvs PASSED [ 27%]
model/tests/test_data_pipeline.py::TestPreprocessing::test_scaler_output_shape PASSED [ 28%]
model/tests/test_data_pipeline.py::TestPreprocessing::test_scaler_zero_mean PASSED [ 29%]
model/tests/test_data_pipeline.py::TestPreprocessing::test_scaler_unit_variance PASSED [ 30%]
model/tests/test_data_pipeline.py::TestPreprocessing::test_transform_without_fit PASSED [ 31%]
model/tests/test_data_pipeline.py::TestSplits::test_split_proportions PASSED [ 32%]
model/tests/test_data_pipeline.py::TestSplits::test_labels_preserved PASSED [ 33%]
model/tests/test_data_pipeline.py::TestSplits::test_split_types PASSED   [ 34%]
model/tests/test_data_pipeline.py::TestDataLoaders::test_loader_creation PASSED [ 35%]
model/tests/test_data_pipeline.py::TestDataLoaders::test_loader_batch_shape PASSED [ 36%]
model/tests/test_data_pipeline.py::TestFeatureConfig::test_save_load_roundtrip PASSED [ 37%]
model/tests/test_data_pipeline.py::TestFeatureSelection::test_mutual_info_count PASSED [ 38%]
model/tests/test_data_pipeline.py::TestFeatureSelection::test_variance_count PASSED [ 39%]
model/tests/test_data_pipeline.py::TestFeatureSelection::test_correlation_count PASSED [ 40%]
model/tests/test_data_pipeline.py::TestFeatureSelection::test_invalid_method_raises PASSED [ 41%]
model/tests/test_phase3_integrity.py::test_schema_known_positions PASSED [ 42%]
model/tests/test_phase3_integrity.py::test_sampling_covers_late_devices_and_rejects_bad_headers PASSED [ 43%]
model/tests/test_phase3_integrity.py::test_preprocessing_never_fits_heldout_rows[identity] PASSED [ 44%]
model/tests/test_phase3_integrity.py::test_preprocessing_never_fits_heldout_rows[signed_log1p] PASSED [ 45%]
model/tests/test_phase3_integrity.py::test_export_preserves_noninteger_bias_logits_and_exact_float_constants[False] PASSED [ 46%]
model/tests/test_phase3_integrity.py::test_export_preserves_noninteger_bias_logits_and_exact_float_constants[True] PASSED [ 47%]
model/tests/test_phase3_integrity.py::test_export_only_loads_frozen_bundle_without_dataset PASSED [ 48%]
model/tests/test_phase3_integrity.py::test_results_quotes_and_schema PASSED [ 50%]
model/tests/test_phase3_integrity.py::test_per_layer_padding_and_fp_classifier_budget PASSED [ 51%]
model/tests/test_phase3_integrity.py::test_heldout_devices_require_disjoint_sets_and_class_coverage PASSED [ 52%]
model/tests/test_phase3_integrity.py::test_feature_selection_does_not_spend_slots_on_duplicates PASSED [ 53%]
model/tests/test_phase3_integrity.py::test_golden_vectors_include_every_class PASSED [ 54%]
model/tests/test_phase3_integrity.py::test_header_compiles_as_c_and_constants_have_expected_sizes PASSED [ 55%]
model/tests/test_phase3_integrity.py::test_signed_log_transform_handles_negative_covariances_and_large_values PASSED [ 56%]
model/tests/test_phase3_integrity.py::test_scaler_config_rejects_feature_dimension_mismatch PASSED [ 57%]
model/tests/test_phase3_integrity.py::test_preprocessing_hash_survives_json_integer_label_keys PASSED [ 58%]
model/tests/test_phase3_integrity.py::test_frozen_rows_and_source_content_are_verified PASSED [ 59%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_output_values_are_ternary PASSED [ 60%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_scale_is_absmean PASSED [ 61%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_threshold_logic PASSED [ 62%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_ste_gradient_identity PASSED [ 63%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_gradient_is_nonzero PASSED [ 64%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_all_zeros_weight PASSED [ 65%]
model/tests/test_ternary_linear.py::TestTernaryQuantize::test_all_same_sign PASSED [ 66%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_output_shape_2d PASSED [ 67%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_output_shape_single_sample PASSED [ 68%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_output_shape_no_bias PASSED [ 69%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_bias_parameter_registration PASSED [ 70%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_gradients_nonzero_weight PASSED [ 71%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_gradients_nonzero_bias PASSED [ 72%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_weight_distribution_balanced PASSED [ 73%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_quantized_weights_are_int8_ternary PASSED [ 75%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_scale_is_positive PASSED [ 76%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_deterministic_forward PASSED [ 77%]
model/tests/test_ternary_linear.py::TestTernaryLinear::test_repr PASSED  [ 78%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_output_shape PASSED [ 79%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_rms_near_one PASSED [ 80%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_gradient_flows PASSED [ 81%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_gamma_gradient PASSED [ 82%]
model/tests/test_ternary_linear.py::TestRMSNorm::test_identity_at_unit_rms PASSED [ 83%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_forward_shape PASSED [ 84%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_forward_multiclass PASSED [ 85%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_training_reduces_loss PASSED [ 86%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_count_parameters PASSED [ 87%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_packed_size_below_flash PASSED [ 88%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_no_rmsnorm PASSED [ 89%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_fp_output_layer PASSED [ 90%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_single_hidden_layer PASSED [ 91%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_deep_model PASSED [ 92%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_estimate_packed_size_includes_rmsnorm PASSED [ 93%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_quantize_paths_agree PASSED [ 94%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_ram_fits_2kb PASSED [ 95%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_pack_ternary_2bit_roundtrip PASSED [ 96%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_export_weights_header PASSED [ 97%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_pack_row_major_not_transposed PASSED [ 98%]
model/tests/test_ternary_linear.py::TestTernaryMLP::test_decode_matches_spec_snippet PASSED [100%]

============================= 96 passed in 40.94s ==============================

```
