# TernaryGuard phase 3 review

Reviewed 2026-09-30 at commit `4bcea726075afcc07bfe7ed58967aca819747faa`.

**Assessment: phase 3 has working training code and a trained model, but should not be considered complete before the data evaluation and export issues below are resolved.** This review did not change the implementation, checkpoint, feature configuration, or recorded results.

## Objective and current state

The project trains one compact ternary-weight intrusion classifier and intends to run equivalent inference in a desktop C reference, an ATmega328P Arduino Nano, and an FPGA design. A later dashboard will compare correctness, latency, memory, and power. The central implementable claim is that ternary weight dot products use add/subtract/skip operations.

The current model is `20 → 32 → 16 → 11`, with RMSNorm before both hidden linear layers and ReLU after them. It has 1,328 ternary weights, 59 biases, and 52 normalization parameters: 1,439 trainable parameters. The ternary weights alone occupy 332 bytes with the documented 2-bit packing.

| Phase | Evidence present | Assessment |
|---|---|---|
| 0: foundations | Repository structure, requirements, PlatformIO configuration, Vivado script scaffold | Structure exists; hardware/toolchain verification is not demonstrated by the current tree. |
| 1: quantization | Absmean thresholding, STE, RMSNorm, packing, unit tests, method note | Implemented; some explanatory claims need correction. |
| 2: dataset/design | Downloader, local data, EDA script/images, selection/scaling/split utilities, budget document | Implemented with substantive schema and evaluation flaws. |
| 3: training/ablation | FP32, simulated INT8, ternary training, five widths, results file, local checkpoint/header | Training performed; export and evaluation gates remain open. |
| 4–7: engines/dashboard | Software and dashboard placeholders; Arduino serial placeholder; FPGA scaffold | Not implemented yet, consistent with the current phase. |

The latest commit correctly adds RMSNorm to the FP32 baseline; INT8 copies that architecture. This fixes the earlier normalization confound.

Recorded headline accuracies are FP32 90.2764%, INT8 89.8636%, ternary 85.5000%. These are provisional: the CSV is malformed, evaluation leaks test information, and only the ternary checkpoint is saved locally. The five recorded ternary widths run from `[16, 8]` to `[64, 32]`, with accuracies approximately 81.92%–86.54%.

## Findings in priority order

### 1. P1 — Export changes the trained model and sharply reduces accuracy

Location: `model/ternary_linear.py:554`.

Training uses float biases, but export rounds them directly to int8 without a bias scale or quantization-aware training. This is a new numerical transformation, not lossless serialization. In the saved model, 41 of 59 biases round to zero.

On the reconstructed 110,000-sample test split:

| Evaluation | Accuracy |
|---|---:|
| Saved checkpoint with saved preprocessing | 85.4973% |
| Same model with exactly the exporter's bias rounding | 60.3182% |

Predictions disagree on 40.6136% of test samples. This comparison isolates bias rounding; it is not a hardware or C-engine measurement. All three packed weight arrays decode exactly to the checkpoint's ternary weights, and regenerating the header produces an identical file.

Fix: preserve FP32 biases for the initial software reference, or define, calibrate, and validate a scaled fixed-point format. Require exported-artifact inference parity before freezing the model. Preserving FP32 biases changes the current model-constant estimate from 611 to **788 bytes**, excluding metadata and preprocessing constants.

### 2. P1 — Test information enters feature selection and normalization

Location: `model/train.py:378–398`.

Mutual-information selection uses every sample and label, then StandardScaler fits all samples, and only afterwards does the code create train/validation/test splits. The test set therefore influences the learned preprocessing, including supervised feature selection. Current scores cannot serve as a clean held-out evaluation; this review does not estimate the magnitude of score inflation.

Fix: establish raw splits first, fit selection and scaling exclusively on training data, and apply frozen transforms to validation/test data. Retrain all compared models and ablations under the corrected protocol.

### 3. P1 — Class caps effectively discard seven devices

Location: `model/data_pipeline.py:104–144`.

Files are processed in filename order and the global 50,000-per-class cap is filled before later devices are considered. With the present flat dataset, device 1 supplies all 50,000 examples for eight classes. Device 2 only fills the remaining benign, GAFGYT junk, and GAFGYT scan quotas. Devices 3–9 contribute nothing.

Thus the 550,000-sample experiment is predominantly device 1, with some device 2 samples. Random row splitting tests performance within those captures; it does not demonstrate unseen-device generalization. Capture/time correlation is an additional evaluation risk, not quantified here.

Fix: retain source-device/capture identity, sample deliberately across devices, and define whether the intended result is within-device or unseen-device detection. Use a device/capture-held-out evaluation for the latter, while checking class coverage. Report the balanced sampling policy and its difference from deployment prevalence.

### 4. P1 — Export-only can overwrite the preprocessing paired with a checkpoint

Location: `model/train.py:617–623`, `model/train.py:637–680`.

Even `--export-only` reruns feature selection and scaling and overwrites `feature_config.json` before loading the old checkpoint. Mutual-information estimation has no explicit random seed, so feature membership/order may change. Checking only that there are 20 features does not verify that they are the same features in the same order. `--ablation-only` has the same preprocessing overwrite risk. If the checkpoint is absent, the script exports random weights and still announces phase completion.

Fix: save architecture, ordered feature schema, scaler, labels, seed, and provenance together with the checkpoint. Export must load that bundle without refitting preprocessing and fail clearly when trained weights are missing.

### 5. P2 — Results CSV is invalid

Location: `results.csv:2` onward.

The file contains literal backslash-escaped quotes around fields containing commas. Standard CSV uses ordinary quote delimiters, not those backslashes. The header has 21 fields; data rows parse as 22–24 fields. `pandas.read_csv` raises `Expected 22 fields in line 4, saw 24`, blocking the plotting script.

Fix: regenerate the file using the existing CSV writer and validate the schema. The logger should also create a header when starting a new results file. Add run identifiers and distinguish estimated hardware budgets from measured results.

### 6. P2 — The 115-feature schema is wrong despite having the right count

Location: `model/data_pipeline.py:22–27`.

Comparison with the dataset's own `data/raw/nbaiot/features.csv` shows 103 of 115 generated names differ. Some differences are names such as variance versus std; others are structural: the generated H block has five statistics rather than the actual three, shifting subsequent meanings. Fourteen of the 20 selected feature names are wrong. For example, index 28 is labelled `H_L1_radius`, but is actually `H_L0.01_mean`.

Positional selection still selects the same numeric columns within this current pipeline, so this naming issue alone does not establish an accuracy change. It does invalidate feature interpretation and can break future name-based extraction and cross-format ingestion.

Fix: use the authoritative ordered schema, validate input headers/column counts, and save both names and indices. Existing tests check count and uniqueness against the implementation's own generated schema, so they miss the error.

### 7. P2 — Aggregate scores hide failure to identify one attack class

Location: `model/train.py:evaluate` and phase 3 result reporting.

The saved checkpoint has **0.0000 precision, recall, and F1 for BASHLITE TCP**, with 10,000 test examples. BASHLITE UDP precision is approximately 0.4996. This is a failure of the promised 11-class classification task; it does not by itself mean those TCP attacks are classified as benign.

Fix: save per-class reports and confusion matrices, separately report benign-versus-attack performance, and investigate class confusion after correcting the split and preprocessing. Define acceptance criteria before selecting the final architecture. Macro and weighted metrics happen to agree here because test support is balanced.

### 8. P2 — INT8 and deployment resource claims exceed what is measured

Locations: `model/train.py:237–285`, `model/ternary_linear.py:estimate_ram_usage`, `docs/quantization_method.md`.

INT8 rounds weights onto an 8-bit grid and immediately stores dequantized FP32 tensors. It is a valid simulated weight-quantization accuracy experiment, but not dynamic INT8 inference, and its 1,784-byte size is a hypothetical packed-constant size rather than this Python model's actual storage.

Similarly, ternary dot products can avoid weight multiplications, but RMSNorm still squares, averages, takes a square root, divides, and multiplies by gamma. Output scaling multiplies each output element: 59 scale multiplications for this architecture, not three scalar arithmetic operations. The current PyTorch path uses floating-point operations. No full fixed-point inference contract, overflow policy, or measured no-DSP implementation exists yet.

The 360-byte RAM and roughly 12KB flash figures are estimates. The Arduino engine is a placeholder, so stack peaks, serial overhead, and final code size are not measured. Exported scale constants also lack `TGPROGMEM`, despite the documented policy that all trained constants use it.

Fix: label simulation/estimates explicitly; define numerical formats and normalization implementation before embedded/FPGA work; validate final resource use with builds and measurements. Keep the thesis scoped to multiplier-free ternary dot products until broader claims are demonstrated.

## Other completion gaps

- Only the production ternary state dictionary is saved; FP32 and ablation models, preprocessing provenance, training histories, and seeds are not preserved. Full experiment reruns are not reproducible from the recorded artifacts alone.
- Operation counts and the planned `docs/benchmarks/accuracy_ablation.csv` are absent. Model sizing helpers also omit an optional FP32 classifier and round total packed weights globally instead of rounding each separately exported layer; these helper issues do not affect the present all-ternary architecture's divisible-by-four sizes.
- `--skip-download` is accepted but unused; the training script never downloads data.
- The initial plan promises physical FPGA deployment, while the FPGA README specifies simulation-only. Resolve that scope explicitly before phase 6. Root README dashboard launch commands refer to files that do not exist yet.
- The EDA script can use synthetic fallback data. The existing images lack an accompanying provenance record establishing the input data used to generate them.

## Validation performed

- Read the implementation plan, README, model/data/training code, tests, research scripts, memory/method notes, result plotting code, and deployment scaffolds; inspected recent history and local artifacts.
- Ran the existing suite: **68 passed in 3.13 seconds**. Pytest was installed into a temporary directory; repository dependencies were not changed. No integration tests currently exercise training-to-export numerical parity or leakage-free preparation.
- Loaded the local checkpoint and reconstructed the current saved preprocessing and deterministic 70/10/20 split of 550,000 samples, without retraining or refitting feature selection.
- Evaluated the original and rounded-bias models on all 110,000 test samples; inspected per-class metrics.
- Verified checkpoint/header regeneration and packed-weight decoding, CSV parsing failure, the dataset feature-name mismatch, and file-order sampling behavior.
- Did not retrain FP32/INT8/ablation models or build hardware engines. PlatformIO, AVR-GCC, and Vivado were not found on the current command path.

## Recommended phase 3 completion order

1. Correct schema and device/capture sampling; freeze raw split identities.
2. Fit feature selection/scaling on training data only and bundle them with model provenance.
3. Retrain comparable baselines and ablations with recorded seeds; save per-class and binary-detection metrics.
4. Preserve bias precision or validate a calibrated quantized representation; check header-versus-PyTorch predictions end to end.
5. Repair result serialization, document estimates honestly, and save golden input/output vectors for phase 4.

The current implementation provides a useful foundation, but freezing its present exported model would carry a confirmed accuracy regression into every downstream engine.
