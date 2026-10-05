# TernaryGuard

**A completed embedded machine learning research prototype for IoT botnet classification, built in PyTorch, C and Arduino Nano.**

TernaryGuard trains an 11-class neural network on N-BaIoT traffic statistics, packs its weights into two bits each, and runs the frozen model through an integer ternary dot product on a workstation and an actual ATmega328P Nano. The dot product decodes **0, +1, −1** and uses add/subtract/skip; preprocessing, normalization and final scaling still use floating point.

The delivered scope is the trained model, verified export, C engine and physically tested Nano firmware. **FPGA is optional future work**, preserved under [future-scope/](future-scope/README.md). No FPGA deployment, Vivado timing closure or dashboard implementation is claimed.

## Verified results

| Result | Value | Evidence and scope |
|---|---:|---|
| Frozen network | 20 → 64 → 32 → 11 | [Active identity](model/active_model.json); ternary layers with RMSNorm |
| Held-out-device test accuracy / macro-F1 | **81.08% / 0.7821** | [Final run](docs/benchmarks/phase3_final_seed42/ternary_2bit.json), 69,040 rows from device 9 |
| Packed model parameters | **1,696 B**, 9.13× smaller than FP32 | [Accounting](docs/phase3_corrections.md): 920 B packed weights plus biases, scales and gamma; exported constants including metadata total 1,712 B |
| Workstation prediction parity | **69,040 / 69,040**, zero disagreements | [C validation](docs/phase4_validation.md), frozen PyTorch reference |
| Workstation median inference | **3.763 μs** | [Apple M3 benchmark](docs/phase4_validation.md), warm cache, preprocessing included, I/O excluded |
| Nano application flash / static SRAM | **8,076 B / 1,212 B** | [Physical report](docs/phase5_hardware_validation.md), avr-size and flash verification |
| Nano observed SRAM high-water mark | **1,321 B** | Physical SRAM canary measurement; approximate, 727 B untouched gap |
| Nano median inference | **36.308 ms** | Physical micros() measurement on 330 stratified rows, preprocessing included, UART excluded |
| Nano prediction parity | **5,855 unique rows**, zero disagreements | 330 stratified + all 5,555 TCP rows; 5,885 transactions with 30 overlapping rows |

**Known limitation:** BASHLITE TCP recall is **1/5,555 (0.018%)** on the frozen test split. The Nano reproduces the same sole detection at test index 59782. The model effectively fails to detect this attack class; overall accuracy must not hide that. This is a completed research prototype, not a production-ready intrusion detector.

## How it works

```text
N-BaIoT CSV statistics (115 columns)
  → 20 training-selected features
  → signed-log1p + training-fit StandardScaler
  → RMSNorm / ternary layer / ReLU
  → RMSNorm / ternary layer / ReLU
  → ternary output layer → 11 logits → class
```

Training uses devices 1–7 (411,915 rows), validation device 8 (69,045), and test device 9 (69,040). Feature selection, redundancy filtering and scaler fitting use training rows only. Model selection uses validation metrics. [Methodology and corrections](docs/phase3_corrections.md) explain the split and rejected earlier results.

The C and AVR engines use a shared exponent to convert activations to bounded int64 values before accumulating packed weights. The inner dot product has no multiply or floating-point operations; scaling and the rest of the network remain floating point. Nano weights and constants reside in PROGMEM. The Nano performs log/scaler preprocessing in float32; its precision and physical coverage are documented separately from the workstation's float64 preprocessing.

Inputs are already extracted N-BaIoT traffic statistics. Packet capture, online feature extraction and live network blocking are outside the delivered scope.

## Run from a fresh clone

Requires Python **3.11+**, a C compiler (`cc`), and the dependencies below. No Nano, raw dataset or FPGA tools are needed for the bundled engineering replay or core regression tests.

```sh
git clone https://github.com/AryanAI0035/TernaryGuard.git
cd TernaryGuard
python3 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python3 -m pip install -r requirements.txt
python3 scripts/demo.py
python3 -m pytest model/tests/ -v --tb=short
```

The demo compares the frozen checkpoint, existing exported header and compiled C engine on **264 bundled preprocessed engineering vectors**. It is an inference replay, not a fresh accuracy evaluation or a live traffic monitor. C/AVR compilation tests require a compatible `cc`; run those on macOS, Linux or WSL. The default/core suite contains **99 tests**.

The exact frozen checkpoint, data manifest and split identities are included under `checkpoints/phase3_final_seed42/`; the raw dataset and other training checkpoints are excluded. For full 69,040-row verification, dataset download, C benchmarking and Nano flashing, see [reproduction instructions](docs/reproduce.md).

## Repository

| Path | Purpose |
|---|---|
| [model/](model/) | Training, preprocessing, artifact validation, golden vectors and core tests |
| [model_weights.h](model_weights.h) | Exact frozen packed weights and trained constants |
| [checkpoints/phase3_final_seed42/](checkpoints/phase3_final_seed42/) | Small frozen reference bundle; no retraining needed |
| [engine-software/](engine-software/README.md) | C inference engine, guarded build, full-dataset parity and benchmarks |
| [engine-arduino/](engine-arduino/README.md) | Nano firmware, PROGMEM, serial protocol and physical validation |
| [docs/](docs/README.md) | Evidence, methodology, reproduction and [résumé wording](docs/resume.md) |
| [research/](research/) | Dataset preparation, exploratory analysis and diagnostic scripts |
| [future-scope/](future-scope/README.md) | Optional FPGA work and dashboard placeholders |

Original implementation proposals are [archived](docs/archive/original_implementation_plan.md). Historical phase names and budgets are retained as records; current completion scope and measured results are defined above. Invalidated CSVs remain explicitly labeled and cannot produce unmarked plots through the plotting tool.

## License and dataset

Project code and the frozen model are released under the [MIT License](LICENSE). N-BaIoT is an external research dataset credited to Meidan et al.; its data is not included here or relicensed by this repository. The download helper uses the UCI archive. Results describe this particular frozen experiment and validation scope.
