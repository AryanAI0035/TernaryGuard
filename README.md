# TernaryGuard

> **One Ternary AI Model. Three Silicon-to-Software Deployments. One Cybersecurity Mission.**

A 1.58-bit ternary neural network (`weights ∈ {-1, 0, 1}`) for IoT botnet/DDoS intrusion detection, being developed for three targets to prove a single engineering thesis:

**Ternary weight dot products use add/subtract/skip. The project aims to validate the same compact model in desktop C, an Arduino Nano with 2KB RAM, and FPGA simulation/synthesis.**

Current status: phase-3 pipeline corrections and export validation complete; class-level model quality remains unresolved. Engine inference and the dashboard are not implemented yet; hardware performance has not been measured. The active FPGA scope is simulation/synthesis without a physical board. See [corrected results and limitations](docs/phase3_corrections.md), [phase-3 audit](docs/phase3_review.md) and [architecture budget](docs/architecture_budget.md).

---

## Architecture

```
                     ┌─────────────────────────────┐
                     │   Ternary Model (trained)    │
                     │   absmean quant + STE        │
                     └──────────────┬───────────────┘
                                    │  same weights, same math
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
 ┌───────────────┐         ┌────────────────┐         ┌────────────────┐
 │ SOFTWARE       │         │ ARDUINO NANO   │         │ FPGA (Vivado)  │
 │ C reference    │         │ ATmega328P     │         │ Basys 3        │
 │ engine         │         │ 32KB / 2KB     │         │ xc7a35t (sim)  │
 └───────┬────────┘         └───────┬────────┘         └───────┬────────┘
         │                          │                          │
         └──────────────┬───────────┴──────────────┬───────────┘
                         ▼                          ▼
                 Serial / UART              Simulation replay
                         │                          │
                         └────────────┬─────────────┘
                                       ▼
                    ┌──────────────────────────────────┐
                    │  FastAPI backend + React dashboard │
                    │  live classifications, 3-way race  │
                    └──────────────────────────────────┘
```

## Headline Numbers

| Metric | Software | Arduino Nano | FPGA (Basys 3) |
|--------|----------|-------------|----------------|
| Accuracy | — | — | — |
| Latency | — | — | — |
| Memory | — | — | — |
| Power | — | — | — |

> *Numbers will be populated as each phase completes.*

## Repository Structure

```
TernaryGuard/
├── research/              # Notebooks, quantization experiments
├── model/                 # Training code, checkpoints, tests
├── engine-software/       # C/Python reference inference
├── engine-arduino/        # Embedded C, PlatformIO (ATmega328P)
├── engine-fpga/           # Verilog RTL, Vivado, testbenches
├── dashboard/
│   ├── backend/           # FastAPI + WebSockets
│   └── frontend/          # React + Recharts/D3
├── docs/
│   ├── benchmarks/        # Comparison tables, plots
│   └── diagrams/          # Architecture diagrams
├── results.csv            # Experiment tracking (all phases)
├── plot_results.py        # Benchmark visualization
└── requirements.txt       # Python dependencies
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 research/download_nbaiot.py
python3 -m pytest -q
python3 model/train.py --run-id my_phase3_run
python3 model/finalize_phase3.py --runs checkpoints/my_phase3_run \
  --output-dir checkpoints/my_final_run
python3 plot_results.py --phase 3 --results checkpoints/my_final_run/results.csv
```

The default run uses 550,000 samples across nine devices: devices 1–7 for
training, 8 for validation, and 9 for testing. A signed log transform (`sign(x) * log1p(abs(x))`) handles skewed statistics,
including negative covariance values. Selection and normalization fit training
data only. Use `--feature-transform identity` for the comparison without the log transform. Mutual-information ranking excludes near-duplicate features
with absolute correlation at least 0.98, measured on the same training-only
selection sample. All three precision variants share preprocessing and architecture.

Checkpoints and detailed reports are saved under `checkpoints/<run-id>/`.
Finalization requires nonzero recall for every validation class, chooses the
highest validation macro-F1 among eligible widths, and trains FP32/INT8 baselines
with the selected architecture and matching maximum epoch budget. This is a
minimal sanity gate, not a deployment acceptance threshold. The checked-in
reference is identified in `model/active_model.json`.
Results append to `results.csv`; plots default to the latest run. INT8 means
simulated weight-only quantization with FP32 execution, not an INT8 runtime.

Export an existing bundle without refitting or accessing the dataset:

```bash
python3 model/train.py --export-only \
  --checkpoint checkpoints/my_final_run/ternary_2bit.pt \
  --output-dir checkpoints/my_final_run/export
```

Recheck the exported header on all frozen test rows and generate golden vectors
covering every class:

```bash
python3 model/validate_export.py \
  --checkpoint checkpoints/my_final_run/ternary_2bit.pt
```

Each run preserves all trained models, ordered features/scaler, source hashes,
split row identities, training history, per-class and binary metrics, and export
metadata. Legacy bare state dictionaries are deliberately rejected by export.
`--ablation-only` creates a separate run and does not change production artifacts.
Run IDs/output directories must be new for training. Legacy phase-3 results are
archived as invalidated under `docs/benchmarks/` and must not be mixed with new
held-out-device results.

## Dataset

**N-BaIoT** — purpose-built for IoT botnet detection (Mirai, BASHLITE variants).

## Tech Stack

| Layer | Tools |
|-------|-------|
| Model Training | PyTorch, NumPy, scikit-learn |
| Software Engine | C (from-scratch, no deps) |
| Embedded | AVR-GCC, PlatformIO, Arduino Nano |
| FPGA | Verilog, Xilinx Vivado, XSim |
| Backend | FastAPI, WebSockets |
| Frontend | React, Recharts/D3 |
| Tracking | CSV logging + matplotlib |

## License

MIT

---

*Built to prove that the same model can run everywhere — from desktop to 2KB of RAM to custom silicon.*
