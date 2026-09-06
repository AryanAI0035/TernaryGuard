# TernaryGuard

> **One Ternary AI Model. Three Silicon-to-Software Deployments. One Cybersecurity Mission.**

A 1.58-bit ternary neural network (`weights ∈ {-1, 0, 1}`) for IoT botnet/DDoS intrusion detection, deployed across three radically different targets to prove a single engineering thesis:

**Ternary weights eliminate the need for multipliers — the same tiny model runs as a software reference, survives inside 2KB of RAM on a bare microcontroller, and becomes custom accelerator silicon in an FPGA.**

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
# Clone
git clone https://github.com/yourusername/TernaryGuard.git
cd TernaryGuard

# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Arduino (requires PlatformIO)
cd engine-arduino && pio run

# FPGA (requires Vivado)
cd engine-fpga && vivado -mode batch -source scripts/run_sim.tcl

# Dashboard
cd dashboard/backend && uvicorn main:app --reload
cd dashboard/frontend && npm start
```

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
