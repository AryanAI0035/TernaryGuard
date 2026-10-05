> Historical proposal, preserved for context. It contains aspirational targets and superseded budgets. For completed scope and verified measurements, use the root README and docs/README.md. FPGA and dashboard are optional future work.

# TernaryGuard — Complete Implementation Plan
### One Ternary AI Model. Three Silicon-to-Software Deployments. One Cybersecurity Mission.

> Current scope clarification (2026-09-30): FPGA work targets simulation and
> synthesis for Basys 3; there is no physical FPGA board in the present scope.
> Physical-board demos below are future extensions. Phase-3 correctness uses
> FP32 activations/biases and multiplier-free ternary dot products. Full-network
> fixed-point arithmetic and measured resource claims remain later validation
> work. See `docs/architecture_budget.md` and `docs/phase3_review.md`.

---

## 1. Project Vision

TernaryGuard is a 1.58-bit (ternary, weights ∈ {-1, 0, 1}) neural network trained once for IoT botnet/DDoS intrusion detection, then deployed across three radically different targets to prove the same thesis at every layer of the stack:

> **Ternary weights eliminate the need for multipliers — which means the same tiny model can run as a full-precision reference on a workstation, survive inside 2KB of RAM on a bare microcontroller, and become custom accelerator silicon in an FPGA — with the entire pipeline benchmarked side by side.**

The "jaw-drop" is not feature count. It's coherence: a single trained model, three deployment stories, one provable engineering thesis, backed by real numbers at every stage.

**End deliverable:** a public GitHub repo + live demo dashboard + research-style PDF report + demo video, structured so a recruiter or engineer can understand the whole system in under 2 minutes and then go as deep as they want.

---

## 2. System Architecture Overview

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
 │ reference      │         │ ATmega328P     │         │ custom RTL     │
 │ engine (C/     │         │ 32KB flash /   │         │ ternary MAC    │
 │ Python)        │         │ 2KB RAM        │         │ array          │
 └───────┬────────┘         └───────┬────────┘         └───────┬────────┘
         │                          │                          │
         └──────────────┬───────────┴──────────────┬───────────┘
                         ▼                          ▼
                 Serial / UART / BLE       Live traffic feature stream
                         │                          │
                         └────────────┬─────────────┘
                                       ▼
                    ┌──────────────────────────────────┐
                    │   FastAPI backend + React dash     │
                    │  live classifications, 3-way race  │
                    │  latency / RAM / flash / power      │
                    └──────────────────────────────────┘
```

---

## 3. Tech Stack Summary

| Layer | Tools |
|---|---|
| Model training | PyTorch, NumPy, scikit-learn (preprocessing) |
| Dataset | N-BaIoT or CICIDS2017 (network flow features) |
| Software engine | Python reference + optimized C (SIMD where useful) |
| Embedded | AVR-GCC, avrdude, PlatformIO, Arduino Nano (ATmega328P) |
| Hardware | Verilog/SystemVerilog, Xilinx Vivado, your FPGA dev board |
| Backend | FastAPI, WebSockets |
| Frontend | React, Recharts/D3 for live charts |
| Tracking | Weights & Biases or simple CSV logging |
| Writeup | LaTeX or Markdown → PDF (Pandoc) |
| Version control | Git/GitHub, structured as a monorepo with clear subfolders |

---

## Phase 0 — Foundations & Environment Setup
**Duration:** Days 1–3 | **Effort:** ~5%

**Objectives**
- Every toolchain installed and verified before any real work starts, so nothing blocks you mid-sprint later.

**Tasks**
1. Set up Python environment (conda/venv): PyTorch, numpy, pandas, sklearn, matplotlib.
2. Install AVR toolchain: `avr-gcc`, `avrdude`, PlatformIO (recommended over raw Arduino IDE for reproducible builds).
3. Verify Vivado license/install, confirm your FPGA board is detected, run a trivial "blink" bitstream to confirm the full synthesis → implementation → program flow works end-to-end.
4. Create GitHub repo with monorepo structure:
   ```
   ternaryguard/
     research/         # notebooks, quantization experiments
     model/             # training code, checkpoints
     engine-software/   # C/Python reference inference
     engine-arduino/    # embedded C, PlatformIO project
     engine-fpga/       # Verilog RTL, Vivado project, testbenches
     dashboard/         # FastAPI + React
     docs/              # paper, diagrams, benchmark tables
   ```
5. Set up experiment tracking (W&B or a simple `results.csv` convention) — decide now so every phase logs numbers the same way.

**Deliverables:** working repo skeleton, verified toolchains, one committed "hello world" per target (Python script runs, Arduino blinks, FPGA LED bitstream loads).

**Risk check:** FPGA toolchain issues are the #1 silent time-sink in projects like this — confirming the full flow now, before you have real RTL to debug, saves you from debugging two problems at once later.

---

## Phase 1 — Research: Ternary Quantization Theory
**Duration:** Week 1 (first half) | **Effort:** ~10%

**Objectives**
- Deeply understand *why* ternary weights work and *why* they remove the multiplier requirement — this becomes both your model's foundation and your FPGA's core design argument.

**Tasks**
1. Study BitNet b1.58 methodology: absmean weight quantization, straight-through estimator (STE) for gradients through the non-differentiable quantization step, RMSNorm-based activation scaling.
2. Implement a `TernaryLinear` layer from scratch in PyTorch:
   - Forward: quantize weights to {-1, 0, 1} via `scale = mean(|W|)`, threshold at `0.5 * scale`.
   - Backward: STE — gradients flow through as if quantization were identity.
3. Unit-test the layer in isolation: confirm forward pass output shapes, confirm gradients are non-zero and sane, confirm quantized weight distribution is roughly balanced across {-1,0,1}.
4. Write a short internal note (this becomes your paper's Methods section later) explaining the math in your own words — do this now while it's fresh, not in Phase 9.

**Deliverables:** `TernaryLinear` module with tests, a 1-page written explanation of the quantization method, sanity-check plots (weight histograms pre/post quantization).

**Why this matters for "ultimate":** this is the phase that proves you understand the *mechanism*, not just that you called a library function. Interviewers will probe here first.

---

## Phase 2 — Dataset & Model Design
**Duration:** Week 1 (second half) – Week 2 (start) | **Effort:** ~10%

**Objectives**
- Get a clean, realistic, resource-appropriate dataset and a model architecture that's honestly deployable on 2KB of RAM.

**Tasks**
1. Download and explore N-BaIoT (recommended: purpose-built for IoT botnet detection, smaller feature set, well-cited) or CICIDS2017 if you want broader attack diversity.
2. Feature selection: reduce to a compact feature vector (aim for ≤20–40 features) — this is critical because the Arduino has to hold activations for this in RAM. Use feature importance (e.g., from a quick Random Forest) to justify your selection, not guesswork.
3. Preprocess: normalize features, encode labels (binary benign/attack, or multi-class per attack type if you want a richer story), split train/val/test with attention to class imbalance (botnet datasets are usually heavily imbalanced — document how you handled this).
4. Design a small MLP or 1D-CNN sized for the constraint: e.g., input(20–40) → hidden(16–32) → hidden(8–16) → output(2 or n-classes). Calculate the theoretical ternary-packed size by hand before writing training code, so you know your target is achievable.

**Deliverables:** cleaned dataset pipeline (scripted, reproducible), documented feature selection rationale, model architecture with a hand-calculated memory budget (flash for weights, RAM for activations).

**Concrete math you should produce here:** with W weights at ~1.58 bits each, packed size ≈ `W * 1.58 / 8` bytes. For a model with ~4,000 weights, that's roughly 790 bytes — comfortably inside 32KB flash, and it tells you immediately whether your architecture is realistic before you invest in training.

---

## Phase 3 — Training & Ablation Study
**Duration:** Week 2 | **Effort:** ~10%

**Objectives**
- Produce a trained model and the comparison table that anchors your entire research story.

**Tasks**
1. Train three versions of the identical architecture: FP32 baseline, INT8 quantized (standard post-training or QAT quantization), and your ternary version.
2. Track: accuracy, precision/recall/F1 (important given class imbalance), model size in bytes, and inference FLOPs/operation count.
3. Run at least one architecture ablation (e.g., hidden layer width) to show you explored the design space, not just accepted the first config.
4. Export the final ternary model's quantized weights to a raw format your C/Arduino/Verilog code can directly consume (e.g., a header file with packed weight arrays).

**Deliverables:** `benchmarks/accuracy_ablation.csv`, trained checkpoints, an exported weight header file (`model_weights.h`) used identically by every downstream engine — this shared artifact is what makes the "three deployments, one model" claim literally true rather than just a nice tagline.

---

## Phase 4 — Software Reference Inference Engine
**Duration:** Week 3 | **Effort:** ~10%

**Objectives**
- Build the ground-truth inference implementation everything else is validated against.

**Tasks**
1. Write a from-scratch ternary inference function in C (not calling PyTorch) that reads the exported weight header and runs forward pass using only add/subtract/skip operations (no multiplication) — this is the software proof of your core thesis before you ever touch hardware constraints.
2. Cross-validate: run the same test set through this C implementation and your PyTorch model; outputs must match (within quantization-expected tolerance) — this is your correctness gate for every later port.
3. Benchmark on your workstation: latency per inference, cycles/operation count, memory footprint.
4. This same C core (minus AVR-specific memory tricks) becomes the shared logic base for the Arduino port in Phase 5 — write it with portability in mind now.

**Deliverables:** `engine-software/ternary_infer.c`, a validation script proving bit-exact (or documented near-exact) agreement with the PyTorch model, baseline latency numbers.

---

## Phase 5 — Arduino Nano Deployment (The Extreme-Constraint Flex)
**Duration:** Weeks 4–5 | **Effort:** ~20% (this is a headline phase)

**Objectives**
- Get real-time inference running inside 2KB RAM / 32KB flash — the number that makes people stop and ask "wait, how?"

**Tasks**
1. Port the Phase 4 C engine to AVR-GCC. Key adaptations:
   - Store weights in **PROGMEM** (flash) using `pgm_read_byte()` — RAM is far too scarce to hold weights.
   - Use fixed-point or int8 arithmetic throughout; avoid floating point unless absolutely unavoidable (AVR has no FPU, and float ops eat both flash and cycles).
   - Manually budget RAM: stack usage, activation buffers, serial buffer — track this with `avr-size` after every build, don't guess.
2. Build the feature-feed pipeline: since a Nano can't easily sniff real network traffic itself, simulate realistic traffic feature vectors (drawn from your test set) streamed over serial — be upfront about this in your writeup; it's still a legitimate embedded-inference demo, and honesty here is a strength, not a weakness.
3. Implement UART protocol: host sends a feature vector, Nano returns classification + timing info.
4. Instrument and measure: flash bytes used (`avr-size`), RAM peak (watch for stack/heap collision — a classic AVR failure mode), inference latency (toggle a GPIO pin and measure with a logic analyzer or `micros()` if using an Arduino core), and estimated current draw (multimeter in series, or datasheet-based estimate if you don't have one).
5. Stress-test: what's the largest model that still fits? Push the boundary and document where it breaks — this makes your final numbers feel earned rather than arbitrary.

**Deliverables:** `engine-arduino/` PlatformIO project, a table of {flash used, RAM peak, latency, estimated power} vs the theoretical limits, and — ideally — a short demo video showing serial output classifying live-streamed traffic in real time.

**This is your single strongest resume bullet point.** Make sure the numbers are exact and reproducible, not estimated.

---

## Phase 6 — FPGA Accelerator in Vivado (Your Centerpiece)
**Duration:** Weeks 5–7 | **Effort:** ~25% (largest single phase, matches your strength)

**Objectives**
- A real, synthesized, working ternary MAC pipeline — not a simulation-only exercise.

**Tasks**
1. **Architecture design (few days):** define a ternary MAC unit: since weights ∈ {-1,0,1}, each MAC reduces to `if w==1: acc+=x; elif w==-1: acc-=x; else: skip` — implementable with a 2-bit weight encoding and a simple adder/subtractor, no multiplier hardware at all. Sketch the datapath before writing RTL.
2. **RTL implementation:** build the MAC unit, then a small MAC array (parallelism level depends on your board's LUT/DSP budget — pick a size you can justify, e.g., 8 or 16 parallel MACs) with control logic to stream in the same packed weights/activations used by the other two engines.
3. **Verification:** write testbenches in a HDL simulator (Vivado's built-in simulator or Verilator) that feed the *exact same test vectors* used in Phase 4's software validation — bit-exact output matching here is your hardware correctness proof.
4. **Synthesis & implementation:** run through Vivado's full flow, achieve timing closure, and pull the utilization report (LUTs, FFs, DSPs, BRAM) and power estimate (Vivado's power estimator, or measured if your board supports it).
5. **On-board demo:** stream feature vectors into the FPGA (via UART, PCIe, or whatever interface your board supports) and light up or serial-print classifications — a live board demo, not just a synthesis report, is what makes this phase land.
6. **Comparative benchmarking:** latency (cycles to classify one sample, converted to real time via your clock frequency), power estimate, and resource utilization — placed directly next to the Arduino and software numbers.

**Deliverables:** `engine-fpga/` Vivado project with RTL + testbenches, synthesis/implementation reports, utilization and power numbers, and a demo capture (video or logic analyzer trace) of the board classifying real feature vectors.

**Honesty note for your writeup:** be precise about what's synthesized vs simulated, and what your board's real capabilities are (LUT/DSP count, clock speed) — precise, verifiable claims impress technical reviewers far more than vague "ran on FPGA" statements.

---

## Phase 7 — Unified Dashboard & Integration
**Duration:** Week 7 (parallel with tail of Phase 6) | **Effort:** ~10%

**Objectives**
- Tie all three engines into one live, visual, unmistakably impressive demo surface.

**Tasks**
1. Build a FastAPI backend with endpoints/WebSocket channels: one feeding features to each of the three engines (software process, Arduino over serial, FPGA over its interface) and collecting their responses.
2. Build a React frontend: live traffic feed simulation, real-time classification results from all three engines side by side, and live-updating charts for latency, memory footprint, and power draw.
3. Add a "race mode": fire the same sample at all three engines simultaneously and visualize which responds first and at what resource cost — this is your single best 30-second demo clip.
4. Polish UI: clear labeling, a short in-app explainer of what ternary quantization is (for non-technical viewers like recruiters), dark theme, smooth animations on data updates.

**Deliverables:** `dashboard/` full-stack app, deployable locally or to a small cloud instance, screen-recorded demo clip.

---

## Phase 8 — Benchmarking & Three-Way Comparison
**Duration:** Week 8 (start) | **Effort:** ~5%

**Objectives**
- Consolidate every number collected in Phases 3–6 into one authoritative comparison table and set of plots.

**Tasks**
1. Build a master results table: {Software, Arduino, FPGA} × {accuracy, latency, memory/flash/LUTs, power} plus the FP32/INT8/ternary accuracy ablation from Phase 3.
2. Generate clean plots: bar charts for latency and power across the three engines, a memory-footprint comparison, and the accuracy-vs-quantization tradeoff curve.
3. Sanity-check every number against its raw log — this table is what a skeptical reviewer will scrutinize hardest, so it needs to survive "show me where that number came from."

**Deliverables:** `docs/benchmarks/` with the master table (CSV + rendered), all comparison plots as publication-quality images.

---

## Phase 9 — Research Writeup & Publication Polish
**Duration:** Week 8 (end) | **Effort:** ~5%

**Objectives**
- Package everything so it's consumable in 2 minutes by a recruiter and in 20 minutes by an engineer.

**Tasks**
1. Write a paper-style PDF (LaTeX or Markdown→Pandoc): Abstract, Motivation, Quantization Method, Model & Dataset, Three-Engine Architecture (software/embedded/FPGA), Results, Discussion of limitations (be upfront about the simulated traffic feed, board constraints, etc.), Conclusion.
2. Write a top-tier README: one-paragraph pitch, an architecture GIF/diagram, the headline numbers table, links to the demo video and PDF, clear build/run instructions for each engine.
3. Record a 60–90 second demo video: dashboard race-mode clip, a few seconds of the FPGA board and Arduino physically running, voiceover or captions stating the headline numbers.
4. Clean commit history / squash into a readable set of commits per phase; tag a `v1.0` release.
5. Optional strong move: write a short technical blog post version for LinkedIn/personal site summarizing the project narrative — this is often what actually gets seen before anyone opens the repo.

**Deliverables:** `docs/TernaryGuard_Paper.pdf`, polished root `README.md`, demo video file/link, tagged GitHub release.

---

## Risk Register (check in weekly)

| Risk | Mitigation |
|---|---|
| AVR RAM overflow discovered late | Check `avr-size` after every build starting Phase 5, not just at the end |
| FPGA timing closure fails | Start with a conservative clock frequency; optimize only after correctness is proven |
| Dataset class imbalance skews accuracy claims | Report precision/recall/F1, not just accuracy, from Phase 3 onward |
| Three engines' outputs disagree | Always validate each new engine against the Phase 4 software reference before trusting its numbers |
| Scope creep re-adds "just one more domain" | Re-read Section 1 — the thesis is coherence, not breadth |

---

## Resume/Portfolio Packaging Checklist

- [ ] One-line project pitch that mentions the 2KB RAM figure explicitly (it's your best hook)
- [ ] Architecture diagram as a repo README image
- [ ] Demo video linked at the top of the README
- [ ] Headline benchmark table visible without scrolling
- [ ] PDF report linked for anyone who wants depth
- [ ] Clear "Skills demonstrated" section: ML quantization research, embedded C, RTL/Vivado, full-stack deployment, cybersecurity ML
- [ ] Repo is buildable by a stranger following only the README

---

*Total estimated timeline: 8 weeks. Heaviest phases (5 and 6) intentionally align with your strongest skill (Vivado/RTL) and your most dramatic constraint (2KB RAM), which is where the "ultimate" reaction actually comes from.*
