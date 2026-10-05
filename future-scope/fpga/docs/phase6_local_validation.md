> **Historical local evidence for an optional future extension.** Commands and source-hash records describe the original layout at the recorded commits. Current source is in `future-scope/fpga/`; evidence is in `../evidence/`. Core completion does not depend on Vivado.

# Phase 6 — Local RTL validation; Phase 7 manual Vivado gate pending

Date: 2026-10-03. Starting repository HEAD: 759740a (includes the approved Phase 5 hardware work at c2d088f and reference-label clarification). Per the user's scope change, Phase 6 is the available local RTL work; Phase 7 is manual Vivado acceptance on a borrowed laptop. Dashboard work is deferred. **No physical FPGA deployment is claimed.**

## Outcome and limits

Implemented an eight-lane signed int64 ternary MAC array and control FSM for the frozen **20→64→32→11** model. The integer MAC uses add/subtract/skip only, with LSB-first `00=0`, `01=+1`, `10=-1`; reserved `11` faults. The C engine's shared exponent, 40-bit magnitude budget, truncation and binary32 RNE restoration are preserved. All constants originate from the exact active root `model_weights.h`, guarded by the existing artifact hashes.

The accelerator accepts the existing **20 host-preprocessed float32 inputs**. Signed-log1p/StandardScaler remain on the host; RMSNorm, gamma, dot products, scales, biases, ReLU and argmax are implemented in the accelerator datapath. The selected input boundary is explicit: no FPGA preprocessing or raw 115-feature inference claim is made.

Float operations are separate from the integer MAC. For synthesis and future XSim, wrappers instantiate AMD Floating-Point Operator add/mul/div/sqrt IP. The local test mode uses **host binary32 arithmetic** via Verilator DPI or Icarus VPI. It is simulation-only and excluded from vendor synthesis. Vendor rounding/denormal behavior remains to be checked in real XSim. Supported numeric inputs/intermediates are finite binary32 normals and signed zero; unsupported subnormal/nonfinite values fault. This full frozen dataset passed locally under that contract.

**RTL is written for synthesis, but actual synthesizability, FPGA fit, timing closure and power remain unverified.** No Vivado/XSim executable is available on this Mac. None of the local simulator results is presented as a vendor tool report. No estimated LUT/FF/DSP/BRAM, WNS/TNS or wattage is substituted for actual reports.

## Real command output

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 engine-fpga/prepare.py --vectors test --output /private/tmp/tg-phase6-final
PYTHONDONTWRITEBYTECODE=1 python3 engine-fpga/run_compiled.py --vectors /private/tmp/tg-phase6-final
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/tg-phase6-checks/mpl python3 -m pytest model/tests/ -v --tb=short -p no:cacheprovider
```

Full raw outputs (including compiler commands) are retained under [docs/benchmarks/phase6_local](../evidence/):

- [Active hashes and frozen vector preparation](../evidence/active-and-vector-preparation.txt)
- [Final compiled RTL run](../evidence/compiled-rtl-run.txt)
- [MAC, array and fixed arithmetic unit output](../evidence/integer-unit-checks.txt)
- [Full pytest output](../evidence/pytest.txt)
- [Actual parity JSON](../evidence/parity.json)
- [Tool availability](../evidence/tools.json)
- [RTL/testbench source hashes](../evidence/source-hashes.json)

Observed output:

```text
MAC_PASS signed_int64_operations=8192 skip=PASS enable=PASS clear=PASS reserved_code=PASS
ARRAY_PASS lanes=8 batches=16 terms_per_batch=64 reserved_lane7=PASS
NUMERIC_PASS cases=2048
RTL_PASS samples=69040 post_input_wait_cycles=387866720 backpressure=PASS NaN_rejection=PASS reset=PASS
107 passed in 62.15s (0:01:02)
```

The local arithmetic model's artificial operation latencies determine the reported cycle count. It is **not an achieved FPGA latency**, nor a clock-closure measurement.

The suite increased **99→107 tests**. New tests exercise signed MAC semantics and reserved-code faults, lane independence, fixed conversion and RNE ties, full golden-vector FSM/stream parity, all 920 packed weight bytes, ROM tamper rejection, wrong hardware argmax rejection and the synthesis wrapper's bit serialization/backpressure. The 264-vector wrapper check passes with zero prediction disagreements. The first regression run found a 128-byte filename truncation bug in testbenches (105 passed / 2 failed); SystemVerilog strings fixed it, and the complete suite was rerun. No skips remained in the final run. Tcl structural completeness was also checked locally; that is not Vivado execution.

## Full frozen-test parity

| Check | Observed local result |
|---|---:|
| Frozen test samples | 69,040 |
| Prediction matches | 69,040 |
| Prediction disagreements | 0 |
| RTL-reported class vs RTL-logit argmax disagreements | 0 |
| Maximum absolute logit error vs frozen PyTorch | 1.1444091796875e-05 |
| Logit tolerance | atol=2e-5, rtol=2e-5; all pass |
| Binary32 logit words bit-identical to PyTorch | 225,829 / 759,440 |
| Test accuracy | 0.8108198146002318 (81.08%) |
| BASHLITE TCP rows | 5,555 |
| BASHLITE TCP true positives | 1 |
| Sole TCP hit test index | 59782 |

The complete RTL logit stream and direct class outputs are saved compressed in the evidence directory. Floating logits are tolerance-equivalent, not generally bit-identical. **TCP 1/5,555 (0.018%) is preserved as the known model limitation, not improved or hidden.** The XSim gate must preserve the same behavior.

Active checkpoint SHA-256: `c335f7d468c932c3855b190061f452ff2b5b0f3fca8276bf1bff6d674ad13774`.
Header SHA-256: `e1d919a0049e2c492d4ce54168cf9cb223bd90b4b1b74554539089ccf4c60ee8`.
Canonical preprocessing JSON SHA-256: `1339886eaa7cde612b32831057afb225b7d5f314f8ebecdc6f8d57ae99c55c7d`.
Frozen test row identity SHA-256: `b3d5c16fceb383423cdab2982d5e119280bdbed8d72246235f798e56d51be1f1`.

## Manual Phase 7

[The manual guide](phase7_manual_vivado.md) supplies exact commands and failure gates. The portable handoff package includes code, all vectors, expected PyTorch logits/labels, active identities and an integrity manifest, without needing the raw dataset on the laptop.

Pending outputs are actual XSim parity using AMD FP IP, full synthesis/place/route for `xc7a35tcpg236-1`, utilization, setup/hold timing at 100 MHz, DRC/route/constraint checks and post-route vectorless power estimate. The flow requests `use_dsp="no"` for the integer MAC and checks the mapped hierarchy, but mapped DSP absence still requires real synthesis evidence. The whole network includes FP multipliers, so total DSP use need not be zero.

Power must be labeled **Vivado vectorless estimate**, with its assumptions/confidence. No physical power or FPGA board execution is claimed. Do not start dashboard work or call the design timing-closed until the returned manual results pass review.

Portable package: `engine-fpga/build/phase7-vivado-handoff.zip` (8,415,029 bytes, 39 entries). External SHA-256: `8c4ca29e72247e666152244887d135f84231f7f2b7ab915ab2d3fac993cd3d8d`. An independent extraction passed `HANDOFF_PASS files=38`; the manifest itself is the 39th entry. The transferred oracle was checked again against the actual local RTL output. Raw package metadata is in `docs/benchmarks/phase6_local/handoff-package.json`.
