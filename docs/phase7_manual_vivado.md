# Phase 7 — Manual Vivado completion on the borrowed laptop

Per the 2026-10-03 scope change, Phase 6 implements and checks RTL locally; **Phase 7 is the manual Vivado/XSim/synthesis/implementation/power gate**. Dashboard work remains deferred. This is FPGA simulation/synthesis only, with no physical FPGA deployment. Do not label the design synthesizable or timing-closed as an achieved result until the gates below pass.

## What to transfer

Copy `phase7-vivado-handoff.zip` from the Phase 6 build folder and extract into a short path without spaces, such as `C:\TG6` on Windows or `$HOME/TG6` on Linux. The package contains the RTL, AMD IP generation scripts, constraints, all **69,040** preprocessed frozen test rows, PyTorch reference logits, labels, active-artifact identity files, a hash manifest, and local simulation evidence. It includes the frozen checkpoint for identity verification; raw N-BaIoT CSVs and a Torch installation are not required on the laptop. Do not regenerate or change weights, model, preprocessing, inputs or references.

The external package SHA-256 is printed by `make_handoff.py` and recorded in the local handoff report. Check it before extracting. Then verify every extracted file:

```text
python engine-fpga/verify_handoff.py .
```

Expected: `HANDOFF_PASS files=...`. A mismatch is a stop condition. `handoff_manifest.json` contains the exact hashes; it is not a license to update hashes after modifying files.

Install Python 3 with NumPy for the final comparison:

```text
python -m pip install numpy
```

Use Vivado with **Artix-7 device support installed**, including `xc7a35tcpg236-1`, XSim and Floating-Point Operator IP v7.1. Record the exact Vivado version. An installation or license/IP error is a tool failure to resolve, not a passing experiment. The Tcl scripts were prepared locally but could not be executed on this Mac; preserve and report any version-specific configuration error.

On Windows, open a Command Prompt and call the actual installation's `settings64.bat` (paths differ by release), then change to `C:\TG6`. On Linux, source the installed `settings64.sh`, then change to the extracted directory. Confirm:

```text
vivado -version
```

## Gate 1: real XSim, including AMD floating-point IP

```text
vivado -mode batch -source engine-fpga/scripts/run_sim.tcl -tclargs vectors xsim_out -log xsim_batch.log -journal xsim_batch.jou
python engine-fpga/compare.py --vectors vectors --logits xsim_out/xsim_logits.mem --backend xsim
```

This creates four vendor binary32 operators (add, multiply, divide, sqrt), runs the integer MAC unit testbench, then runs `tb_top` across all 69,040 rows. It does **not** define `TG_PORTABLE_SIM`, compile the VPI/DPI arithmetic models, or use the local host simulator as the floating-point oracle. Inspect `xsim_out/ip_configuration.txt` to confirm the actual generated configurations.

The accelerator's input boundary is **20 host-preprocessed float32 values**. Signed-log1p and StandardScaler run before the FPGA, using the frozen pipeline. RMSNorm, shared-exponent conversion, packed integer dot products, trained scales/biases, ReLU and argmax run in the RTL/IP datapath. This does not claim FPGA log-transform implementation.

Required output:

- `MAC_PASS signed_int64_operations=8192 ... reserved_code=PASS`.
- `RTL_PASS samples=69040 ... backpressure=PASS NaN_rejection=PASS reset=PASS`.
- Comparison JSON: `prediction_matches=69040`, `prediction_disagreements=0`, `rtl_argmax_disagreements=0`, `logits_within_tolerance=true`.
- Every logit must satisfy `abs(RTL-reference) <= 2e-5 + 2e-5*abs(reference)`. Floating-point bit identity is not promised. Integer ternary accumulation must match the C int64 semantics.
- TCP: `total=5555`, `true_positives=1`, `true_positive_test_indices=[59782]`.

**Any TCP deviation is a bug, including apparent improvement.** A passing simulator exit alone is insufficient; run the strict comparison and preserve `vectors/xsim-parity.json`. Missing/unknown logits, changed ROMs/vectors/references, incomplete class-output streams and argmax disagreements cause the comparison to fail.

If XSim errors, copy the batch log and `xsim_out/project/ternaryguard.sim/sim_1/behav/xsim/` logs. Do not report local Verilator results as XSim results. Do not relax tolerances, change the checkpoint or alter the test split to obtain a pass. If a vendor IP arithmetic difference appears, return the mismatch indices and logits here.

## Gate 2: actual synthesis and implementation

After Gate 1 passes:

```text
vivado -mode batch -source engine-fpga/scripts/run_impl.tcl -tclargs vectors impl_out -log implementation_batch.log -journal implementation_batch.jou
```

Target: **xc7a35tcpg236-1**. Constraint: **100 MHz / 10 ns**. The small synthesis wrapper exposes a synchronous LSB-first bit stream with ready/valid; it is **not UART**. Pins are from the Digilent Basys 3 master XDC. The external 2 ns I/O budget is an assumed design constraint, not a measured interface. No board should be programmed.

Collect these real tool-generated outputs:

| File | Required evidence |
|---|---|
| `impl_out/ip_configuration.txt` | Actual FP operator precision, flow control, reset, latency and resource configuration |
| `impl_out/post_synth_utilization.rpt` | Synthesized LUTs, FFs, DSPs, RAM/BRAM, hierarchy |
| `impl_out/routed_utilization.rpt` | Implemented LUTs, FFs, DSPs, RAM/BRAM; include the integer MAC hierarchy separately |
| `impl_out/routed_timing.rpt` | WNS, TNS, WHS, THS, failing endpoints, unconstrained paths |
| `impl_out/route_status.rpt` | Fully routed, with no routing errors |
| `impl_out/drc.rpt` | DRC results; no errors and no waived clock/pin checks |
| `impl_out/check_timing.rpt` | No unexplained unconstrained endpoints, missing clocks or timing exceptions |
| `impl_out/power_vectorless.rpt` | Vivado post-route **vectorless power estimate**, operating assumptions and confidence |
| `impl_out/routed.dcp` | Reopenable implemented design |

The Tcl flow checks negative setup/hold slack, DRC errors, and any DSP primitive inside the integer MAC array, and preserves reports before reporting failure. Hierarchy is retained for auditing. The MAC has `use_dsp="no"`; actual mapped DSP absence still needs confirmation. The **full network may legitimately use DSPs** for FP32 operations; do not call the whole accelerator multiplier-free.

Acceptance requires **WNS >= 0, TNS = 0, WHS >= 0, THS = 0**, fully routed design and no unexplained unconstrained paths or DRC errors at the unchanged 10 ns clock. The printed setup/hold gate is not a substitute for reading the complete report. If it fails, return the reports for RTL optimization. Do not quietly change the target, clock, I/O delays or add blanket false paths.

Power is a Vivado estimate, not measured board power. This initial flow is **vectorless** (no SAIF/VCD annotation); report that limitation and the tool's default activity/temperature/voltage assumptions. Do not invent workload-specific watts. No utilization/timing/power numbers were measured on the Mac.

## Return evidence before closing Phase 7

Send `vivado -version`, both batch logs/journals, XSim simulator logs, `xsim_out/xsim_logits.mem` and its `.predictions` companion, `vectors/xsim-parity.json`, and all `impl_out/*.rpt` plus `ip_configuration.txt`. Keep `routed.dcp` locally (it can be large); provide it if debugging is needed.

Fill this from actual command output:

```text
Vivado version:
Target: xc7a35tcpg236-1
XSim rows / prediction disagreements / max logit error:
TCP: true positives / 5555; sole hit index:
LUTs / FFs / total DSPs / BRAM36 / BRAM18:
Integer MAC array DSP count:
Clock period / WNS / TNS / WHS / THS:
Unconstrained endpoints / route errors / DRC errors:
Post-route vectorless power estimate / confidence / assumptions:
Failures or warnings requiring review:
```

We will review the evidence here before declaring Vivado parity, synthesizability, timing closure or power results complete. This remains **simulation and implementation verification, not a physical FPGA deployment**. Dashboard work starts only after this gate is reviewed and approved.

## Primary references

- [AMD supported operating systems](https://docs.amd.com/r/2025.1-English/ug973-vivado-release-notes-install-license/Supported-Operating-Systems)
- [AMD Floating-Point Operator PG060](https://docs.amd.com/v/u/en-US/pg060-floating-point): vendor denormal/rounding behavior must be validated in XSim, not assumed from host libm.
- [AMD USE_DSP attribute](https://docs.amd.com/r/en-US/ug901-vivado-synthesis/USE_DSP)
- [AMD XSim command reference](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/xsim)
- [Digilent Basys 3 master constraints](https://github.com/Digilent/digilent-xdc/blob/master/Basys-3-Master.xdc)
