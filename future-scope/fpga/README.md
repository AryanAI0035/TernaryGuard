> **Optional future extension.** The software and Arduino research prototype is complete independently. FPGA is excluded from core acceptance and résumé deployment claims. Historical Phase 6/7 names below describe this extension only.

# TernaryGuard — FPGA accelerator

**Phase 6: locally verified RTL. Phase 7: manual Vivado acceptance pending.**
Target `xc7a35tcpg236-1` (Basys 3), simulation/synthesis only; no physical board deployment. Dashboard work remains deferred under the 2026-10-03 scope change. See [Phase 7 manual instructions](docs/phase7_manual_vivado.md) and [Phase 6 evidence](docs/phase6_local_validation.md).

## Datapath

`ternary_mac.sv` implements signed int64 add/subtract/skip with `00=0`, `01=+1`, `10=-1`; reserved `11` faults. `mac_array.sv` instantiates eight lanes. `control.sv` sequences the exact 20→64→32→11 architecture, including two RMSNorms, shared-exponent conversion, trained scale/bias, ReLU and first-index argmax. `numeric.sv` implements bit-level conversion/truncation and round-to-nearest-even restoration. The integer kernel contains no weight multiply, floating-point operations or dynamic allocation.

The input stream contains **20 preprocessed binary32 values** from the frozen signed-log1p/StandardScaler pipeline. Preprocessing stays on the host; this accelerator does not implement feature extraction or log1p. Outputs are 11 binary32 logits plus the RTL's class ID. Finite normals and signed zeros are supported; subnormal/nonfinite inputs or intermediates are rejected. The full frozen dataset passes this contract locally; arbitrary inputs outside it have no parity claim.

`fp32_unit.sv` wraps four AMD FP32 IP cores for separate add/multiply/divide/sqrt operations outside the dot product. `TG_PORTABLE_SIM` selects **simulation-only host arithmetic**, via VPI for Icarus or DPI for Verilator. That mode is never used in synthesis or the XSim acceptance scripts. Local parity is not proof of vendor-IP arithmetic parity or timing closure.

`top.sv` provides a ready/valid word-stream interface. `basys3_top.sv` is the small pin-count synthesis wrapper, using a synchronous LSB-first bit stream; it is not UART and has not run on a physical FPGA. Reset is synchronous and must be asserted for at least five clock edges in the validation setup. A fault latches until reset.

## Local reproduction

From the repository root, with Python's existing model dependencies and Verilator installed:

```sh
python3 future-scope/fpga/prepare.py --vectors test --output future-scope/fpga/build/frozen
python3 future-scope/fpga/run_compiled.py --vectors future-scope/fpga/build/frozen
python3 -m pytest future-scope/fpga/tests/ -v --tb=short
python3 future-scope/fpga/make_handoff.py --vectors future-scope/fpga/build/frozen --output future-scope/fpga/build/phase7-vivado-handoff.zip
```

The generator checks the existing active artifact hashes, runs `verify_active`, loads frozen row identities and computes a fresh PyTorch oracle. It derives ROM bytes and binary32 constants from the exact root `model_weights.h`; it does not regenerate or modify that header. The comparator checks all generated hashes, exact prediction/class-output agreement, per-logit tolerance (`atol=rtol=2e-5`) and TCP **1/5,555 at index 59782**.

`run_local.py` offers slower Icarus/VPI network simulation. MAC, array, numeric, full golden-vector and synthesis-wrapper regressions are in `future-scope/fpga/tests/test_fpga_port.py`. Tool-dependent tests report explicit skips if their required simulator is absent; no skip is accepted as proof of FPGA correctness.

## Manual Vivado gate

Use the portable ZIP and [instructions](docs/phase7_manual_vivado.md) on the borrowed laptop. Scripts generate XSim, post-synthesis/post-route utilization, timing, DRC, routing and **vectorless power** reports. The target clock is 100 MHz; this is a constraint, not an achieved frequency until routing closes.

No LUT/FF/DSP/BRAM count, WNS/TNS or wattage is claimed without an actual Vivado report. RTL is written for synthesis; vendor synthesis and timing closure remain unverified. The whole network includes FP multipliers, even though its ternary MAC array uses add/subtract/skip only.
