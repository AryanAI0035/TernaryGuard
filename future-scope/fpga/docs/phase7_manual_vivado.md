> **Future-scope workbook.** Core TernaryGuard is complete through software and Nano validation. This optional FPGA work does not block that completion. In this repository the code lives in `future-scope/fpga/`; the portable ZIP intentionally retains `engine-fpga/`, so all borrowed-laptop commands below remain valid. Previously transferred ZIPs remain usable.

# Phase 7 — Detailed manual Vivado verification

Updated **2026-10-04**. Follow this on the borrowed laptop. Phase 6 RTL is locally verified; **real Vivado/XSim verification is still pending**. This phase covers simulation, synthesis and implementation for `xc7a35tcpg236-1` (Basys 3). It does not involve a physical FPGA board. Dashboard work stays deferred until the results have been reviewed here.

Use this document as a workbook: complete one numbered step, check its output, then continue. Commands below are commands to run, not transcripts of a successful Vivado run. No Vivado result has yet been measured for this design.

## 1. Understand what you are verifying

There are two separate jobs:

| Job | Question it answers | Tool |
|---|---|---|
| Functional simulation | Does the FPGA design produce the frozen model's predictions and sufficiently close logits? | XSim, with real AMD floating-point IP |
| Synthesis and implementation | Can the design fit the selected FPGA and meet the timing constraints? What resources and estimated power does it use? | Vivado synthesis, placement, routing and reporting |

Passing simulation does not prove timing closure. Passing synthesis does not prove model parity. You must complete both jobs.

The model stays **20→64→32→11**. The simulator receives 20 float32 values that have already undergone the frozen signed-log1p and StandardScaler preprocessing. The FPGA datapath implements RMSNorm, integer ternary dot products, scales, biases, ReLU and class selection. **Feature extraction and preprocessing stay on the host**; there is no claim of FPGA log1p implementation.

The acceptance conditions are fixed:

- All **69,040** frozen rows must be simulated.
- **Zero prediction disagreements** against the frozen PyTorch reference.
- The RTL's own class output must agree with its own logits: **zero argmax disagreements**.
- Each logit must satisfy `abs(RTL-reference) <= 2e-5 + 2e-5*abs(reference)`.
- BASHLITE TCP must remain **1 true positive out of 5,555**, at frozen test index **59782**. Better or worse recall is a discrepancy to investigate, not permission to change the model.
- Target part: **xc7a35tcpg236-1**. Clock constraint: **10 ns / 100 MHz**.
- Routed setup/hold timing must pass, and routing, constraints and DRC must be checked.
- Power must be reported as a **Vivado post-route vectorless estimate**, with assumptions and confidence. It is not measured board power.

Floating-point logits need not be bit-identical to PyTorch. Integer ternary accumulation must preserve the C int64 semantics. Do not increase tolerances, reduce the sample count or change the checkpoint to get a pass.

## 2. Take these files from this Mac

Use the newly supplied **`phase7-vivado-handoff-detailed.zip`**, together with its adjacent **`.zip.sha256`** checksum file. The older `phase7-vivado-handoff.zip` has the shorter manual; it should not be your working copy for these instructions.

The files are in the repository's `future-scope/fpga/build/` folder. Transfer them using a USB drive or another file transfer method. You do not need the entire repository, raw N-BaIoT dataset, Torch installation or Arduino.

The ZIP contains:

```text
engine-fpga/                    RTL, testbenches, scripts and verification tools
vectors/
    inputs.mem                 all 69,040 x 20 preprocessed float32 inputs
    model_rom.sv               ROM derived from the exact active model_weights.h
    reference.npz              frozen PyTorch logits and true class labels
    vectors.json               hashes, sample count and row-identity fingerprint
model/                         active model identity, preprocessing and export manifest
model_weights.h                original frozen exported header
checkpoints/phase3_final_seed42/ternary_2bit.pt
                               frozen checkpoint, retained for identity verification
evidence/                      existing LOCAL simulation evidence
handoff_manifest.json           hashes of the transferred files
PHASE7_MANUAL.md                this manual
```

The checkpoint is included to preserve identity. Laptop comparison uses the already verified PyTorch reference in `reference.npz`; it does not need to load or retrain the checkpoint.

Do not edit the ROM, header, vectors, reference, checkpoint, manifest or RTL. The package has an integrity check. If a script or IP compatibility error needs a fix, preserve the error and return it here for review rather than updating hashes yourself.

## 3. Check the borrowed laptop before starting

You need:

- A Windows or Linux installation supported by the installed Vivado release. Check [AMD's current OS support table](https://www.amd.com/en/support/adaptive-socs-and-fpgas/installer-info-general.html) if the laptop is unfamiliar.
- The **full Vivado Design Suite**, including XSim and synthesis/implementation. Hardware Manager/Lab Edition alone is insufficient for this task. [AMD's explanation](https://adaptivesupport.amd.com/s/question/0D54U00005cV4DvSAK/i-have-installed-latest-vivado-lab-version-20221-for-programming-boolean-board-but-this-version-does-not-bring-up-project-type-rtl-etc-while-creating-the-project-i-am-wondering-whether-there-is-a-bug-in-the-version?language=en_US).
- **Artix-7 device support**, including the exact part `xc7a35tcpg236-1`.
- **Floating-Point Operator IP v7.1**, usable for simulation and synthesis.
- Python 3 and NumPy. A project-local Python environment is created below.
- Enough free disk space for generated Vivado projects and simulator files. The 8 MB transfer ZIP is not the eventual build size. Check available space; no specific run-time or disk-use estimate has been measured for this Vivado flow.

If Vivado is already installed, use that installation and record its version. If installation is necessary, get the full tool from [AMD's downloads](https://www.amd.com/en/support/downloads/adaptive-socs-and-fpgas.html), select Artix-7 support, and follow that release's installation/licensing instructions. We do not need Vitis, board cable drivers or a Basys 3 board definition to run this part-targeted flow. Licensing changes between releases; check what the actual installation requires rather than buying a license based on an assumption.

Plug the laptop into power. Allow long-running commands to finish; avoid sleep during simulation/implementation. A command that has not printed anything recently is not automatically finished. There is no measured XSim duration to promise here.

## 4. Choose your operating-system instructions

**Windows:** follow Sections 5–8, then the common report-reading and evidence sections.

**Linux:** follow Section 9, which gives equivalent setup and run commands, then the common report-reading and evidence sections.

Keep Windows commands in **Command Prompt (`cmd.exe`)**. The `call`, `set` and `%ERRORLEVEL%` instructions below are CMD syntax, not PowerShell syntax. PowerShell is used only where explicitly named, for hashing or packing results.

## 5. Windows: extract and verify the package

### 5.1 Check the ZIP checksum

Open PowerShell. Run the following, replacing the path with the actual transferred ZIP location:

```powershell
Get-FileHash -Algorithm SHA256 "C:\Users\YOUR_NAME\Downloads\phase7-vivado-handoff-detailed.zip"
Get-Content "C:\Users\YOUR_NAME\Downloads\phase7-vivado-handoff-detailed.zip.sha256"
```

The 64-character hexadecimal digest must match the digest in the checksum file; uppercase/lowercase is immaterial. Compare the digest, not the filename. A mismatch means the transfer is damaged or the files came from different package versions: stop and recopy the correct pair.

### 5.2 Extract to a simple path

In File Explorer, extract the ZIP so its contents are directly inside **`C:\TG6`**. Avoid OneDrive, paths with spaces and deeply nested directories.

Check that these exist:

```text
C:\TG6\engine-fpga\scripts\run_sim.tcl
C:\TG6\vectors\inputs.mem
C:\TG6\vectors\reference.npz
C:\TG6\handoff_manifest.json
```

If the actual path is `C:\TG6\phase7-vivado-handoff-detailed\engine-fpga`, you extracted an extra directory level. Move the extracted contents so the four paths above are correct. Do not move individual source files out of their folder layout.

### 5.3 Open Command Prompt and set the working directory

Open Start, type **Command Prompt**, and open it. Run:

```bat
cd /d C:\TG6
dir engine-fpga\scripts\run_sim.tcl
dir vectors\inputs.mem
dir handoff_manifest.json
```

All three `dir` commands must find their file. Use the same CMD window for the remaining Windows steps.

### 5.4 Set up Python locally

Check the Python launcher:

```bat
py -3 --version
```

If `py` is unavailable but `python --version` shows Python 3, use `python` instead of `py -3` for the environment-creation command. If neither works, install Python 3 from [python.org](https://www.python.org/downloads/), reopen CMD and check again.

Create and activate a project-local environment:

```bat
py -3 -m venv .venv
call .venv\Scripts\activate.bat
python --version
python -m pip install numpy
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

The last command must print a NumPy version without an import error. If creating the environment fails, stop and fix that error before continuing. No Torch installation is needed.

### 5.5 Verify the extracted file contents

```bat
python engine-fpga\verify_handoff.py .
```

For this package structure, the expected success marker is:

```text
HANDOFF_PASS files=38
```

It also prints the frozen checkpoint SHA-256:

```text
c335f7d468c932c3855b190061f452ff2b5b0f3fca8276bf1bff6d674ad13774
```

The manifest itself is the 39th ZIP entry. If any file is missing or a hash differs, do not continue. Re-extract the verified ZIP into a fresh folder. Do not edit `handoff_manifest.json` to suppress the error.

## 6. Windows: enable Vivado and check device/IP support

### 6.1 Find the installed environment script

In File Explorer, locate the installed Vivado folder and its **`settings64.bat`**. Installation locations vary. These are examples to help identify the file, not paths guaranteed to exist:

```text
C:\Xilinx\Vivado\2025.2\settings64.bat
C:\AMDDesignTools\2025.2\Vivado\settings64.bat
```

In your existing CMD window, call the path that actually exists:

```bat
call "C:\YOUR_ACTUAL_INSTALLATION\Vivado\settings64.bat"
where vivado
vivado -version > vivado_version.txt 2>&1
type vivado_version.txt
```

Replace `C:\YOUR_ACTUAL_INSTALLATION\Vivado\settings64.bat`; do not copy that placeholder literally. `where vivado` must find the installed executable, and `vivado_version.txt` must contain a Vivado version. Calling the script in a different terminal does not set this terminal's environment.

If `vivado` is still not recognized, confirm the script exists and was called successfully. Keep the full error text. Do not continue with an unrelated `vivado_lab` command.

### 6.2 Check the exact FPGA part and IP

Start an interactive Vivado Tcl session from CMD:

```bat
vivado -mode tcl -log preflight.log -journal preflight.jou
```

When the Vivado Tcl prompt appears, enter these **Tcl commands one at a time**:

```tcl
puts [version -short]
puts [get_parts xc7a35tcpg236-1]
create_project -in_memory tg_preflight -part xc7a35tcpg236-1
puts [get_ipdefs -all xilinx.com:ip:floating_point:7.1]
close_project
exit
```

Check that the part query prints `xc7a35tcpg236-1`, project creation accepts the part, and the IP query prints an entry for `floating_point:7.1`. Empty output or an error is not success. Return `preflight.log` if there is a problem.

If the part is missing, add Artix-7 support using that installation's device-management/installer option. If the IP is missing or license-restricted, resolve the actual installation issue and repeat the check. Do not substitute another FPGA or IP version in the scripts.

After `exit`, you should be back at the normal CMD prompt in `C:\TG6`. The next commands are **CMD commands**, not commands to paste inside a Vivado Tcl console.

## 7. Windows: run full XSim verification

### 7.1 Start the simulation job

With the Python environment and Vivado environment still active:

```bat
vivado -mode batch -source engine-fpga/scripts/run_sim.tcl -tclargs vectors xsim_out -log xsim_batch.log -journal xsim_batch.jou
set "TG7_SIM_EXIT=%ERRORLEVEL%"
echo XSim batch exit code: %TG7_SIM_EXIT%
```

Let this finish before running the comparison. Do not start a second copy in the same output directory.

The script creates the project, generates four AMD FP32 IP operators, runs the MAC unit test, then runs the full-network testbench on all 69,040 rows. It uses the vendor IP branch of `fp32_unit.sv`: `TG_PORTABLE_SIM` is not defined, and the host VPI/DPI arithmetic models are not used.

You should see:

- A `MAC_PASS` marker for the integer unit test.
- Progress messages such as `RTL_PROGRESS samples=1000/69040` as the network simulation advances.
- A final `RTL_PASS samples=69040 ...` marker, including backpressure, NaN rejection and reset checks.
- The normal CMD prompt returning after the script finishes.

The simulator's cycle count can differ from the local host-model count because the vendor IP latencies differ. Do not require that count to match Phase 6. The exact logit-error maximum may also differ within the unchanged tolerance.

A successful batch exit alone is **not acceptance**. AMD documents that `launch_simulation` can return Tcl success despite simulation errors; the final markers, complete files and independent comparison are essential. [Command reference](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/launch_simulation).

### 7.2 Check the produced files

```bat
dir xsim_out\xsim_logits.mem
dir xsim_out\xsim_logits.mem.predictions
dir xsim_out\ip_configuration.txt
findstr /c:"MAC_PASS" /c:"RTL_PASS" /c:"ERROR" /c:"FATAL" xsim_batch.log
```

The files must exist and the pass markers must appear. Error lines need investigation; an error from an earlier stage is not erased by a later pass marker. Preserve the logs.

`xsim_logits.mem` must eventually contain **759,440 words** (69,040 x 11). Its `.predictions` companion must contain **69,040 class IDs**. The comparison checks these lengths automatically; you do not need to count manually.

### 7.3 Run the independent comparison

```bat
python engine-fpga/compare.py --vectors vectors --logits xsim_out/xsim_logits.mem --backend xsim > xsim_compare.txt 2>&1
set "TG7_COMPARE_EXIT=%ERRORLEVEL%"
type xsim_compare.txt
echo Comparison exit code: %TG7_COMPARE_EXIT%
```

Require exit code **0** and all these fields in the printed JSON:

```text
samples: 69040
prediction_matches: 69040
prediction_disagreements: 0
rtl_argmax_disagreements: 0
logits_within_tolerance: true
bashlite_tcp.total: 5555
bashlite_tcp.true_positives: 1
bashlite_tcp.true_positive_test_indices: [59782]
```

These are expected acceptance values, not an already-observed XSim result. Check the actual output. The comparator also verifies generated ROM/input/reference hashes, rejects nonfinite/unknown or missing logits, and checks the RTL's emitted class IDs against its logits.

On completion it writes **`vectors/xsim-parity.json`**. Keep this file and `xsim_compare.txt`, even if the comparison fails. A report JSON can be written before a later assertion fails; its existence is not proof of success. Check the command exit code and all fields.

The `--backend xsim` argument labels the comparison; it does not turn local simulation into XSim. Only outputs from the actual Vivado/XSim command above may be labeled as vendor-IP results.

### 7.4 Check the generated IP configuration

Open `xsim_out\ip_configuration.txt` in a text editor. It must list the four operator instances:

```text
tg_fp_add
tg_fp_mul
tg_fp_div
tg_fp_sqrt
```

Check that they are single-precision operations, use the expected add/multiply/divide/square-root functions, nonblocking flow, and reset support. Keep the file so we can inspect the actual latency/resource configuration. Do not edit it; it is an output report.

### 7.5 Decide whether to continue

Continue to implementation only when the batch run completed, MAC/network checks passed, and the strict comparison passed. If TCP differs, even by one sample, stop and return the output. If the vendor IP causes a numerical mismatch, return the mismatch indices and logs; do not change weights, tolerances or references.

## 8. Windows: run synthesis, placement and routing

### 8.1 Start implementation

```bat
vivado -mode batch -source engine-fpga/scripts/run_impl.tcl -tclargs vectors impl_out -log implementation_batch.log -journal implementation_batch.jou
set "TG7_IMPL_EXIT=%ERRORLEVEL%"
echo Implementation batch exit code: %TG7_IMPL_EXIT%
```

The script runs synthesis, creates a post-synthesis utilization report, then performs implementation through routing. It creates reports before its final timing/resource gates. It does **not** generate or program a bitstream.

Expected final markers are:

```text
TG_ROUTED_WNS_NS=... TG_ROUTED_WHS_NS=...
TG_MAC_ARRAY_DSP_COUNT=0
TG_IMPLEMENTATION_GATES_PASS...
```

Do not fill the ellipses with assumed values. Read the actual report and log. A gate marker is useful, but complete report inspection is still required.

### 8.2 Check the report files

```bat
dir impl_out\*.rpt
dir impl_out\ip_configuration.txt
dir impl_out\routed.dcp
findstr /c:"TG_ROUTED" /c:"TG_MAC" /c:"TG_IMPLEMENTATION" /c:"ERROR" implementation_batch.log
```

The expected files and how to read them are listed in Section 10. A successful synthesis with failed routing or negative slack is not a completed Phase 7.

### 8.3 View the project in the GUI, if useful

The automated flow creates `impl_out\project\ternaryguard.xpr`. Open that project in Vivado's GUI to inspect the implemented design and reports. If reopening it, use **Open Implemented Design** after the run completes. The simulation project is separately at `xsim_out\project\ternaryguard.xpr`.

Use the GUI to inspect outputs, not to create a second design with manually chosen settings. Do not change the part, clock, IP configuration or constraints. Do not click Generate Bitstream or Open Hardware Manager; no physical board is part of this phase.

## 9. Linux: equivalent full workflow

Use a supported Linux installation for the actual Vivado release. These commands assume a Bash terminal. Do not paste Windows `call`/`set` commands here.

### 9.1 Verify and extract the transfer

Put the ZIP and its checksum file in the same directory, then:

```bash
sha256sum phase7-vivado-handoff-detailed.zip
cat phase7-vivado-handoff-detailed.zip.sha256
mkdir -p "$HOME/TG6"
unzip phase7-vivado-handoff-detailed.zip -d "$HOME/TG6"
cd "$HOME/TG6"
ls engine-fpga/scripts/run_sim.tcl vectors/inputs.mem handoff_manifest.json
```

Compare the digest before proceeding. Extract into a fresh directory; do not merge with a previous attempt. Verify that the folder layout matches Section 2.

### 9.2 Set up Python

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
python -c "import numpy; print('NumPy:', numpy.__version__)"
python engine-fpga/verify_handoff.py .
```

Require `HANDOFF_PASS files=38`. If the OS reports that the venv module is unavailable, install the corresponding Python venv support through the laptop's normal administration process, then retry. Do not replace this with an unverified system Python package setup.

### 9.3 Enable Vivado

Find the installed `settings64.sh`. Replace the placeholder below with its real path:

```bash
source /YOUR/ACTUAL/Vivado/INSTALLATION/settings64.sh
command -v vivado
vivado -version > vivado_version.txt 2>&1
cat vivado_version.txt
```

Run the same preflight Tcl session as Section 6.2:

```bash
vivado -mode tcl -log preflight.log -journal preflight.jou
```

At the Vivado prompt enter:

```tcl
puts [version -short]
puts [get_parts xc7a35tcpg236-1]
create_project -in_memory tg_preflight -part xc7a35tcpg236-1
puts [get_ipdefs -all xilinx.com:ip:floating_point:7.1]
close_project
exit
```

Check the part and IP output. You are back in Bash after `exit`.

### 9.4 Run XSim and compare

```bash
vivado -mode batch -source engine-fpga/scripts/run_sim.tcl -tclargs vectors xsim_out -log xsim_batch.log -journal xsim_batch.jou
tg7_sim_exit=$?
printf 'XSim batch exit code: %s\n' "$tg7_sim_exit"
```

Then, only after simulation has finished:

```bash
python engine-fpga/compare.py --vectors vectors --logits xsim_out/xsim_logits.mem --backend xsim > xsim_compare.txt 2>&1
tg7_compare_exit=$?
cat xsim_compare.txt
printf 'Comparison exit code: %s\n' "$tg7_compare_exit"
```

Require exit 0 and the Section 7 acceptance values. Inspect `xsim_out/ip_configuration.txt`. Do not continue after a failed comparison.

### 9.5 Run implementation

```bash
vivado -mode batch -source engine-fpga/scripts/run_impl.tcl -tclargs vectors impl_out -log implementation_batch.log -journal implementation_batch.jou
tg7_impl_exit=$?
printf 'Implementation batch exit code: %s\n' "$tg7_impl_exit"
ls impl_out/*.rpt impl_out/ip_configuration.txt impl_out/routed.dcp
```

Read the reports below; do not rely only on an exit code.

## 10. Read the synthesis and implementation reports

Open reports in a text editor or Vivado's report viewer. Record numbers from the actual files, including units and device/clock context.

### 10.1 Utilization: how much of the FPGA is used

Read both:

- `impl_out/post_synth_utilization.rpt`: synthesized resource use.
- `impl_out/routed_utilization.rpt`: final implemented resource use and hierarchy.

Record used LUTs, registers/FFs, DSPs, RAMB36 and RAMB18, and distributed memory if reported. Vivado can show a BRAM summary in equivalent tiles rather than separate primitive counts; retain the report and its original units rather than silently converting. Report the actual used and available values.

The integer MAC hierarchy is `accelerator/core/array`. It must have **zero DSP primitives**, consistent with add/subtract/skip arithmetic. The full accelerator may use DSPs in its floating-point multiplier or other FP units. Therefore **total DSP count need not be zero**. Preserve the hierarchy table and `TG_MAC_ARRAY_DSP_COUNT=0` log marker; both are useful evidence.

### 10.2 Timing: whether the requested clock closes

Read `impl_out/routed_timing.rpt`, especially the final design timing summary:

| Field | Meaning | Required result |
|---|---|---|
| WNS | Worst setup slack | >= 0 ns |
| TNS | Sum of negative setup slack | 0 ns |
| WHS | Worst hold slack | >= 0 ns |
| THS | Sum of negative hold slack | 0 ns |
| Failing endpoints | Paths failing the checks | 0 |

Also check pulse-width/other timing checks if the report lists them. Any failure requires review. Check that the reported clock remains **10 ns**. A timing report from a different part or clock is not this experiment.

Negative slack, even a small negative value, means timing has not closed. Missing/unconstrained paths also prevent an unconditional closure claim. Read `impl_out/check_timing.rpt` for missing clocks, unconstrained endpoints and constraint issues. Return warnings that you do not understand.

The 2 ns input/output delay budget is an explicit hypothetical synchronous-interface constraint, not measured interface timing. Do not change it, disable the dedicated clock route requirement, add blanket false paths or lower the requested clock to turn a failed run into a pass.

### 10.3 Routing and design checks

Read:

- `impl_out/route_status.rpt`: the design must be completely routed with no routing errors.
- `impl_out/drc.rpt`: no DRC errors. Preserve critical warnings and other warnings for review.
- `impl_out/check_timing.rpt`: no unexplained constraint/timing coverage issues.

A zero-DSP marker and positive WNS are insufficient if routing or DRC failed. Do not waive checks to proceed.

### 10.4 Power: an estimate from Vivado, with assumptions

Read `impl_out/power_vectorless.rpt`. Record the total on-chip power and its reported breakdown, confidence level, temperature/voltage assumptions and switching activity assumptions.

This flow supplies no SAIF/VCD workload annotation to Report Power, so the result is a **post-route vectorless estimate**. Do not describe it as measured power, inference energy or workload-accurate watts. A low-confidence estimate should be recorded with that confidence. Do not alter activity rates until the method is agreed.

Timing and power interpretation follow [AMD's timing guidance](https://docs.amd.com/r/2021.1-English/ug1388-acap-system-integration-validation-methodology/Understanding-Timing-Reports) and [vectorless power analysis guidance](https://docs.amd.com/r/2024.1-English/ug907-vivado-power-analysis-optimization/Vectorless-Power-Analysis).

## 11. Troubleshooting: stop at the first unexplained failure

| Symptom | What to check | What to send back |
|---|---|---|
| `vivado` not recognized/found | Call/source the actual environment script in the same terminal; confirm full Vivado installation | Exact command and terminal error |
| Invalid or missing part | Artix-7 support; exact spelling `xc7a35tcpg236-1` | `preflight.log`, Vivado version |
| Floating-point IP missing/licensing error | Full tool and IP catalog/license for the installed release | Preflight/batch log and exact IP message |
| Invalid `CONFIG.*` property | Release-specific IP interface/configuration mismatch | Vivado version, property name, complete error; do not silently delete settings |
| Missing `vectors.json`, ROM or input file | Working directory and ZIP extraction layout | `dir`/`ls` output, package verification output |
| `HANDOFF HASH MISMATCH` | Correct ZIP/checksum pair; fresh extraction; no changed files | Named mismatch and ZIP digest |
| Simulation still running | Check `RTL_PROGRESS` messages and active process; do not launch another job in the same directory | Last log lines if it appears stuck |
| Simulation `FATAL`, timeout or incomplete logits | First simulator error, vendor FP latency/handshake, input contract | Batch log plus XSim simulator logs; partial outputs |
| Class disagreements or wrong TCP recall | Treat as a parity bug; inspect mismatch indices | `xsim_compare.txt`, parity JSON and both RTL output streams |
| All classes match but logits exceed tolerance | Numerical parity still fails; do not loosen tolerance | Comparison failure, logits, IP configuration |
| Synthesis/implementation failed | First synthesis/implementation error; target capacity and constraints | Batch log, run logs and any generated reports |
| Negative slack | Inspect worst routed paths; RTL optimization may be needed | Routed timing, utilization and checkpoint if requested |
| Missing reports or empty timing paths | Job may have failed before reporting; no timing claim allowed | Batch log and files that actually exist |
| Re-running an attempt | Preserve the first attempt; use a fresh output folder or rename old folders first | Separate evidence from each attempt |

For simulation errors, logs are normally under:

```text
xsim_out/project/ternaryguard.sim/sim_1/behav/xsim/
```

Vivado may vary file names by release; preserve available elaboration, compile and simulation logs in that directory.

For synthesis/implementation errors, look under:

```text
impl_out/project/ternaryguard.runs/synth_1/
impl_out/project/ternaryguard.runs/impl_1/
```

Keep their `runme.log` files and IP run logs if the failure happened while building an IP.

The scripts use `create_project -force`, and output files can be overwritten. Before retrying, preserve the failed attempt's folders and logs. Do not mix a passing comparison from one attempt with synthesis reports from another modified design. Return partial evidence even if the phase cannot finish; do not fill missing numbers with estimates.

## 12. Fill out this results sheet

Create `phase7_results.txt` in the extracted package folder. Copy this template and fill it only from actual outputs:

```text
Run date/time:
Laptop OS:
Vivado version:
Package ZIP SHA-256:
Handoff integrity check result:
Target part: xc7a35tcpg236-1
Clock constraint: 10 ns / 100 MHz

XSim batch exit code:
MAC_PASS marker present:
RTL_PASS samples=69040 marker present:
Strict comparison exit code:
Simulated rows:
Prediction disagreements vs frozen checkpoint:
RTL class output vs RTL-logit argmax disagreements:
Maximum absolute logit error:
All logits pass atol=2e-5, rtol=2e-5:
TCP true positives / total:
TCP true-positive test indices:

Implementation batch exit code:
Used LUTs / available:
Used registers/FFs / available:
Total DSPs / available:
Integer MAC array DSPs:
RAMB36 / RAMB18 / other reported BRAM units:
Distributed memory:
WNS / TNS / WHS / THS, in ns:
Failing endpoints and other timing failures:
Unconstrained/missing-clock issues:
Route status/errors:
DRC errors / critical warnings / other warnings:

Post-route vectorless power estimate, in W:
Power breakdown:
Power confidence level:
Temperature / voltage / activity assumptions:

Any failed gate, warning or unresolved question:
```

Write **not available** if a stage failed before producing a value. Do not write zero or PASS for a missing result.

## 13. Package the evidence to bring back

Bring back these files, preserving their relative paths:

```text
phase7_results.txt
vivado_version.txt
preflight.log
preflight.jou
xsim_batch.log
xsim_batch.jou
xsim_compare.txt
vectors/xsim-parity.json
xsim_out/xsim_logits.mem
xsim_out/xsim_logits.mem.predictions
xsim_out/ip_configuration.txt
impl_out/ip_configuration.txt
impl_out/post_synth_utilization.rpt
impl_out/routed_utilization.rpt
impl_out/routed_timing.rpt
impl_out/route_status.rpt
impl_out/drc.rpt
impl_out/check_timing.rpt
impl_out/power_vectorless.rpt
```

Include available XSim logs and synthesis/implementation `runme.log` files, particularly for a failure. Keep `impl_out/routed.dcp` locally; it can be supplied if debugging is needed. The raw output streams compress well; preserve them rather than sending only screenshots or typed summaries.

### Windows: create a return ZIP in PowerShell

Open PowerShell, change to `C:\TG6`, and paste this block. It copies only files that exist, so you can also return a failed/partial attempt. Check the final ZIP contents before sending it.

```powershell
Set-Location C:\TG6
$tg7Return = Join-Path (Get-Location) ("phase7_return_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $tg7Return | Out-Null
$tg7Files = @(
  "phase7_results.txt", "vivado_version.txt", "preflight.log", "preflight.jou",
  "xsim_batch.log", "xsim_batch.jou", "xsim_compare.txt",
  "vectors\xsim-parity.json", "xsim_out\xsim_logits.mem",
  "xsim_out\xsim_logits.mem.predictions", "xsim_out\ip_configuration.txt",
  "impl_out\ip_configuration.txt"
)
if (Test-Path "impl_out") {
  $tg7Files += Get-ChildItem "impl_out" -Filter *.rpt -File | ForEach-Object { "impl_out\" + $_.Name }
}
foreach ($tg7Relative in $tg7Files) {
  if (Test-Path $tg7Relative) {
    $tg7Destination = Join-Path $tg7Return $tg7Relative
    New-Item -ItemType Directory -Force -Path (Split-Path $tg7Destination) | Out-Null
    Copy-Item $tg7Relative $tg7Destination
  }
}
Compress-Archive -Path "$tg7Return\*" -DestinationPath "$tg7Return.zip"
Write-Output "Return package: $tg7Return.zip"
```

Add failure-specific simulator/run logs to that ZIP manually if needed. The block does not include large generated project directories or the checkpoint.

### Linux: create a return archive

From the extracted package directory:

```bash
tg7_return="phase7_return_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$tg7_return"
for tg7_file in phase7_results.txt vivado_version.txt preflight.log preflight.jou xsim_batch.log xsim_batch.jou xsim_compare.txt vectors/xsim-parity.json xsim_out/xsim_logits.mem xsim_out/xsim_logits.mem.predictions xsim_out/ip_configuration.txt impl_out/ip_configuration.txt impl_out/*.rpt; do
  if [ -f "$tg7_file" ]; then
    mkdir -p "$tg7_return/$(dirname "$tg7_file")"
    cp "$tg7_file" "$tg7_return/$tg7_file"
  fi
done
tar -czf "$tg7_return.tar.gz" "$tg7_return"
printf 'Return package: %s.tar.gz\n' "$tg7_return"
```

Add failure-specific logs when applicable. Sending an incomplete archive is fine if its results sheet says what failed; treating missing evidence as success is not.

## 14. Final checklist before returning the laptop

- [ ] ZIP checksum and extracted-file integrity passed.
- [ ] Vivado version, exact target and generated IP configurations are recorded.
- [ ] Actual XSim finished all 69,040 rows using AMD IP.
- [ ] Independent comparison passed with zero prediction and RTL argmax disagreements.
- [ ] TCP remains 1/5,555 at index 59782.
- [ ] Synthesis, placement and routing completed for the exact part.
- [ ] WNS/WHS are nonnegative; TNS/THS and failing endpoint counts are zero at 10 ns.
- [ ] Routing, DRC and timing coverage were checked; unresolved warnings are recorded.
- [ ] Utilization and vectorless power reports are retained with original units and assumptions.
- [ ] Both RTL output streams, reports and logs are saved in the return archive.
- [ ] Every missing/failed gate is stated plainly in `phase7_results.txt`.
- [ ] No FPGA was programmed and no physical power/latency claim was made.

An unchecked item means the gate is incomplete or needs review. Return the evidence here before calling Phase 7 closed. Only then will we decide whether the design is synthesizable, timing-closed and vendor-IP parity-verified for the agreed simulation-only scope.

## References

- [AMD installation and OS support](https://www.amd.com/en/support/adaptive-socs-and-fpgas/installer-info-general.html)
- [AMD supported devices, 2025.2](https://docs.amd.com/r/2025.2-English/ug973-vivado-release-notes-install-license/Supported-Devices)
- [AMD Floating-Point Operator PG060](https://docs.amd.com/v/u/en-US/pg060-floating-point)
- [AMD launch_simulation reference](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/launch_simulation)
- [AMD timing-report interpretation](https://docs.amd.com/r/2021.1-English/ug1388-acap-system-integration-validation-methodology/Understanding-Timing-Reports)
- [AMD vectorless power analysis](https://docs.amd.com/r/2024.1-English/ug907-vivado-power-analysis-optimization/Vectorless-Power-Analysis)
- [AMD USE_DSP synthesis attribute](https://docs.amd.com/r/en-US/ug901-vivado-synthesis/USE_DSP)
- [Digilent Basys 3 master XDC](https://github.com/Digilent/digilent-xdc/blob/master/Basys-3-Master.xdc)
