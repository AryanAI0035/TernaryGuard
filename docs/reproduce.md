# Reproduce the completed research prototype

## 1. Local setup and bundled replay

Use Python 3.11+ and a C11 compiler available as `cc`. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/demo.py
python3 -m pytest model/tests/ -v --tb=short
python3 engine-software/build.py --output /tmp/tg-c-build
```

The core has 99 tests. It exercises synthetic leakage/device-coverage regressions, precision/export guards, packed integer dot products, all 264 bundled engineering vectors, and the host build of the Nano source and serial protocol. These checks need neither a board nor the raw dataset. They do not replace the full-data or physical validation below.

The demo verifies checkpoint/header/preprocessing identity and compares checkpoint, header reader and C logits/predictions on preprocessed engineering vectors. It does not exercise raw feature preprocessing or compute dataset accuracy. The frozen reference files are included; training is unnecessary.

## 2. Full frozen-device test evaluation

Download the external N-BaIoT data (large archive; requires network/disk space):

```sh
python3 research/download_nbaiot.py --source uci
python3 model/validate_export.py --verify-active
python3 engine-software/validate.py --output /tmp/tg-c-validation
/tmp/tg-c-validation/build/ternary_infer --benchmark /tmp/tg-c-validation/benchmark_raw_f64.bin 100000
```

Default CSV location is `data/raw/nbaiot/`. For a different location, pass `--data-dir PATH` to both validation commands. Each source must match its recorded SHA-256 and schema; validation also guards the exact frozen test row identities. If the upstream dataset changes, stop and resolve identity rather than recreating the split or updating hashes.

Expected identity: `model/active_model.json` → `checkpoints/phase3_final_seed42/ternary_2bit.pt`, checkpoint SHA-256 `c335f7d468c932c3855b190061f452ff2b5b0f3fca8276bf1bff6d674ad13774`. The active verifier reads the exact existing root `model_weights.h`; it does not overwrite it.

Acceptance is 69,040 samples with zero prediction disagreements. Logits use `atol=rtol=2e-5`; BASHLITE TCP must be exactly 1/5,555. Full-data verification may take minutes and several GB of RAM; the bundled replay is the quick entry point.

## 3. Nano build and physical repetition

Install PlatformIO separately (`python3 -m pip install platformio`) and connect an Arduino Nano. With the raw dataset available:

```sh
python3 engine-arduino/prepare_validation.py --output /tmp/tg-nano-validation
pio run -d engine-arduino -e nano_new
pio run -d engine-arduino -e nano_new -t upload --upload-port PORT
python3 engine-arduino/serial_validate.py --port PORT --cases /tmp/tg-nano-validation/serial_cases.npz --group subset --output /tmp/tg-nano-validation/hardware-subset.jsonl
python3 engine-arduino/serial_validate.py --port PORT --cases /tmp/tg-nano-validation/serial_cases.npz --group tcp --output /tmp/tg-nano-validation/hardware-tcp.jsonl
```

Replace PORT with the actual USB serial device. Use `nano_old` only for the corresponding installed bootloader. The subset is 330 rows (30/class), including TCP's sole true positive; the separate TCP pass is all 5,555 rows. Preserve serial logs and check predictions, CRC, status, timing and SRAM gap. See [firmware instructions](../engine-arduino/README.md) and [physical report](phase5_hardware_validation.md) for measurement interpretation.

## 4. Optional FPGA continuation

FPGA is independent future scope. Follow [the detailed Vivado workbook](../future-scope/fpga/docs/phase7_manual_vivado.md). The repository stores the extension under `future-scope/fpga/`; portable handoff ZIPs intentionally contain `engine-fpga/`, matching all laptop commands in the workbook. Previously transferred ZIPs continue to work.

Local optional tests run explicitly with `python3 -m pytest future-scope/fpga/tests/ -v --tb=short` (8 tests; requires Verilator/Icarus). Core CI does not require FPGA tooling. No local simulation result replaces actual Vivado/XSim, routed timing or physical FPGA evidence.
