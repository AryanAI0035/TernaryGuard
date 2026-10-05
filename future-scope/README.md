# Optional future work

The completed TernaryGuard research prototype comprises the frozen model, workstation C engine, and physically validated Arduino Nano port. These extensions are independent of core completion.

| Extension | Existing work | Remaining acceptance |
|---|---|---|
| [FPGA accelerator](fpga/README.md) | RTL, local portable simulation, frozen vectors, testbenches | Actual AMD XSim, Vivado synthesis/implementation, resource/timing/power reports |
| [Dashboard](dashboard/README.md) | Directory placeholders only | Backend, frontend and actual integration |

FPGA is **not** a physical deployment or a timing-closed implementation. No FPGA resource, timing or power result is claimed. The [detailed borrowed-laptop workbook](fpga/docs/phase7_manual_vivado.md) remains available; its portable ZIP keeps the original `engine-fpga/` layout.

Core tests: `python3 -m pytest model/tests/ -v --tb=short` (99 tests).
Optional FPGA tests: `python3 -m pytest future-scope/fpga/tests/ -v --tb=short` (8 tests, requires local simulators).
Both together: `python3 -m pytest model/tests/ future-scope/fpga/tests/ -v --tb=short` (107 tests).

Do not change the frozen model, tolerances or known TCP limitation to obtain extension parity. Future acceptance should preserve the exact model identity and report the tool/hardware coverage actually exercised.
