## TernaryGuard — FPGA Engine (Vivado)

### Target
- **Part:** `xc7a35tcpg236-1` (Artix-7, Basys 3)
- **Mode:** Simulation-only (no physical board)
- **Simulator:** Vivado XSim

### Directory Structure
```
engine-fpga/
├── rtl/                    # Synthesizable Verilog/SystemVerilog
│   ├── ternary_mac.v       # Single ternary MAC unit
│   ├── mac_array.v         # Parallel MAC array
│   ├── control.v           # Sequencer / control FSM
│   └── top.v               # Top-level integration
├── tb/                     # Testbenches (non-synthesizable)
│   ├── tb_ternary_mac.v    # MAC unit testbench
│   ├── tb_mac_array.v      # MAC array testbench
│   └── tb_top.v            # Full-system testbench
├── constraints/            # Vivado constraints
│   └── basys3.xdc          # Pin/timing constraints for xc7a35t
├── sim_results/            # XSim output logs for dashboard replay
│   └── .gitkeep
├── scripts/                # TCL automation scripts
│   └── run_sim.tcl         # Batch simulation script
└── README.md               # This file
```

### Validation Strategy
- Testbenches use the **exact same test vectors** as `engine-software/` (Phase 4)
- Outputs compared bit-exact against software reference
- Simulation results exported as JSON to `sim_results/` for dashboard replay

### Synthesis Reports (generated)
- Utilization report (LUTs, FFs, DSPs, BRAMs)
- Timing summary (WNS, TNS, WHS)
- Power estimate (Vivado power analyzer)
