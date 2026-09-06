# TernaryGuard — Vivado XSim Batch Script
# Runs the full-system testbench and exports results for dashboard replay.
#
# Usage: vivado -mode batch -source run_sim.tcl

# ──────────────── Project Setup ────────────────
# Note: Adjust paths if running from a different directory
set proj_dir [file normalize [file dirname [info script]]/..]
set rtl_dir "$proj_dir/rtl"
set tb_dir "$proj_dir/tb"
set sim_dir "$proj_dir/sim_results"

puts "TernaryGuard FPGA Simulation"
puts "  RTL dir: $rtl_dir"
puts "  TB dir:  $tb_dir"
puts "  Output:  $sim_dir"

# ──────────────── Phase 6: Add simulation commands here ────────────────
# xvlog -sv $rtl_dir/ternary_mac.v
# xvlog -sv $rtl_dir/mac_array.v
# xvlog -sv $rtl_dir/control.v
# xvlog -sv $rtl_dir/top.v
# xvlog -sv $tb_dir/tb_top.v
# xelab tb_top -debug typical
# xsim tb_top -runall -log $sim_dir/simulation.log

puts "Simulation script ready — RTL not yet implemented (Phase 6)"
