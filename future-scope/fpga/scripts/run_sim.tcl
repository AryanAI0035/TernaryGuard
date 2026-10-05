# XSim validation with REAL AMD FP IP; no TG_PORTABLE_SIM, no VPI host oracle.
source [file join [file dirname [info script]] create_project.tcl]
add_files -fileset sim_1 [file join $engine tb tb_ternary_mac.sv]
add_files -fileset sim_1 [file join $engine tb tb_top.sv]
set_property top tb_ternary_mac [get_filesets sim_1]
launch_simulation
run all
close_sim
set handle [open [file join $vectors vectors.json] r]
set meta [read $handle];close $handle
if {![regexp {"samples":\s*([0-9]+)} $meta match samples]} {error "No sample count"}
if {$samples!=69040} {error "Final XSim acceptance requires ALL 69,040 frozen rows"}
set_property top tb_top [get_filesets sim_1]
# Quote paths inside plusargs; use a short directory without spaces on Windows.
set opts "-testplusarg SAMPLES=$samples -testplusarg INPUT=[file join $vectors inputs.mem] -testplusarg OUTPUT=[file join $output xsim_logits.mem]"
set_property xsim.simulate.xsim.more_options $opts [get_filesets sim_1]
set_property xsim.simulate.runtime 1000ns [get_filesets sim_1]
launch_simulation
run all
close_sim
puts "XSIM FINISHED: run compare.py --backend xsim; completion alone is not acceptance"
