# Full target-part synthesis/place/route. A failed gate leaves raw reports intact.
source [file join [file dirname [info script]] create_project.tcl]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]]!="100%"} {error "Synthesis failed"}
open_run synth_1
report_utilization -hierarchical -file [file join $output post_synth_utilization.rpt]
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]]!="100%"} {error "Implementation failed"}
open_run impl_1
report_utilization -hierarchical -file [file join $output routed_utilization.rpt]
report_timing_summary -report_unconstrained -file [file join $output routed_timing.rpt]
report_route_status -file [file join $output route_status.rpt]
report_drc -file [file join $output drc.rpt]
check_timing -verbose -file [file join $output check_timing.rpt]
# Vectorless power estimate from placed/routed design, NOT measured board power.
report_power -file [file join $output power_vectorless.rpt]
write_checkpoint -force [file join $output routed.dcp]
set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup]==0 || [llength $hold]==0} {error "Missing timing paths; cannot claim closure"}
set wns [get_property SLACK $setup]
set whs [get_property SLACK $hold]
puts "TG_ROUTED_WNS_NS=$wns TG_ROUTED_WHS_NS=$whs"
set dsp_mac [get_cells -hierarchical -filter {REF_NAME =~ DSP* && NAME =~ *core/array/*}]
if {[llength $dsp_mac]!=0} {error "DSP primitive found inside ternary MAC array: $dsp_mac"}
puts "TG_MAC_ARRAY_DSP_COUNT=0"
if {$wns<0 || $whs<0} {error "TIMING NOT CLOSED: inspect routed_timing.rpt; do not lower clock silently"}
set violations [get_drc_violations -filter {SEVERITY == Error}]
if {[llength $violations]!=0} {error "DRC errors remain: $violations"}
puts "TG_IMPLEMENTATION_GATES_PASS; inspect unconstrained paths, TNS/THS and reports before acceptance"
