# Shared project setup for XSim and implementation. No portable arithmetic stub.
if {$argc != 2} {error "Usage: -tclargs GENERATED_VECTOR_DIR OUTPUT_DIR"}
set engine [file normalize [file join [file dirname [info script]] ..]]
set vectors [file normalize [lindex $argv 0]]
set output [file normalize [lindex $argv 1]]
file mkdir $output
create_project -force ternaryguard [file join $output project] -part xc7a35tcpg236-1
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
foreach name {numeric.sv ternary_mac.sv mac_array.sv fp32_unit.sv control.sv top.sv basys3_top.sv} {
    add_files [file join $engine rtl $name]
}
add_files [file join $vectors model_rom.sv]
add_files -fileset constrs_1 [file join $engine constraints basys3.xdc]
set_property top basys3_top [get_filesets sources_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY none [get_runs synth_1]
foreach {name op} {tg_fp_add Add_Subtract tg_fp_mul Multiply tg_fp_div Divide tg_fp_sqrt Square_Root} {
    create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name $name
    set config [list CONFIG.Operation_Type $op CONFIG.A_Precision_Type Single \
        CONFIG.Result_Precision_Type Single CONFIG.Flow_Control NonBlocking \
        CONFIG.Has_RESULT_TREADY false CONFIG.Has_ARESETn true \
        CONFIG.Maximum_Latency true CONFIG.C_Rate 1]
    if {$op == "Add_Subtract"} {lappend config CONFIG.Add_Sub_Value Add}
    set_property -dict $config [get_ips $name]
    generate_target all [get_ips $name]
}
set handle [open [file join $output ip_configuration.txt] w]
foreach ip [get_ips] {
    puts $handle "IP: $ip"
    foreach property [lsort [list_property $ip]] {
        if {[string match CONFIG.* $property]} {puts $handle "$property = [get_property $property $ip]"}
    }
}
close $handle
update_compile_order -fileset sources_1
