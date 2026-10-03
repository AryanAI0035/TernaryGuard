# Simulation/synthesis target only; synchronous bit-stream wrapper, NOT UART.
# Pin map from https://github.com/Digilent/digilent-xdc/blob/master/Basys-3-Master.xdc
set_property -dict {PACKAGE_PIN W5 IOSTANDARD LVCMOS33} [get_ports clk]
create_clock -name sys_clk -period 10.000 [get_ports clk]
foreach {port pin} {rst U18 in_bit J1 in_valid L2 in_ready J2 out_bit G2 out_valid H1 out_ready K2 busy H2 done G3 error U16} {
    set_property PACKAGE_PIN $pin [get_ports $port]
    set_property IOSTANDARD LVCMOS33 [get_ports $port]
}
foreach {port pin} {prediction[0] E19 prediction[1] U19 prediction[2] V19 prediction[3] W18} {
    set_property PACKAGE_PIN $pin [get_ports $port]
    set_property IOSTANDARD LVCMOS33 [get_ports $port]
}
# Synchronous hypothetical source/sink: 2ns external budget, 0ns minimum.
# These are design constraints, not measured interface timings.
set_input_delay -clock sys_clk -max 2.0 [get_ports {rst in_bit in_valid out_ready}]
set_input_delay -clock sys_clk -min 0.0 [get_ports {rst in_bit in_valid out_ready}]
set_output_delay -clock sys_clk -max 2.0 [get_ports {in_ready out_bit out_valid busy done error prediction[*]}]
set_output_delay -clock sys_clk -min 0.0 [get_ports {in_ready out_bit out_valid busy done error prediction[*]}]
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
# No CLOCK_DEDICATED_ROUTE override and no blanket false-path exceptions.
