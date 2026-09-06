## Basys 3 Constraints — xc7a35tcpg236-1
## TernaryGuard FPGA Engine
##
## This XDC provides the pin and timing constraints for synthesis targeting
## the Basys 3 board. Since we are simulation-only (no physical board),
## these constraints enable realistic synthesis/implementation reports
## (utilization, timing, power) without requiring actual hardware.

## ──────────────── Clock ────────────────
## 100 MHz system clock (Basys 3 onboard oscillator)
set_property -dict { PACKAGE_PIN W5   IOSTANDARD LVCMOS33 } [get_ports clk]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

## ──────────────── Reset ────────────────
## Active-high reset mapped to center pushbutton
set_property -dict { PACKAGE_PIN U18  IOSTANDARD LVCMOS33 } [get_ports rst]

## ──────────────── UART ────────────────
## USB-UART bridge (per Digilent master XDC: RsRx=B18, RsTx=A18)
set_property -dict { PACKAGE_PIN B18  IOSTANDARD LVCMOS33 } [get_ports uart_rx]
set_property -dict { PACKAGE_PIN A18  IOSTANDARD LVCMOS33 } [get_ports uart_tx]

## ──────────────── Status LEDs ────────────────
## LED[0]: inference active, LED[1]: classification result
set_property -dict { PACKAGE_PIN U16  IOSTANDARD LVCMOS33 } [get_ports {led[0]}]
set_property -dict { PACKAGE_PIN E19  IOSTANDARD LVCMOS33 } [get_ports {led[1]}]
set_property -dict { PACKAGE_PIN U19  IOSTANDARD LVCMOS33 } [get_ports {led[2]}]
set_property -dict { PACKAGE_PIN V19  IOSTANDARD LVCMOS33 } [get_ports {led[3]}]

## ──────────────── Configuration ────────────────
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

## ──────────────── Timing ────────────────
## Relax timing for initial development; tighten in optimization pass
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets clk_IBUF]
