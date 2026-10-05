`timescale 1ns/1ps
// Internal word-stream interface for accelerator integration and verification.
module top (
    input wire clk,rst,in_valid,out_ready,
    input wire [31:0] in_data,
    output wire in_ready,out_valid,
    output wire [31:0] out_data,
    output wire [3:0] out_index,prediction,
    output wire busy,done,error
);
    control core(.*);
endmodule
