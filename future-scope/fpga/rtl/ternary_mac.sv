`timescale 1ns/1ps
// One signed int64 accumulator. Reserved code is an error, never a silent zero.
(* use_dsp = "no" *) module ternary_mac (
    input wire clk, rst, clear, enable,
    input wire [1:0] code,
    input wire signed [63:0] x,
    output reg signed [63:0] sum,
    output reg fault
);
    always @(posedge clk) begin
        if (rst || clear) begin sum <= 0; fault <= 0; end
        else if (enable) begin
            case (code)
                2'b00: sum <= sum;
                2'b01: sum <= sum + x;
                2'b10: sum <= sum - x;
                2'b11: fault <= 1;
            endcase
        end
    end
endmodule
