`timescale 1ns/1ps
module mac_array #(parameter LANES=8) (
    input wire clk, rst, clear, enable,
    input wire [LANES*2-1:0] codes,
    input wire signed [63:0] x,
    output wire [LANES*64-1:0] sums,
    output wire fault
);
    wire [LANES-1:0] faults;
    genvar k;
    generate for (k=0; k<LANES; k=k+1) begin: lane
        ternary_mac mac(.clk(clk), .rst(rst), .clear(clear), .enable(enable),
            .code(codes[k*2+:2]), .x(x), .sum(sums[k*64+:64]), .fault(faults[k]));
    end endgenerate
    assign fault = |faults;
endmodule
