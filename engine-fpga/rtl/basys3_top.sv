`timescale 1ns/1ps
// Small pin-count synthesis wrapper. Synchronous bit stream, NOT a UART.
// LSB-first, 32 handshakes per word. No physical board validation is claimed.
module basys3_top (
    input wire clk,rst,in_bit,in_valid,out_ready,
    output wire in_ready,out_bit,out_valid,busy,done,error,
    output wire [3:0] prediction
);
    reg [31:0] input_shift;
    reg [4:0] input_count,output_count;
    reg word_valid;
    wire word_ready,word_out_valid,word_out_ready;
    wire [31:0] word_out_data;
    wire [3:0] unused_index;
    assign in_ready=word_ready && !word_valid;
    assign out_valid=word_out_valid;
    assign out_bit=word_out_data[output_count];
    assign word_out_ready=out_ready && output_count==31;
    top accelerator(.clk(clk),.rst(rst),.in_valid(word_valid),.in_ready(word_ready),
        .in_data(input_shift),.out_valid(word_out_valid),.out_ready(word_out_ready),
        .out_data(word_out_data),.out_index(unused_index),.prediction(prediction),
        .busy(busy),.done(done),.error(error));
    always @(posedge clk) begin
        if (rst) begin input_count<=0; output_count<=0; word_valid<=0; input_shift<=0; end
        else begin
            if (word_valid && word_ready) word_valid<=0;
            if (in_valid && in_ready) begin
                input_shift[input_count]<=in_bit;
                if (input_count==31) begin input_count<=0; word_valid<=1; end
                else input_count<=input_count+1;
            end
            if (out_valid && out_ready) output_count<=output_count+1;
        end
    end
endmodule
