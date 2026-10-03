`timescale 1ns/1ps
// Sequential RMSNorm and postprocessing, eight parallel int64 dot accumulators.
// Stream contract: exactly 20 preprocessed binary32 words, then 11 logit words.
module control (
    input wire clk,rst,
    input wire in_valid,
    output wire in_ready,
    input wire [31:0] in_data,
    output wire out_valid,
    input wire out_ready,
    output wire [31:0] out_data,
    output wire [3:0] out_index,
    output reg [3:0] prediction,
    output wire busy,
    output reg done,error
);
    import tg_numeric::*;
    localparam LOAD=0, NINIT=1, SQUARE=2, SDONE=3, ACC=4, ADONE=5,
        MEAN=6, MDONE=7, EPS=8, EDONE=9, ROOT=10, RDONE=11,
        NDIV=12, NDDONE=13, GAMMA=14, GDONE=15, SCAN=16,
        FIX=17, CLEAR=18, DOT=19, DRAIN=20, SCALE=21, SCDONE=22,
        BIAS=23, BDONE=24, OUTPUT=25, WAIT_FP=26, FAILED=27;
    reg [4:0] state,return_state;
    reg [1:0] layer;
    reg [6:0] index,column,base_row,lane_index,output_index;
    reg [7:0] max_exp;
    reg [31:0] act[0:63];
    reg signed [63:0] fixed[0:63];
    reg [31:0] logits[0:10];
    reg [31:0] squares,temp,rms,best,fp_value;
    reg request;
    reg [1:0] operation;
    reg [31:0] a,b;
    wire fp_valid;
    wire [31:0] fp_result;
    wire [6:0] ins=layer==0 ? 20 : layer==1 ? 64 : 32;
    wire [6:0] outs=layer==0 ? 64 : layer==1 ? 32 : 11;
    wire [31:0] ins_f32=layer==0 ? 32'h41a00000 : 32'h42800000;
    wire [15:0] codes;
    wire [511:0] sums;
    wire mac_fault;
    wire [31:0] bias,gamma,scale,epsilon;
    wire [6:0] row=base_row+lane_index;
    model_rom rom(.layer(layer),.column(column),.base_row(base_row),.row(row),
        .norm_index(index),.codes(codes),.bias(bias),.gamma(gamma),.scale(scale),.epsilon(epsilon));
    mac_array #(.LANES(8)) array(.clk(clk),.rst(rst),.clear(state==CLEAR),
        .enable(state==DOT),.codes(codes),.x(fixed[column[5:0]]),.sums(sums),.fault(mac_fault));
    fp32_unit fp(.clk(clk),.rst(rst),.request(request),.operation(operation),
        .a(a),.b(b),.valid(fp_valid),.result(fp_result));
    assign in_ready=state==LOAD;
    assign out_valid=state==OUTPUT;
    assign out_data=logits[output_index[3:0]];
    assign out_index=output_index[3:0];
    assign busy=state!=LOAD && state!=FAILED;
    task issue(input [1:0] op,input [31:0] x,input [31:0] y,input [4:0] resume);
        begin request<=1; operation<=op; a<=x; b<=y;
            return_state<=resume; state<=WAIT_FP; end
    endtask
    always @(posedge clk) begin
        request<=0; done<=0;
        if (rst) begin
            state<=LOAD; layer<=0; index<=0; column<=0; base_row<=0;
            lane_index<=0; output_index<=0; error<=0; prediction<=0;
            max_exp<=0; squares<=0; temp<=0; rms<=0; best<=0;
            operation<=0; a<=0; b<=0; return_state<=LOAD;
        end else if (mac_fault) begin error<=1; state<=FAILED; end
        else case (state)
            LOAD: if (in_valid) begin
                if (!finite_normal_or_zero(in_data)) begin error<=1; state<=FAILED; end
                else begin
                    act[index[5:0]]<=in_data;
                    if (index==19) begin index<=0; layer<=0; state<=NINIT; end
                    else index<=index+1;
                end
            end
            NINIT: begin index<=0; squares<=0; max_exp<=0;
                state<=layer<2 ? SQUARE : SCAN; end
            SQUARE: issue(1,act[index[5:0]],act[index[5:0]],SDONE);
            SDONE: begin temp<=fp_value; state<=ACC; end
            ACC: issue(0,squares,temp,ADONE);
            ADONE: begin squares<=fp_value;
                if (index==ins-1) state<=MEAN;
                else begin index<=index+1; state<=SQUARE; end end
            MEAN: issue(2,squares,ins_f32,MDONE);
            MDONE: begin temp<=fp_value; state<=EPS; end
            EPS: issue(0,temp,epsilon,EDONE);
            EDONE: begin temp<=fp_value; state<=ROOT; end
            ROOT: issue(3,temp,0,RDONE);
            RDONE: begin
                if (fp_value[31] || fp_value[30:0]==0) begin error<=1; state<=FAILED; end
                else begin rms<=fp_value; index<=0; state<=NDIV; end
            end
            NDIV: issue(2,act[index[5:0]],rms,NDDONE);
            NDDONE: begin temp<=fp_value; state<=GAMMA; end
            GAMMA: issue(1,temp,gamma,GDONE);
            GDONE: begin act[index[5:0]]<=fp_value;
                if (fp_value[30:23]>max_exp) max_exp<=fp_value[30:23];
                if (index==ins-1) begin index<=0; state<=FIX; end
                else begin index<=index+1; state<=NDIV; end end
            SCAN: begin
                if (act[index[5:0]][30:23]>max_exp) max_exp<=act[index[5:0]][30:23];
                if (index==ins-1) begin index<=0; state<=FIX; end
                else index<=index+1;
            end
            FIX: begin fixed[index[5:0]]<=to_fixed(act[index[5:0]],max_exp);
                if (index==ins-1) begin base_row<=0; column<=0; state<=CLEAR; end
                else index<=index+1; end
            CLEAR: begin column<=0; lane_index<=0; state<=DOT; end
            DOT: if (column==ins-1) state<=DRAIN; else column<=column+1;
            DRAIN: state<=SCALE;
            SCALE: begin
                temp<=restore(sums[lane_index*64+:64],max_exp);
                if (!finite_normal_or_zero(restore(sums[lane_index*64+:64],max_exp))) begin
                    error<=1; state<=FAILED;
                end else issue(1,restore(sums[lane_index*64+:64],max_exp),scale,SCDONE);
            end
            SCDONE: begin temp<=fp_value; state<=BIAS; end
            BIAS: issue(0,temp,bias,BDONE);
            BDONE: begin
                act[row[5:0]]<=(layer<2 && fp_value[31] && fp_value[30:0]!=0) ? 0 : fp_value;
                if (layer==2) begin
                    logits[row[3:0]]<=fp_value;
                    if (row==0 || greater(fp_value,best)) begin best<=fp_value; prediction<=row[3:0]; end
                end
                if (row==outs-1) begin
                    if (layer==2) begin output_index<=0; state<=OUTPUT; end
                    else begin layer<=layer+1; state<=NINIT; end
                end else if (lane_index==7) begin base_row<=base_row+8; state<=CLEAR; end
                else begin lane_index<=lane_index+1; state<=SCALE; end
            end
            OUTPUT: if (out_ready) begin
                if (output_index==10) begin done<=1; index<=0; state<=LOAD; end
                else output_index<=output_index+1;
            end
            WAIT_FP: if (fp_valid) begin
                if (!finite_normal_or_zero(fp_result)) begin error<=1; state<=FAILED; end
                else begin fp_value<=fp_result; state<=return_state; end
            end
            FAILED: error<=1;
            default: begin error<=1; state<=FAILED; end
        endcase
    end
endmodule
