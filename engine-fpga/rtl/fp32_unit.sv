`timescale 1ns/1ps
// Vendor FP32 IP is deliberately separate from the multiplier-free ternary MAC.
// One outstanding transaction; all selected cores are nonblocking AXI, rate 1.
module fp32_unit (
    input wire clk, rst, request,
    input wire [1:0] operation, // 0 add, 1 multiply, 2 divide, 3 sqrt
    input wire [31:0] a, b,
    output wire valid,
    output wire [31:0] result
);
`ifdef TG_PORTABLE_SIM
    // NON-SYNTHESIZABLE local arithmetic model. Not AMD IP or XSim validation.
`ifdef VERILATOR
    import "DPI-C" function int unsigned tg_f32(input int unsigned op,aa,bb);
`endif
    reg [31:0] pending, data;
    reg done;
    integer cycles;
    always @(posedge clk) begin
        done <= 0;
        if (rst) begin cycles <= 0; pending <= 0; data <= 0; end
        else if (request) begin
            if (cycles!=0) $fatal(1,"overlapping FP requests");
`ifdef VERILATOR
            pending <= tg_f32({30'b0,operation},a,b);
`else
            pending <= $tg_f32(operation,a,b);
`endif
            case (operation)
                0: cycles <= 3;
                1: cycles <= 4;
                default: cycles <= 12;
            endcase
        end else if (cycles>0) begin
            cycles <= cycles-1;
            if (cycles==1) begin data <= pending; done <= 1; end
        end
    end
    assign valid=done;
    assign result=data;
`else
    wire [3:0] ready;
    wire [31:0] r0,r1,r2,r3;
    reg [1:0] selected;
    always @(posedge clk) if (rst) selected<=0; else if (request) selected<=operation;
    tg_fp_add add_ip(.aclk(clk),.aresetn(!rst),
        .s_axis_a_tvalid(request && operation==0),.s_axis_a_tdata(a),
        .s_axis_b_tvalid(request && operation==0),.s_axis_b_tdata(b),
        .m_axis_result_tvalid(ready[0]),.m_axis_result_tdata(r0));
    tg_fp_mul mul_ip(.aclk(clk),.aresetn(!rst),
        .s_axis_a_tvalid(request && operation==1),.s_axis_a_tdata(a),
        .s_axis_b_tvalid(request && operation==1),.s_axis_b_tdata(b),
        .m_axis_result_tvalid(ready[1]),.m_axis_result_tdata(r1));
    tg_fp_div div_ip(.aclk(clk),.aresetn(!rst),
        .s_axis_a_tvalid(request && operation==2),.s_axis_a_tdata(a),
        .s_axis_b_tvalid(request && operation==2),.s_axis_b_tdata(b),
        .m_axis_result_tvalid(ready[2]),.m_axis_result_tdata(r2));
    tg_fp_sqrt sqrt_ip(.aclk(clk),.aresetn(!rst),
        .s_axis_a_tvalid(request && operation==3),.s_axis_a_tdata(a),
        .m_axis_result_tvalid(ready[3]),.m_axis_result_tdata(r3));
    assign valid=ready[selected];
    assign result=selected==0 ? r0 : selected==1 ? r1 : selected==2 ? r2 : r3;
`endif
endmodule
