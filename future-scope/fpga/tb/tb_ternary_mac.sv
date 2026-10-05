`timescale 1ns/1ps
module tb_ternary_mac;
    reg clk=0;always #5 clk=~clk;
    reg rst=1,clear=0,enable=0;
    reg [1:0] code=0;
    reg signed [63:0] x=0,expected=0;
    wire signed [63:0] sum;
    wire fault;
    ternary_mac dut(.*);
    integer i;reg [63:0] rng=64'hc0ffee,mag;
    initial begin
        repeat(3) @(negedge clk);rst=0;
        for(i=0;i<8192;i=i+1) begin
            rng=rng^(rng<<13);rng=rng^(rng>>7);rng=rng^(rng<<17);
            mag=rng&64'h0000007fffffffff;x=rng[63] ? -mag : mag;
            code=i%3;enable=1;
            if(code==1) expected=expected+x;
            else if(code==2) expected=expected-x;
            @(negedge clk);
            if(sum!==expected || fault) $fatal(1,"MAC mismatch at %0d",i);
        end
        enable=0;x=123;code=1;@(negedge clk);
        if(sum!==expected) $fatal(1,"enable gating failed");
        clear=1;@(negedge clk);clear=0;
        if(sum!==0 || fault) $fatal(1,"clear failed");
        enable=1;code=3;@(negedge clk);
        if(!fault || sum!==0) $fatal(1,"reserved code not rejected");
        $display("MAC_PASS signed_int64_operations=8192 skip=PASS enable=PASS clear=PASS reserved_code=PASS");$finish;
    end
endmodule
