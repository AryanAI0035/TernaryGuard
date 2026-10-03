`timescale 1ns/1ps
module tb_mac_array;
    reg clk=0;always #5 clk=~clk;
    reg rst=1,clear=0,enable=0;
    reg [15:0] codes=0;
    reg signed [63:0] x=0;
    wire [511:0] sums;
    wire fault;
    reg signed [63:0] expected[0:7];
    integer row,lane,term;
    mac_array dut(.*);
    initial begin
        repeat(3) @(negedge clk);rst=0;
        for(row=0;row<16;row=row+1) begin
            clear=1;@(negedge clk);clear=0;
            for(lane=0;lane<8;lane=lane+1) expected[lane]=0;
            for(term=0;term<64;term=term+1) begin
                x=(64'sd1<<39)-row*12345-term*987;
                if(term%2) x=-x;
                for(lane=0;lane<8;lane=lane+1) begin
                    codes[lane*2+:2]=(row+lane+term)%3;
                    case(codes[lane*2+:2])
                        1:expected[lane]=expected[lane]+x;
                        2:expected[lane]=expected[lane]-x;
                    endcase
                end
                enable=1;@(negedge clk);
            end
            enable=0;
            for(lane=0;lane<8;lane=lane+1)
                if(sums[lane*64+:64]!==expected[lane]) $fatal(1,"array mismatch row=%0d lane=%0d",row,lane);
            if(fault) $fatal(1,"unexpected array fault");
        end
        codes=0;codes[7*2+:2]=3;enable=1;@(negedge clk);
        if(!fault) $fatal(1,"array reserved code missing");
        $display("ARRAY_PASS lanes=8 batches=16 terms_per_batch=64 reserved_lane7=PASS");$finish;
    end
endmodule
