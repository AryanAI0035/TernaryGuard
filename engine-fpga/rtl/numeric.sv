`timescale 1ns/1ps
// Bit operations for the C engine's shared exponent, truncation and RNE restore.
// Supported contract: finite binary32 normals and zeros; faults reject subnormals.
package tg_numeric;
    function automatic [63:0] to_fixed(input [31:0] value, input [7:0] max_exp);
        reg [63:0] magnitude;
        integer amount;
        begin
            magnitude = {40'b0, 1'b1, value[22:0]};
            amount = 16 + integer'(value[30:23]) - integer'(max_exp);
            if (value[30:0] == 0) magnitude = 0;
            else if (amount >= 0) magnitude = magnitude << amount;
            else magnitude = magnitude >> (-amount);
            to_fixed = value[31] ? -magnitude : magnitude;
        end
    endfunction
    // Cast int64 to binary32 round-to-nearest-even, then apply C's ldexpf.
    function automatic [31:0] restore(input [63:0] value, input [7:0] max_exp);
        reg [63:0] magnitude, remainder, half;
        reg [24:0] significand;
        integer msb, k, right_shift, exponent;
        begin
            magnitude = value[63] ? -value : value;
            msb = 0;
            for (k=0; k<64; k=k+1) if (magnitude[k]) msb = k;
            right_shift = msb - 23;
            if (right_shift > 0) begin
                significand = 25'(magnitude >> right_shift);
                remainder = magnitude & ((64'b1 << right_shift)-1);
                half = 64'b1 << (right_shift-1);
                if (remainder > half || (remainder == half && significand[0]))
                    significand = significand + 1;
            end else significand = 25'(magnitude << (-right_shift));
            exponent = msb + 127 - (166-integer'(max_exp));
            if (significand[24]) begin significand = significand >> 1; exponent=exponent+1; end
            if (magnitude==0) restore=0;
            else if (exponent<=0 || exponent>=255) restore=32'h7fc00000;
            else restore={value[63],8'(exponent),significand[22:0]};
        end
    endfunction
    function automatic finite_normal_or_zero(input [31:0] value);
        finite_normal_or_zero = value[30:23]!=255 &&
            (value[30:23]!=0 || value[22:0]==0);
    endfunction
    function automatic greater(input [31:0] a, input [31:0] b);
        begin
            if (a[30:0]==0 && b[30:0]==0) greater=0;
            else if (a[31]!=b[31]) greater=!a[31];
            else greater=a[31] ? a[30:0]<b[30:0] : a[30:0]>b[30:0];
        end
    endfunction
endpackage
