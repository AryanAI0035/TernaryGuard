`timescale 1ns/1ps
module tb_numeric;
    import tg_numeric::*;
    reg [199:0] vectors[0:4095];
    string path;
    integer count,k;
    initial begin
        if(!$value$plusargs("VECTORS=%s",path)) $fatal(1,"numeric vectors required");
        if(!$value$plusargs("COUNT=%d",count)) $fatal(1,"numeric count required");
        $readmemh(path,vectors,0,count-1);
        for(k=0;k<count;k=k+1) begin
            if(to_fixed(vectors[k][199:168],vectors[k][167:160])!==vectors[k][159:96])
                $fatal(1,"float-to-fixed mismatch %0d",k);
            if(restore(vectors[k][95:32],vectors[k][167:160])!==vectors[k][31:0]) begin
                $fatal(1,"restore mismatch %0d",k);
            end
        end
        $display("NUMERIC_PASS cases=%0d",count);$finish;
    end
endmodule
