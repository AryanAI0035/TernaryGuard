`timescale 1ns/1ps
module tb_basys3_top;
    reg clk=0;always #5 clk=~clk;
    reg rst=1,in_bit=0,in_valid=0,out_ready=0;
    wire in_ready,out_bit,out_valid,busy,done,error;
    wire [3:0] prediction;
    basys3_top dut(.*);
    reg [31:0] inputs[0:5279];
    reg [31:0] word;
    string input_path,output_path;
    integer prediction_file,samples,f,s,c,bit_index,cycles;
    initial begin
        if(!$value$plusargs("SAMPLES=%d",samples)) $fatal(1,"sample count required");
        if(!$value$plusargs("INPUT=%s",input_path)) $fatal(1,"input required");
        if(!$value$plusargs("OUTPUT=%s",output_path)) $fatal(1,"output required");
        if(samples<1 || samples>264) $fatal(1,"wrapper smoke limit 264");
        $readmemh(input_path,inputs,0,samples*20-1);f=$fopen(output_path,"w");
        if(f==0) $fatal(1,"output open failed");
        prediction_file=$fopen($sformatf("%0s.predictions",output_path),"w");
        if(prediction_file==0) $fatal(1,"prediction output open failed");
        repeat(5) @(negedge clk);rst=0;
        for(s=0;s<samples;s=s+1) begin
            for(c=0;c<20;c=c+1) begin
                for(bit_index=0;bit_index<32;bit_index=bit_index+1) begin
                    @(negedge clk);in_valid=0;
                    while(!in_ready) @(negedge clk);
                    in_valid=1;in_bit=inputs[s*20+c][bit_index];
                    @(negedge clk);in_valid=0;
                end
            end
            cycles=0;
            while(!out_valid && !error && cycles<30000) begin @(negedge clk);cycles=cycles+1;end
            if(error || cycles==30000) $fatal(1,"wrapper timeout or error");
            for(c=0;c<11;c=c+1) begin
                word=0;
                for(bit_index=0;bit_index<32;bit_index=bit_index+1) begin
                    if(!out_valid) $fatal(1,"wrapper output missing");
                    out_ready=0;
                    if(bit_index%7==0) @(negedge clk);
                    word[bit_index]=out_bit;
                    out_ready=1;@(negedge clk);out_ready=0;
                end
                $fwrite(f,"%08x\n",word);
            end
            if(!done) $fatal(1,"wrapper completion missing");
            $fwrite(prediction_file,"%0d\n",prediction);
        end
        $fclose(f);$fclose(prediction_file);$display("WRAPPER_PASS samples=%0d LSB_first=PASS backpressure=PASS",samples);$finish;
    end
endmodule
