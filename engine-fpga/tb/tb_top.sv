`timescale 1ns/1ps
module tb_top;
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1,in_valid=0,out_ready=0;
    reg [31:0] in_data=0;
    wire in_ready,out_valid,busy,done,error;
    wire [31:0] out_data;
    wire [3:0] out_index,prediction;
    top dut(.*);
    reg [31:0] inputs[0:1380799]; // all 69,040 rows x 20
    integer prediction_file,samples,output_file,sample,column,cycles,total_cycles=0,stalls;
    string input_path,output_path;
    reg [31:0] held_data;
    reg [3:0] held_index;
    initial begin
        if (!$value$plusargs("SAMPLES=%d",samples)) $fatal(1,"SAMPLES required");
        if (!$value$plusargs("INPUT=%s",input_path)) $fatal(1,"INPUT required");
        if (!$value$plusargs("OUTPUT=%s",output_path)) $fatal(1,"OUTPUT required");
        if (samples<1 || samples>69040) $fatal(1,"invalid sample count");
        $readmemh(input_path,inputs,0,samples*20-1);
        output_file=$fopen(output_path,"w");
        if (output_file==0) $fatal(1,"cannot open output");
        prediction_file=$fopen($sformatf("%0s.predictions",output_path),"w");
        if (prediction_file==0) $fatal(1,"cannot open prediction output");
        repeat (5) @(negedge clk);rst=0;
        for (sample=0;sample<samples;sample=sample+1) begin
            cycles=0;
            for (column=0;column<20;column=column+1) begin
                // Idle gaps exercise input backpressure/handshake sequencing.
                @(negedge clk); in_valid=0;
                if (sample<3) repeat(column%3) @(negedge clk);
                while (!in_ready) @(negedge clk);
                in_data=inputs[sample*20+column];in_valid=1;
                @(negedge clk);in_valid=0;
            end
            while (!out_valid && !error && cycles<30000) begin
                @(negedge clk);cycles=cycles+1;
            end
            if (error || cycles==30000) $fatal(1,"sample %0d failed: error=%0d cycles=%0d",sample,error,cycles);
            for (column=0;column<11;column=column+1) begin
                if (!out_valid || out_index!=4'(column)) $fatal(1,"output sequence sample=%0d",sample);
                held_data=out_data;held_index=out_index;
                // Stall each of the first three samples' outputs, assert stability.
                out_ready=0;
                if (sample<3) for(stalls=0;stalls<column%3+1;stalls=stalls+1) begin
                    @(negedge clk);
                    if (!out_valid || out_data!==held_data || out_index!==held_index)
                        $fatal(1,"output changed under backpressure");
                end
                $fwrite(output_file,"%08x\n",out_data);
                out_ready=1;@(negedge clk);out_ready=0;
            end
            if (!done || !in_ready) $fatal(1,"completion handshake missing");
            $fwrite(prediction_file,"%0d\n",prediction);
            total_cycles=total_cycles+cycles;
            if ((sample+1)%1000==0) $display("RTL_PROGRESS samples=%0d/%0d",sample+1,samples);
        end
        $fclose(output_file);$fclose(prediction_file);
        // The advertised finite-normal/zero contract must reject NaN.
        @(negedge clk); in_data=32'h7fc00000;in_valid=1;
        @(negedge clk); in_valid=0;
        if (!error || in_ready) $fatal(1,"NaN input was not rejected");
        rst=1;repeat(5) @(negedge clk);rst=0;
        @(negedge clk);if(error || !in_ready) $fatal(1,"reset failed");
        $display("RTL_PASS samples=%0d post_input_wait_cycles=%0d backpressure=PASS NaN_rejection=PASS reset=PASS",samples,total_cycles);
        $finish;
    end
endmodule
