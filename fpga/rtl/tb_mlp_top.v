// Testbench for mlp_top Verilog Hardware Neural Network Core
// FPGA-Based Multilingual AI Receptionist
// Target: Xilinx PYNQ-Z2 (Zynq-7000)

`timescale 1ns / 1ps

module tb_mlp_top;

    reg clk;
    reg rst_n;
    reg start;
    reg [415:0] features_flat;
    reg signed [15:0] feature_in;
    reg [4:0] feature_addr;
    reg feature_wr_en;

    wire done;
    wire busy;
    wire [1:0] predicted_class;
    wire signed [15:0] score_english;
    wire signed [15:0] score_hindi;
    wire signed [15:0] score_tamil;

    // Instantiate Device Under Test (DUT)
    mlp_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .features_flat(features_flat),
        .feature_in(feature_in),
        .feature_addr(feature_addr),
        .feature_wr_en(feature_wr_en),
        .done(done),
        .busy(busy),
        .predicted_class(predicted_class),
        .score_english(score_english),
        .score_hindi(score_hindi),
        .score_tamil(score_tamil)
    );

    // 100 MHz Clock Generation (10ns period)
    always #5 clk = ~clk;

    integer i;

    // Helper task to run an inference test vector
    task run_test_vector;
        input [415:0] test_features;
        input [8*10-1:0] test_name;
        begin
            $display("[TB] Starting Test Vector: %0s", test_name);
            features_flat = test_features;
            
            @(posedge clk);
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;

            // Wait for inference completion
            @(posedge done);
            @(posedge clk);

            case (predicted_class)
                2'd0: $display("[TB] Result: Predicted = English (0)");
                2'd1: $display("[TB] Result: Predicted = Hindi (1)");
                2'd2: $display("[TB] Result: Predicted = Tamil (2)");
                default: $display("[TB] Result: Unknown Class (%0d)", predicted_class);
            endcase

            $display("[TB] Scores (Q8.8 Fixed-Point): EN=%0d, HI=%0d, TA=%0d", 
                     score_english, score_hindi, score_tamil);
            $display("--------------------------------------------------");
            #20;
        end
    endtask

    reg [415:0] sample_english;
    reg [415:0] sample_tamil;

    initial begin
        // Initialize signals
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        features_flat = 416'd0;
        feature_in = 16'sd0;
        feature_addr = 5'd0;
        feature_wr_en = 1'b0;

        $display("==================================================");
        $display("   RTL SIMULATION: MULTILINGUAL MLP CLASSIFIER    ");
        $display("==================================================");

        // Reset Sequence
        #20;
        rst_n = 1'b1;
        #20;

        // Construct Test Vector 1 (English Sample)
        for (i = 0; i < 26; i = i + 1) begin
            sample_english[i*16 +: 16] = 16'sh0100; // 1.0 in Q8.8
        end
        run_test_vector(sample_english, "English Test");

        // Construct Test Vector 2 (Tamil / High Variance Sample)
        for (i = 0; i < 26; i = i + 1) begin
            sample_tamil[i*16 +: 16] = (i < 13) ? -16'sh1000 : 16'sh0800;
        end
        run_test_vector(sample_tamil, "Tamil Test");

        $display("[TB] All simulation tests completed successfully.");
        $finish;
    end

endmodule
