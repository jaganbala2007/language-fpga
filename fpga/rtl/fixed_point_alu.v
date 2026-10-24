// Fixed-Point Arithmetic Unit (Q8.8 Format)
// FPGA-Based Multilingual AI Receptionist
// Target: Xilinx Zynq-7000 (PYNQ-Z2)

`timescale 1ns / 1ps

module fixed_point_mac (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,
    input  wire        clear_acc,
    input  wire signed [15:0] a_in,      // Operand A (Q8.8, 16-bit signed)
    input  wire signed [15:0] b_in,      // Operand B (Q8.8, 16-bit signed)
    input  wire signed [31:0] bias_in,   // Initial bias (Q16.16 or shifted Q8.8)
    output reg  signed [31:0] acc_out    // Accumulated result
);

    wire signed [31:0] product;
    assign product = a_in * b_in; // 16x16 -> 32-bit (Q16.16)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= 32'sd0;
        end else if (clear_acc) begin
            acc_out <= bias_in;
        end else if (enable) begin
            acc_out <= acc_out + product;
        end
    end

endmodule


module relu_activation (
    input  wire signed [15:0] in_val,    // Q8.8 signed input
    output wire signed [15:0] out_val   // ReLU output: max(0, in_val)
);

    assign out_val = (in_val[15] == 1'b1) ? 16'sh0000 : in_val;

endmodule
