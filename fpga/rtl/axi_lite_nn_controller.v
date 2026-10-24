// AXI4-Lite Slave Wrapper for Multilingual MLP Neural Network Accelerator
// FPGA-Based Multilingual AI Receptionist
// Target: Xilinx PYNQ-Z2 (Zynq-7000)

`timescale 1ns / 1ps

module axi_lite_nn_controller # (
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 8
)(
    // AXI4-Lite Global Signals
    input  wire                                  S_AXI_ACLK,
    input  wire                                  S_AXI_ARESETN,

    // AXI4-Lite Write Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]        S_AXI_AWADDR,
    input  wire [2:0]                            S_AXI_AWPROT,
    input  wire                                  S_AXI_AWVALID,
    output wire                                  S_AXI_AWREADY,

    // AXI4-Lite Write Data Channel
    input  wire [C_S_AXI_DATA_WIDTH-1:0]        S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0]    S_AXI_WSTRB,
    input  wire                                  S_AXI_WVALID,
    output wire                                  S_AXI_WREADY,

    // AXI4-Lite Write Response Channel
    output wire [1:0]                            S_AXI_BRESP,
    output wire                                  S_AXI_BVALID,
    input  wire                                  S_AXI_BREADY,

    // AXI4-Lite Read Address Channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]        S_AXI_ARADDR,
    input  wire [2:0]                            S_AXI_ARPROT,
    input  wire                                  S_AXI_ARVALID,
    output wire                                  S_AXI_ARREADY,

    // AXI4-Lite Read Data Channel
    output wire [C_S_AXI_DATA_WIDTH-1:0]        S_AXI_RDATA,
    output wire [1:0]                            S_AXI_RRESP,
    output wire                                  S_AXI_RVALID,
    input  wire                                  S_AXI_RREADY
);

    // Register Map Offsets
    // 0x00: Control/Status (Bit 0: ap_start, Bit 1: ap_done, Bit 2: ap_idle, Bit 3: ap_ready)
    // 0x10: Predicted Class (0: English, 1: Hindi, 2: Tamil)
    // 0x14: English Score (Q8.8)
    // 0x18: Hindi Score (Q8.8)
    // 0x1C: Tamil Score (Q8.8)
    // 0x20 - 0x84: Feature 0 to Feature 25 (26 x 32-bit registers)

    reg [31:0] reg_ctrl;
    reg [31:0] reg_features [0:25];

    reg axi_awready;
    reg axi_wready;
    reg [1:0] axi_bresp;
    reg axi_bvalid;
    reg axi_arready;
    reg [31:0] axi_rdata;
    reg [1:0] axi_rresp;
    reg axi_rvalid;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    wire nn_start;
    wire nn_done;
    wire nn_busy;
    wire [1:0] nn_pred_class;
    wire signed [15:0] nn_score_en;
    wire signed [15:0] nn_score_hi;
    wire signed [15:0] nn_score_ta;

    assign nn_start = reg_ctrl[0];

    // Pack 26 feature registers (Q8.8) into flat bus
    wire [415:0] flat_features;
    genvar g;
    generate
        for (g = 0; g < 26; g = g + 1) begin: pack_feat
            assign flat_features[g*16 +: 16] = reg_features[g][15:0];
        end
    endgenerate

    // Instantiate MLP core
    mlp_top u_mlp_core (
        .clk             (S_AXI_ACLK),
        .rst_n           (S_AXI_ARESETN),
        .start           (nn_start),
        .features_flat   (flat_features),
        .feature_in      (16'sd0),
        .feature_addr    (5'd0),
        .feature_wr_en   (1'b0),
        .done            (nn_done),
        .busy            (nn_busy),
        .predicted_class (nn_pred_class),
        .score_english   (nn_score_en),
        .score_hindi     (nn_score_hi),
        .score_tamil     (nn_score_ta)
    );

    // AXI Write Address & Data Handshake
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 1'b0;
            axi_wready  <= 1'b0;
            axi_bvalid  <= 1'b0;
            axi_bresp   <= 2'b00;
            reg_ctrl    <= 32'd0;
        end else begin
            // Address Ready
            if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID) begin
                axi_awready <= 1'b1;
                axi_wready  <= 1'b1;
            end else begin
                axi_awready <= 1'b0;
                axi_wready  <= 1'b0;
            end

            // Write Data
            if (axi_awready && S_AXI_AWVALID && axi_wready && S_AXI_WVALID) begin
                if (S_AXI_AWADDR[7:0] == 8'h00) begin
                    reg_ctrl <= S_AXI_WDATA;
                end else if (S_AXI_AWADDR[7:0] >= 8'h20 && S_AXI_AWADDR[7:0] <= 8'h84) begin
                    reg_features[(S_AXI_AWADDR[7:0] - 8'h20) >> 2] <= S_AXI_WDATA;
                end
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b00; // OKAY
            end else if (S_AXI_BREADY && axi_bvalid) begin
                axi_bvalid <= 1'b0;
            end

            // Auto-clear start bit when NN finishes
            if (nn_done) begin
                reg_ctrl[0] <= 1'b0; // clear start
                reg_ctrl[1] <= 1'b1; // set done
            end
        end
    end

    // AXI Read Channel Handshake
    always @(posedge S_AXI_ACLK or negedge S_AXI_ARESETN) begin
        if (!S_AXI_ARESETN) begin
            axi_arready <= 1'b0;
            axi_rvalid  <= 1'b0;
            axi_rresp   <= 2'b00;
            axi_rdata   <= 32'd0;
        end else begin
            if (~axi_arready && S_AXI_ARVALID) begin
                axi_arready <= 1'b1;
                axi_rvalid  <= 1'b1;
                axi_rresp   <= 2'b00;

                case (S_AXI_ARADDR[7:0])
                    8'h00: axi_rdata <= {28'd0, 1'b0, ~nn_busy, nn_done, nn_start};
                    8'h10: axi_rdata <= {30'd0, nn_pred_class};
                    8'h14: axi_rdata <= {{16{nn_score_en[15]}}, nn_score_en};
                    8'h18: axi_rdata <= {{16{nn_score_hi[15]}}, nn_score_hi};
                    8'h1C: axi_rdata <= {{16{nn_score_ta[15]}}, nn_score_ta};
                    default: begin
                        if (S_AXI_ARADDR[7:0] >= 8'h20 && S_AXI_ARADDR[7:0] <= 8'h84) begin
                            axi_rdata <= reg_features[(S_AXI_ARADDR[7:0] - 8'h20) >> 2];
                        end else begin
                            axi_rdata <= 32'd0;
                        end
                    end
                endcase
            end else if (S_AXI_RREADY && axi_rvalid) begin
                axi_arready <= 1'b0;
                axi_rvalid  <= 1'b0;
            end
        end
    end

endmodule
