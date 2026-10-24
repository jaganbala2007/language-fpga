/*
 * Synthesizable Vivado HLS Multilingual MLP Neural Network Core
 * FPGA-Based Multilingual AI Receptionist
 * Target: Xilinx PYNQ-Z2 (Zynq-7000 XC7Z020-1CLG400C)
 */

#include "mlp_nn_hls.h"

void mlp_classifier_hls(
    const int16_t raw_features[NUM_FEATURES],
    output_prediction_t *result
) {
    // AXI4-Lite Slave Pragma Configuration for PYNQ MMIO
    #pragma HLS INTERFACE s_axilite port=raw_features bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=result       bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=return       bundle=CTRL_BUS

    #pragma HLS PIPELINE II=1

    // 1. Fixed-Point Feature Standardization
    int16_t scaled[NUM_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=scaled complete dim=1

    Scale_Loop: for (int i = 0; i < NUM_FEATURES; i++) {
        #pragma HLS UNROLL
        int32_t diff = (int32_t)raw_features[i] - (int32_t)SCALER_MEAN[i];
        int32_t prod = (diff * (int32_t)SCALER_INV_SCALE[i]) >> FIXED_SHIFT;
        if (prod > 32767) prod = 32767;
        if (prod < -32768) prod = -32768;
        scaled[i] = (int16_t)prod;
    }

    // 2. Hidden Layer 1 (26 inputs -> 16 hidden neurons with ReLU)
    int16_t hidden[NUM_HIDDEN];
    #pragma HLS ARRAY_PARTITION variable=hidden complete dim=1

    Hidden_Neuron_Loop: for (int j = 0; j < NUM_HIDDEN; j++) {
        #pragma HLS UNROLL
        int32_t acc = ((int32_t)B1[j]) << FIXED_SHIFT;

        Hidden_Dot_Loop: for (int i = 0; i < NUM_FEATURES; i++) {
            #pragma HLS UNROLL
            acc += (int32_t)scaled[i] * (int32_t)W1[i][j];
        }

        int32_t acc_scaled = acc >> FIXED_SHIFT;
        // ReLU Activation
        hidden[j] = (acc_scaled > 0) ? (int16_t)acc_scaled : (int16_t)0;
    }

    // 3. Output Layer (16 hidden -> 3 classes)
    int16_t logits[NUM_CLASSES];
    #pragma HLS ARRAY_PARTITION variable=logits complete dim=1

    Output_Neuron_Loop: for (int k = 0; k < NUM_CLASSES; k++) {
        #pragma HLS UNROLL
        int32_t acc = ((int32_t)B2[k]) << FIXED_SHIFT;

        Output_Dot_Loop: for (int j = 0; j < NUM_HIDDEN; j++) {
            #pragma HLS UNROLL
            acc += (int32_t)hidden[j] * (int32_t)W2[j][k];
        }

        logits[k] = (int16_t)(acc >> FIXED_SHIFT);
    }

    // 4. Argmax Decision Logic
    int32_t best_class = 0;
    if (logits[1] > logits[0] && logits[1] >= logits[2]) {
        best_class = 1; // Hindi
    } else if (logits[2] > logits[0] && logits[2] > logits[1]) {
        best_class = 2; // Tamil
    } else {
        best_class = 0; // English
    }

    // Output assignment to AXI-Lite registers
    result->predicted_class = best_class;
    result->score_english   = logits[0];
    result->score_hindi     = logits[1];
    result->score_tamil     = logits[2];
}
