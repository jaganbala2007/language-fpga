/*
 * Vivado HLS Neural Network Kernel Header
 * FPGA-Based Multilingual AI Receptionist
 * Target: Xilinx PYNQ-Z2 (XC7Z020)
 */

#ifndef MLP_NN_HLS_H
#define MLP_NN_HLS_H

#include <stdint.h>
#include "model_weights_fixed.h"

// Interface payload structure for AXI4-Lite
typedef struct {
    int16_t features[NUM_FEATURES];
} input_features_t;

typedef struct {
    int32_t predicted_class; // 0: English, 1: Hindi, 2: Tamil
    int16_t score_english;
    int16_t score_hindi;
    int16_t score_tamil;
} output_prediction_t;

// Top-level synthesizable HLS function
void mlp_classifier_hls(
    const int16_t raw_features[NUM_FEATURES],
    output_prediction_t *result
);

#endif // MLP_NN_HLS_H
