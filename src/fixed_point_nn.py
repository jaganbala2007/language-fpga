"""
Fixed-Point Quantization and Hardware Simulation Module
FPGA-Based Multilingual AI Receptionist

Converts the trained floating-point MLPClassifier (26 -> 16 -> 3) and StandardScaler
into 16-bit signed fixed-point (Q8.8 representation) for FPGA deployment.
Generates:
1. C header files for Vivado HLS
2. Verilog memory initialization files
3. Bit-accurate Python fixed-point emulator for verification
"""

import os
import sys
import numpy as np
import joblib
from typing import Dict, Tuple, List, Any, Union

# Fixed-point configuration: Q8.8 format (1 sign bit, 7 integer bits, 8 fractional bits = 16 bits)
# Fraction scaling factor: 2^8 = 256
FRACTION_BITS = 8
SCALE_FACTOR = 1 << FRACTION_BITS  # 256.0
MAX_VAL_16BIT = 32767
MIN_VAL_16BIT = -32768


def float_to_fixed16(val: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """Converts a float or numpy float array to 16-bit signed fixed-point integers (Q8.8)."""
    scaled = np.round(val * SCALE_FACTOR)
    clipped = np.clip(scaled, MIN_VAL_16BIT, MAX_VAL_16BIT)
    return clipped.astype(np.int16)


def fixed16_to_float(val: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts a 16-bit signed fixed-point integer (Q8.8) back to float."""
    return val.astype(np.float64) / float(SCALE_FACTOR)


class FixedPointMLPEmulator:
    """
    Bit-accurate Python emulator of the FPGA Verilog/HLS hardware datapath.
    Performs all operations in signed integer arithmetic matching the FPGA ALU.
    """

    def __init__(
        self,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        classes: List[str] = None
    ):
        self.classes = classes or ["English", "Hindi", "Tamil"]

        # Precompute scaler inv_scale: (x - mean) / scale = (x - mean) * (1 / scale)
        inv_scale = 1.0 / np.maximum(scaler_scale, 1e-7)

        # Quantize parameters to 16-bit integers
        self.mean_fixed = float_to_fixed16(scaler_mean)
        self.inv_scale_fixed = float_to_fixed16(inv_scale)
        self.w1_fixed = float_to_fixed16(w1)  # shape (26, 16)
        self.b1_fixed = float_to_fixed16(b1)  # shape (16,)
        self.w2_fixed = float_to_fixed16(w2)  # shape (16, 3)
        self.b2_fixed = float_to_fixed16(b2)  # shape (3,)

    @classmethod
    def from_artifacts(cls, model_path: str = "language_classifier.pkl", scaler_path: str = "feature_scaler.pkl"):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return cls(
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            w1=model.coefs_[0],
            b1=model.intercepts_[0],
            w2=model.coefs_[1],
            b2=model.intercepts_[1],
            classes=list(model.classes_)
        )

    def forward(self, raw_features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs hardware fixed-point pipeline on 26 raw MFCC features:
        1. Quantize raw input to Q8.8
        2. Feature Scaling: z[i] = ((x[i] - mean[i]) * inv_scale[i]) >> 8
        3. Layer 1 (Hidden 16 units): h[j] = ReLU(sum(z[i] * w1[i][j]) >> 8 + b1[j])
        4. Layer 2 (Output 3 units): out[k] = (sum(h[j] * w2[j][k]) >> 8) + b2[k]
        5. Softmax / argmax
        """
        # Step 1: Input conversion to fixed-point
        x_fixed = float_to_fixed16(raw_features).astype(np.int32)

        # Step 2: Hardware scaler unit
        # (x - mean) * inv_scale >> FRACTION_BITS
        diff = x_fixed - self.mean_fixed.astype(np.int32)
        z_fixed = (diff * self.inv_scale_fixed.astype(np.int32)) >> FRACTION_BITS
        z_fixed = np.clip(z_fixed, MIN_VAL_16BIT, MAX_VAL_16BIT)

        # Step 3: Layer 1 MAC + ReLU
        # Shape: (16,)
        hidden_fixed = np.zeros(16, dtype=np.int32)
        for j in range(16):
            acc = int(self.b1_fixed[j]) << FRACTION_BITS
            for i in range(26):
                acc += int(z_fixed[i]) * int(self.w1_fixed[i, j])
            acc_scaled = acc >> FRACTION_BITS
            # ReLU activation
            hidden_fixed[j] = max(0, acc_scaled)

        # Step 4: Layer 2 MAC
        # Shape: (3,)
        output_fixed = np.zeros(3, dtype=np.int32)
        for k in range(3):
            acc = int(self.b2_fixed[k]) << FRACTION_BITS
            for j in range(16):
                acc += int(hidden_fixed[j]) * int(self.w2_fixed[j, k])
            output_fixed[k] = acc >> FRACTION_BITS

        # Step 5: Convert logits to float for softmax probability
        logits_float = fixed16_to_float(output_fixed)
        
        # Numerically stable softmax
        exp_logits = np.exp(logits_float - np.max(logits_float))
        probs = exp_logits / np.sum(exp_logits)

        best_idx = int(np.argmax(output_fixed))
        predicted_class = self.classes[best_idx]
        confidence = float(probs[best_idx])
        scores = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}

        return predicted_class, confidence, scores


def export_c_header(emulator: FixedPointMLPEmulator, output_path: str = "fpga/hls/model_weights_fixed.h"):
    """Exports quantized model parameters to a C/C++ header for Vivado HLS."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("/* Auto-generated Fixed-Point Parameters for PYNQ-Z2 Vivado HLS */\n")
        f.write("#ifndef MODEL_WEIGHTS_FIXED_H\n#define MODEL_WEIGHTS_FIXED_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_FEATURES {len(emulator.mean_fixed)}\n")
        f.write(f"#define NUM_HIDDEN 16\n")
        f.write(f"#define NUM_CLASSES {len(emulator.classes)}\n")
        f.write(f"#define FIXED_SHIFT {FRACTION_BITS}\n\n")

        # Scaler parameters
        f.write("// Feature Scaler Means (Q8.8)\n")
        f.write("static const int16_t SCALER_MEAN[NUM_FEATURES] = {\n  " + ", ".join(map(str, emulator.mean_fixed)) + "\n};\n\n")
        
        f.write("// Feature Scaler Inverse Scales (Q8.8)\n")
        f.write("static const int16_t SCALER_INV_SCALE[NUM_FEATURES] = {\n  " + ", ".join(map(str, emulator.inv_scale_fixed)) + "\n};\n\n")

        # W1
        f.write("// Layer 1 Weights [26][16] (Q8.8)\nstatic const int16_t W1[NUM_FEATURES][NUM_HIDDEN] = {\n")
        for i in range(26):
            row_str = ", ".join(map(str, emulator.w1_fixed[i]))
            f.write(f"  {{ {row_str} }}{',' if i < 25 else ''}\n")
        f.write("};\n\n")

        # b1
        f.write("// Layer 1 Biases [16] (Q8.8)\nstatic const int16_t B1[NUM_HIDDEN] = {\n  " + ", ".join(map(str, emulator.b1_fixed)) + "\n};\n\n")

        # W2
        f.write("// Layer 2 Weights [16][3] (Q8.8)\nstatic const int16_t W2[NUM_HIDDEN][NUM_CLASSES] = {\n")
        for j in range(16):
            row_str = ", ".join(map(str, emulator.w2_fixed[j]))
            f.write(f"  {{ {row_str} }}{',' if j < 15 else ''}\n")
        f.write("};\n\n")

        # b2
        f.write("// Layer 2 Biases [3] (Q8.8)\nstatic const int16_t B2[NUM_CLASSES] = {\n  " + ", ".join(map(str, emulator.b2_fixed)) + "\n};\n\n")

        f.write("#endif // MODEL_WEIGHTS_FIXED_H\n")

    print(f"[OK] Exported C/HLS header to: {output_path}")


def export_hex_memories(emulator: FixedPointMLPEmulator, output_dir: str = "fpga/weights"):
    """Exports weights to Verilog hex files for $readmemh in Verilog testbenches."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export W1
    w1_path = os.path.join(output_dir, "weights_w1.hex")
    with open(w1_path, "w") as f:
        for i in range(26):
            for j in range(16):
                val = int(emulator.w1_fixed[i, j]) & 0xFFFF
                f.write(f"{val:04X}\n")

    # Export W2
    w2_path = os.path.join(output_dir, "weights_w2.hex")
    with open(w2_path, "w") as f:
        for j in range(16):
            for k in range(3):
                val = int(emulator.w2_fixed[j, k]) & 0xFFFF
                f.write(f"{val:04X}\n")

    print(f"[OK] Exported Verilog hex memory files to: {output_dir}")


def main():
    print("=" * 60)
    print("  QUANTIZING MODEL WEIGHTS FOR PYNQ-Z2 FPGA")
    print("=" * 60)

    model_path = "language_classifier.pkl" if os.path.exists("language_classifier.pkl") else "FPGA_Multilingual_Receptionist/language_classifier.pkl"
    scaler_path = "feature_scaler.pkl" if os.path.exists("feature_scaler.pkl") else "FPGA_Multilingual_Receptionist/feature_scaler.pkl"

    emulator = FixedPointMLPEmulator.from_artifacts(model_path, scaler_path)
    export_c_header(emulator, "fpga/hls/model_weights_fixed.h")
    export_hex_memories(emulator, "fpga/weights")

    # Test equivalence on sample features
    dummy_feat = np.ones(26, dtype=np.float64) * -20.0
    pred, conf, scores = emulator.forward(dummy_feat)
    print(f"Fixed-point test inference result: {pred} ({conf*100:.2f}%)")


if __name__ == "__main__":
    main()
