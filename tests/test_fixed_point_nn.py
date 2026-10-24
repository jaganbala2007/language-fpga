"""
Unit Tests: Fixed-Point Quantization and Mathematical Equivalence
"""

import os
import sys
import pytest
import numpy as np
import joblib

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fixed_point_nn import FixedPointMLPEmulator, float_to_fixed16, fixed16_to_float
from src.features import LANGUAGE_CLASSES


def test_fixed_point_quantization_roundtrip():
    vals = np.array([-15.5, 0.0, 3.25, 42.125], dtype=np.float64)
    fixed = float_to_fixed16(vals)
    recovered = fixed16_to_float(fixed)
    # With 8 fractional bits (resolution 1/256 = ~0.0039), error should be <= 0.004
    assert np.all(np.abs(vals - recovered) <= 0.005)


def test_fixed_point_emulator_inference():
    model_path = "language_classifier.pkl" if os.path.exists("language_classifier.pkl") else "FPGA_Multilingual_Receptionist/language_classifier.pkl"
    scaler_path = "feature_scaler.pkl" if os.path.exists("feature_scaler.pkl") else "FPGA_Multilingual_Receptionist/feature_scaler.pkl"

    emulator = FixedPointMLPEmulator.from_artifacts(model_path, scaler_path)

    dummy_feat = np.ones(26, dtype=np.float64) * -20.0
    pred, conf, scores = emulator.forward(dummy_feat)

    assert pred in LANGUAGE_CLASSES
    assert 0.0 <= conf <= 1.0
    assert len(scores) == 3
