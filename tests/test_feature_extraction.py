"""
Unit Tests: 26-Feature MFCC Extraction Pipeline
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import (
    extract_26_features,
    EXPECTED_FEATURE_COUNT,
    TARGET_SAMPLE_RATE
)


def test_extract_26_features_shape_and_finite():
    # 1.5 seconds synthetic voice-like signal (harmonics)
    t = np.linspace(0, 1.5, int(1.5 * TARGET_SAMPLE_RATE))
    synth_signal = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 800 * t)
    ).astype(np.float32)

    features = extract_26_features(synth_signal, sr=TARGET_SAMPLE_RATE)
    assert features.shape == (EXPECTED_FEATURE_COUNT,)
    assert np.all(np.isfinite(features))
    assert features.dtype == np.float64


def test_extract_features_invalid_signal_raises():
    with pytest.raises(ValueError):
        extract_26_features(np.zeros(100, dtype=np.float32), sr=TARGET_SAMPLE_RATE)
