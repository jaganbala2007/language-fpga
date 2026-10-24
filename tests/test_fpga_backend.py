"""
Unit Tests: Inference Backend Drivers (CPU, FPGA, Simulated, AUTO)
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fpga_backend import (
    CPUBackend,
    SimulatedFPGABackend,
    AutoInferenceBackend,
    get_backend,
    LANGUAGE_CLASSES
)


def test_cpu_backend():
    backend = get_backend("cpu")
    assert not backend.is_hardware
    features = np.zeros(26, dtype=np.float64)
    pred, conf, scores = backend.predict(features)
    assert pred in LANGUAGE_CLASSES
    assert 0.0 <= conf <= 1.0


def test_simulated_fpga_backend():
    backend = get_backend("simulated_fpga")
    features = np.ones(26, dtype=np.float64) * 2.0
    pred, conf, scores = backend.predict(features)
    assert pred in LANGUAGE_CLASSES
    assert 0.0 <= conf <= 1.0


def test_auto_backend_fallback():
    backend = get_backend("auto")
    # On regular OS without physical PYNQ board, it should safely fall back to CPU
    diag = backend.get_diagnostics()
    assert "status" in diag
    features = np.zeros(26, dtype=np.float64)
    pred, conf, scores = backend.predict(features)
    assert pred in LANGUAGE_CLASSES
