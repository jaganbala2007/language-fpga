"""
Unified Inference Backend Interface (CPU / FPGA / AUTO)
FPGA-Based Multilingual AI Receptionist

Provides a transparent backend execution abstraction:
- CPU: Pure software inference via Scikit-Learn MLPClassifier
- FPGA: Hardware-accelerated inference via PYNQ-Z2 MMIO / Overlay
- AUTO: Dynamic hardware probe with graceful, non-blocking CPU fallback
- SIMULATED_FPGA: Bit-accurate fixed-point integer arithmetic emulator
"""

import os
import sys
import time
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import joblib

from src.fixed_point_nn import FixedPointMLPEmulator, float_to_fixed16, fixed16_to_float, FRACTION_BITS

LANGUAGE_CLASSES = ["English", "Hindi", "Tamil"]

# PYNQ-Z2 Memory Map Constants (AXI4-Lite Slave Interface)
# Base Address typically assigned by Vivado address editor: 0x43C00000
DEFAULT_AXI_BASE_ADDR = 0x43C00000
REG_CTRL_OFFSET = 0x00      # Bit 0: ap_start, Bit 1: ap_done, Bit 2: ap_idle, Bit 3: ap_ready
REG_PRED_CLASS_OFFSET = 0x10  # 0: English, 1: Hindi, 2: Tamil
REG_SCORE_0_OFFSET = 0x14   # English Score (Q8.8)
REG_SCORE_1_OFFSET = 0x18   # Hindi Score (Q8.8)
REG_SCORE_2_OFFSET = 0x1C   # Tamil Score (Q8.8)
REG_FEATURE_BASE_OFFSET = 0x20  # 26 features x 4 bytes (0x20 to 0x84)


class BaseInferenceBackend(ABC):
    """Abstract base class for multilingual inference backends."""

    def __init__(self, name: str):
        self.name = name
        self.is_hardware = False
        self.status = "Initialized"
        self.last_latency_ms = 0.0
        self.diagnostics = {}

    @abstractmethod
    def predict(self, raw_features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Runs inference on 26 unscaled MFCC features.

        Returns:
            predicted_class (str): 'English', 'Hindi', or 'Tamil'
            confidence (float): Probability of the top class (0.0 to 1.0)
            class_scores (Dict[str, float]): Scores for all 3 classes
        """
        pass

    def is_ready(self) -> bool:
        """Returns True if the backend is initialized and ready for inference."""
        return self.status is not None and "Ready" in self.status or "Active" in self.status

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "backend_name": self.name,
            "is_hardware": self.is_hardware,
            "status": self.status,
            "last_latency_ms": round(self.last_latency_ms, 3),
            "details": self.diagnostics
        }



class CPUBackend(BaseInferenceBackend):
    """Scikit-Learn software inference backend."""

    def __init__(self, model_path: str = "language_classifier.pkl", scaler_path: str = "feature_scaler.pkl", model=None, scaler=None):
        super().__init__("CPU (Scikit-Learn MLP)")
        self.is_hardware = False
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = model or joblib.load(model_path)
            self.scaler = scaler or joblib.load(scaler_path)
            
        self.classes = list(getattr(self.model, "classes_", LANGUAGE_CLASSES))
        self.status = "Ready (CPU Software)"
        self.diagnostics = {
            "model_type": type(self.model).__name__,
            "hidden_units": 16,
            "input_features": 26,
            "classes": self.classes
        }

    def predict(self, raw_features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        t0 = time.perf_counter()
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled = self.scaler.transform(raw_features)
            pred = self.model.predict(scaled)[0]
            probs = self.model.predict_proba(scaled)[0]

        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        scores = {str(c): float(p) for c, p in zip(self.classes, probs)}
        conf = float(scores.get(str(pred), max(probs)))
        return str(pred), conf, scores


class SimulatedFPGABackend(BaseInferenceBackend):
    """Bit-accurate Fixed-Point integer arithmetic backend matching FPGA ALU."""

    def __init__(self, model_path: str = "language_classifier.pkl", scaler_path: str = "feature_scaler.pkl"):
        super().__init__("Simulated FPGA (Q8.8 Fixed-Point ALU)")
        self.is_hardware = False
        self.emulator = FixedPointMLPEmulator.from_artifacts(model_path, scaler_path)
        self.status = "Ready (Bit-Accurate Fixed-Point Emulation)"
        self.diagnostics = {
            "arithmetic": "Signed 16-bit Fixed-Point (Q8.8)",
            "pipeline_cycles": 45,
            "target_clock_mhz": 100.0,
            "simulated_hw_latency_us": 0.45
        }

    def predict(self, raw_features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        t0 = time.perf_counter()
        pred, conf, scores = self.emulator.forward(raw_features)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return pred, conf, scores


class FPGABackend(BaseInferenceBackend):
    """
    Physical PYNQ-Z2 FPGA MMIO Hardware Backend.
    Interacts with the synthesizable MLP hardware IP through AXI4-Lite registers.
    """

    def __init__(self, bitstream_path: str = "fpga/bitstream/mlp_classifier.bit", base_addr: int = DEFAULT_AXI_BASE_ADDR):
        super().__init__("PYNQ-Z2 FPGA (Hardware AXI MMIO)")
        self.is_hardware = True
        self.bitstream_path = bitstream_path
        self.base_addr = base_addr
        self.overlay = None
        self.mmio = None
        self.classes = LANGUAGE_CLASSES
        self._init_hardware()

    def _init_hardware(self):
        """Initializes PYNQ overlay and MMIO interface."""
        try:
            from pynq import Overlay, MMIO  # type: ignore
        except ImportError:
            self.status = "Failed: 'pynq' library not found (running on host OS instead of PYNQ Linux)"
            raise RuntimeError(self.status)

        if not os.path.exists(self.bitstream_path):
            self.status = f"Failed: Bitstream not found at '{self.bitstream_path}'"
            raise FileNotFoundError(self.status)

        try:
            self.overlay = Overlay(self.bitstream_path)
            self.mmio = MMIO(self.base_addr, 0x1000)
            self.status = "Ready (FPGA Hardware Connected)"
            self.diagnostics = {
                "target_device": "XC7Z020-1CLG400C",
                "bitstream": self.bitstream_path,
                "base_address": hex(self.base_addr),
                "clock_freq": "100 MHz",
                "bus_interface": "AXI4-Lite"
            }
        except Exception as e:
            self.status = f"FPGA Initialization Error: {e}"
            raise

    def predict(self, raw_features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if self.mmio is None:
            raise RuntimeError(f"FPGA hardware is not initialized. Status: {self.status}")

        t0 = time.perf_counter()

        # 1. Convert 26 features to Q8.8 fixed-point integers
        features_fixed = float_to_fixed16(raw_features).astype(np.int32)

        # 2. Write features to AXI-Lite registers (0x20 + i*4)
        for i in range(26):
            # Write 32-bit register (lower 16 bits active)
            self.mmio.write(REG_FEATURE_BASE_OFFSET + i * 4, int(features_fixed[i]) & 0xFFFFFFFF)

        # 3. Assert ap_start (Bit 0 of REG_CTRL)
        self.mmio.write(REG_CTRL_OFFSET, 0x01)

        # 4. Poll ap_done (Bit 1) with timeout
        timeout = 0.05  # 50 ms timeout (hardware takes < 1 us)
        poll_t0 = time.perf_counter()
        while True:
            status = self.mmio.read(REG_CTRL_OFFSET)
            if status & 0x02:  # Done bit set
                break
            if time.perf_counter() - poll_t0 > timeout:
                raise TimeoutError("FPGA Hardware inference timed out waiting for ap_done.")

        # 5. Read predicted class index and scores
        pred_idx = self.mmio.read(REG_PRED_CLASS_OFFSET) % 3
        raw_score_0 = np.int16(self.mmio.read(REG_SCORE_0_OFFSET) & 0xFFFF)
        raw_score_1 = np.int16(self.mmio.read(REG_SCORE_1_OFFSET) & 0xFFFF)
        raw_score_2 = np.int16(self.mmio.read(REG_SCORE_2_OFFSET) & 0xFFFF)

        logits = np.array([raw_score_0, raw_score_1, raw_score_2], dtype=np.float32) / 256.0
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        scores = {self.classes[i]: float(probs[i]) for i in range(3)}

        return pred_class, confidence, scores


class AutoInferenceBackend(BaseInferenceBackend):
    """
    Intelligent Auto Backend.
    Attempts hardware FPGA initialization, and automatically falls back
    to CPU software execution if FPGA hardware or PYNQ environment is absent.
    """

    def __init__(
        self,
        model_path: str = "language_classifier.pkl",
        scaler_path: str = "feature_scaler.pkl",
        bitstream_path: str = "fpga/bitstream/mlp_classifier.bit",
        model=None,
        scaler=None
    ):
        super().__init__("AUTO (Hardware Probe with CPU Fallback)")
        self.active_backend: BaseInferenceBackend
        self.fallback_reason: Optional[str] = None
        self.fpga_attempted = True

        try:
            # Check if running on PYNQ board and bitstream exists
            if os.path.exists(bitstream_path) and os.path.exists("/dev/mem"):
                self.active_backend = FPGABackend(bitstream_path=bitstream_path)
                self.is_hardware = True
                self.status = "Active on PYNQ-Z2 FPGA Hardware"
            else:
                raise RuntimeError("Host environment is not a physical PYNQ board or bitstream not present.")
        except Exception as e:
            self.fallback_reason = str(e)
            self.active_backend = CPUBackend(model_path=model_path, scaler_path=scaler_path, model=model, scaler=scaler)
            self.is_hardware = False
            self.status = f"Active on CPU (Fallback Reason: {self.fallback_reason})"

        self.diagnostics = {
            "active_backend": getattr(self.active_backend, "name", "CPU"),
            "fallback_reason": self.fallback_reason,
            "hardware_present": self.is_hardware
        }

    def is_ready(self) -> bool:
        """Returns True if the underlying active backend is ready."""
        if hasattr(self, "active_backend") and hasattr(self.active_backend, "is_ready"):
            return self.active_backend.is_ready()
        return True

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "backend_name": self.name,
            "is_hardware": self.is_hardware,
            "status": self.status,
            "last_latency_ms": round(self.last_latency_ms, 3),
            "details": {
                "active_backend": getattr(self.active_backend, "name", "CPU"),
                "fallback_reason": self.fallback_reason,
                "hardware_present": self.is_hardware
            }
        }


    def predict(self, raw_features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        t0 = time.perf_counter()
        pred, conf, scores = self.active_backend.predict(raw_features)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return pred, conf, scores



def get_backend(
    mode: str = "auto",
    model_path: str = "language_classifier.pkl",
    scaler_path: str = "feature_scaler.pkl",
    bitstream_path: str = "fpga/bitstream/mlp_classifier.bit",
    model: Any = None,
    scaler: Any = None
) -> BaseInferenceBackend:
    """Factory helper to instantiate backend by mode name: 'cpu', 'fpga', 'simulated_fpga', 'auto'."""
    mode_lower = mode.strip().lower()
    if mode_lower == "cpu":
        return CPUBackend(model_path=model_path, scaler_path=scaler_path, model=model, scaler=scaler)
    elif mode_lower == "fpga":
        return FPGABackend(bitstream_path=bitstream_path)
    elif mode_lower in ["simulated_fpga", "simulated", "fixed_point"]:
        return SimulatedFPGABackend(model_path=model_path, scaler_path=scaler_path)
    elif mode_lower == "auto":
        return AutoInferenceBackend(model_path=model_path, scaler_path=scaler_path, bitstream_path=bitstream_path, model=model, scaler=scaler)
    else:
        raise ValueError(f"Unknown backend mode: '{mode}'. Supported: 'cpu', 'fpga', 'simulated_fpga', 'auto'")
