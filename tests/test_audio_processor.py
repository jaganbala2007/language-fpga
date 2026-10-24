"""
Unit Tests: Audio Preprocessing and Signal Validation
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import (
    load_audio_signal,
    validate_audio_signal,
    extract_26_features,
    TARGET_SAMPLE_RATE
)


def test_validate_empty_audio():
    empty_signal = np.array([], dtype=np.float32)
    valid, msg = validate_audio_signal(empty_signal, sr=16000)
    assert not valid
    assert "empty" in msg.lower()


def test_validate_short_audio():
    # 0.05 seconds of audio (shorter than minimum 0.2s)
    short_signal = np.sin(np.linspace(0, 440 * 2 * np.pi * 0.05, 800)).astype(np.float32)
    valid, msg = validate_audio_signal(short_signal, sr=16000)
    assert not valid
    assert "too short" in msg.lower()


def test_validate_silent_audio():
    # 1 second of pure zeros
    silent_signal = np.zeros(16000, dtype=np.float32)
    valid, msg = validate_audio_signal(silent_signal, sr=16000)
    assert not valid
    assert "silent" in msg.lower()


def test_validate_valid_audio():
    # 1 second 440 Hz tone
    t = np.linspace(0, 1.0, 16000)
    valid_signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    valid, msg = validate_audio_signal(valid_signal, sr=16000)
    assert valid
    assert "Valid" in msg


def test_load_audio_from_numpy_array():
    audio_arr = np.random.uniform(-0.5, 0.5, 32000).astype(np.float32)
    wf, sr = load_audio_signal(audio_arr, sr=16000)
    assert len(wf) == 32000
    assert sr == 16000
