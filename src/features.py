"""
Shared Audio Processing and Feature Extraction Module
FPGA-Based Multilingual AI Receptionist

This module implements the canonical 10-step audio preprocessing
and 26-feature MFCC extraction pipeline shared across training,
evaluation, inference, and the real-time receptionist application.
"""

import os
import io
import tempfile
import subprocess
import warnings
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import librosa
import soundfile as sf
import joblib

# Target parameters strictly compatible with the trained model
TARGET_SAMPLE_RATE = 16000
N_MFCC = 13
EXPECTED_FEATURE_COUNT = 26
MIN_AUDIO_DURATION_SEC = 0.2
MIN_RMS_ENERGY = 1e-5

LANGUAGE_CLASSES = ["English", "Hindi", "Tamil"]


def get_ffmpeg_executable() -> Optional[str]:
    """Locate the ffmpeg binary from imageio_ffmpeg or system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def convert_audio_to_wav(input_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> str:
    """
    Converts any audio file (M4A, MP3, AAC, OGG, FLAC, WebM, etc.)
    into a temporary 16kHz mono WAV file using FFmpeg.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    ffmpeg_exe = get_ffmpeg_executable()
    if not ffmpeg_exe:
        raise RuntimeError("FFmpeg is required for multi-format audio conversion but was not found.")

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_path,
        "-ar", str(target_sr),
        "-ac", "1",
        "-f", "wav",
        temp_wav.name
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        if os.path.exists(temp_wav.name):
            os.remove(temp_wav.name)
        raise RuntimeError(f"FFmpeg audio conversion failed: {result.stderr.decode('utf-8', errors='ignore')}")

    return temp_wav.name


def load_audio_signal(
    audio_source: Union[str, bytes, np.ndarray],
    sr: int = TARGET_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """
    Loads audio from a file path, raw bytes, or existing numpy array,
    converts to mono, and resamples to target sample rate (default 16 kHz).

    Returns:
        waveform: 1D numpy float32 array normalized to [-1.0, 1.0].
        sample_rate: sampling rate (16000 Hz).
    """
    if isinstance(audio_source, np.ndarray):
        # Audio is already an array
        waveform = audio_source.astype(np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0 if waveform.shape[0] < waveform.shape[1] else 1)
        return waveform, sr

    if isinstance(audio_source, bytes):
        # Bytes stream
        try:
            waveform, original_sr = sf.read(io.BytesIO(audio_source))
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            if original_sr != sr:
                waveform = librosa.resample(waveform.astype(np.float32), orig_sr=original_sr, target_sr=sr)
            return waveform.astype(np.float32), sr
        except Exception:
            # Fallback to temp file conversion
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
                f.write(audio_source)
                tmp_path = f.name
            try:
                wav_path = convert_audio_to_wav(tmp_path, target_sr=sr)
                waveform, original_sr = sf.read(wav_path)
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                return waveform.astype(np.float32), sr
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if isinstance(audio_source, str):
        if not os.path.exists(audio_source):
            raise FileNotFoundError(f"Audio file does not exist: {audio_source}")

        # Try direct soundfile loading first
        try:
            waveform, original_sr = sf.read(audio_source)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            if original_sr != sr:
                waveform = librosa.resample(waveform.astype(np.float32), orig_sr=original_sr, target_sr=sr)
            return waveform.astype(np.float32), sr
        except Exception:
            # Fall back to FFmpeg conversion (handles m4a, mp3, etc.)
            wav_path = convert_audio_to_wav(audio_source, target_sr=sr)
            try:
                waveform, original_sr = sf.read(wav_path)
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)
                return waveform.astype(np.float32), sr
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)

    raise ValueError(f"Unsupported audio source type: {type(audio_source)}")


def validate_audio_signal(waveform: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> Tuple[bool, str]:
    """
    Validates that the audio signal is non-empty, finite, meets minimum duration,
    and has sufficient energy (not silent/noise floor).
    """
    if waveform is None or len(waveform) == 0:
        return False, "Audio waveform is empty."

    if not np.all(np.isfinite(waveform)):
        return False, "Audio waveform contains NaN or Inf values."

    duration = len(waveform) / float(sr)
    if duration < MIN_AUDIO_DURATION_SEC:
        return False, f"Audio is too short ({duration:.2f}s, minimum required is {MIN_AUDIO_DURATION_SEC}s)."

    # Calculate Root Mean Square (RMS) energy
    rms = np.sqrt(np.mean(np.square(waveform)))
    if rms < MIN_RMS_ENERGY:
        return False, f"Audio signal is near-silent or empty (RMS energy: {rms:.6f})."

    return True, "Valid audio signal."


def extract_26_features(
    waveform: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mfcc: int = N_MFCC
) -> np.ndarray:
    """
    Extracts canonical 26 features:
    - 13 MFCC time-averaged means
    - 13 MFCC standard deviations along time axis

    Returns:
        1D numpy float64 array of length exactly 26.
    """
    is_valid, msg = validate_audio_signal(waveform, sr)
    if not is_valid:
        raise ValueError(f"Invalid audio input: {msg}")

    # Compute 13 MFCCs
    # Uses standard librosa parameters matching the original dataset extraction
    mfcc = librosa.feature.mfcc(
        y=waveform.astype(np.float32),
        sr=sr,
        n_mfcc=n_mfcc
    )

    # Calculate mean and standard deviation along time axis (axis 1)
    mean_features = np.mean(mfcc, axis=1)
    std_features = np.std(mfcc, axis=1)

    # Concatenate into exact 26-feature vector
    features = np.concatenate([mean_features, std_features]).astype(np.float64)

    if features.shape != (EXPECTED_FEATURE_COUNT,):
        raise ValueError(f"Feature shape mismatch: expected ({EXPECTED_FEATURE_COUNT},), got {features.shape}")

    return features


def extract_features_from_file(audio_path: str) -> np.ndarray:
    """Convenience helper to extract 26 features directly from an audio file path."""
    waveform, sr = load_audio_signal(audio_path, sr=TARGET_SAMPLE_RATE)
    return extract_26_features(waveform, sr=sr)


def predict_language_from_features(
    features: np.ndarray,
    model: Any,
    scaler: Any
) -> Tuple[str, float, Dict[str, float]]:
    """
    Standardizes a 26-feature vector and predicts the language using the trained model.

    Returns:
        predicted_language (str): 'English', 'Hindi', or 'Tamil'
        confidence (float): Probability of the top class (0.0 to 1.0)
        class_probabilities (Dict[str, float]): Scores for all 3 classes
    """
    if features.ndim == 1:
        features_2d = features.reshape(1, -1)
    else:
        features_2d = features

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled_features = scaler.transform(features_2d)
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

    classes = getattr(model, "classes_", LANGUAGE_CLASSES)
    scores = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
    confidence = float(scores.get(str(prediction), max(probabilities)))

    return str(prediction), confidence, scores


def process_and_classify_audio(
    audio_source: Union[str, bytes, np.ndarray],
    model: Any,
    scaler: Any
) -> Dict[str, Any]:
    """
    Complete 10-step audio pipeline:
    1. Load audio
    2. Convert to mono
    3. Resample to 16 kHz
    4. Validate signal
    5. Extract 13 MFCCs
    6. Compute mean & std along time axis
    7. Form 26-element vector
    8. Apply StandardScaler
    9. Run MLP classifier
    10. Return predicted language, confidence, and scores
    """
    waveform, sr = load_audio_signal(audio_source, sr=TARGET_SAMPLE_RATE)
    is_valid, msg = validate_audio_signal(waveform, sr)
    if not is_valid:
        return {
            "success": False,
            "error": msg,
            "predicted_language": "Unknown",
            "confidence": 0.0,
            "probabilities": {cls: 0.0 for cls in LANGUAGE_CLASSES},
            "features": None,
            "duration": len(waveform) / sr if waveform is not None else 0.0
        }

    features = extract_26_features(waveform, sr=sr)
    lang, conf, scores = predict_language_from_features(features, model, scaler)

    return {
        "success": True,
        "error": None,
        "predicted_language": lang,
        "confidence": conf,
        "probabilities": scores,
        "features": features,
        "waveform": waveform,
        "duration": len(waveform) / sr
    }
