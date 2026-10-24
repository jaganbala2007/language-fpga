"""
Language Prediction CLI & Inference Tool
FPGA-Based Multilingual AI Receptionist

Usage:
    python src/predict_language.py <path_to_audio> [--model <model.pkl>] [--scaler <scaler.pkl>] [--backend {cpu,fpga,auto}]
"""

import os
import sys
import argparse
import warnings
import joblib
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import (
    process_and_classify_audio,
    load_audio_signal,
    extract_26_features,
    LANGUAGE_CLASSES
)


def load_artifacts(model_path: str = "language_classifier.pkl", scaler_path: str = "feature_scaler.pkl"):
    """Finds and loads the model and scaler from either current dir or parent."""
    # Check current dir, then FPGA_Multilingual_Receptionist/
    candidates_model = [
        model_path,
        os.path.join("FPGA_Multilingual_Receptionist", model_path),
        os.path.join(os.path.dirname(__file__), "..", model_path)
    ]
    candidates_scaler = [
        scaler_path,
        os.path.join("FPGA_Multilingual_Receptionist", scaler_path),
        os.path.join(os.path.dirname(__file__), "..", scaler_path)
    ]

    resolved_model = None
    for p in candidates_model:
        if os.path.exists(p):
            resolved_model = p
            break

    resolved_scaler = None
    for p in candidates_scaler:
        if os.path.exists(p):
            resolved_scaler = p
            break

    if not resolved_model:
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not resolved_scaler:
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(resolved_model)
        scaler = joblib.load(resolved_scaler)

    return model, scaler, resolved_model, resolved_scaler


def main():
    parser = argparse.ArgumentParser(description="Classify spoken language (English, Hindi, Tamil) from audio.")
    parser.add_argument("audio_file", type=str, help="Path to input audio file (WAV, M4A, MP3, etc.)")
    parser.add_argument("--model", type=str, default="language_classifier.pkl", help="Path to trained model pickle")
    parser.add_argument("--scaler", type=str, default="feature_scaler.pkl", help="Path to feature scaler pickle")
    parser.add_argument("--backend", type=str, choices=["cpu", "fpga", "auto"], default="auto", help="Inference backend")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Input audio file '{args.audio_file}' not found.")
        sys.exit(1)

    try:
        model, scaler, m_path, s_path = load_artifacts(args.model, args.scaler)
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        sys.exit(1)

    # Use FPGA backend if requested
    if args.backend in ["fpga", "auto"]:
        try:
            from src.fpga_backend import get_backend
            backend = get_backend(mode=args.backend, model=model, scaler=scaler)
            waveform, sr = load_audio_signal(args.audio_file)
            features = extract_26_features(waveform, sr=sr)
            pred, conf, scores = backend.predict(features)
            
            print(f"Inference Backend: {backend.name} (Status: {backend.status})")
            print(f"Audio File: {args.audio_file}")
            print(f"Predicted Language: {pred}")
            print(f"Confidence: {conf * 100:.2f}%\n")
            print("Class Probabilities / Scores:")
            for lang, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {lang:<10}: {score * 100:.2f}%")
            return
        except Exception as e:
            if args.backend == "fpga":
                print(f"FPGA Inference Error: {e}")
                sys.exit(1)
            # Auto falls back to CPU

    # Standard CPU path
    res = process_and_classify_audio(args.audio_file, model, scaler)
    if not res["success"]:
        print(f"Processing Error: {res['error']}")
        sys.exit(1)

    print(f"Inference Backend: CPU (Scikit-Learn MLP)")
    print(f"Audio File: {args.audio_file}")
    print(f"Duration: {res['duration']:.2f}s")
    print(f"Predicted Language: {res['predicted_language']}")
    print(f"Confidence: {res['confidence'] * 100:.2f}%\n")
    print("Class Probabilities:")
    for lang, prob in sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang:<10}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
