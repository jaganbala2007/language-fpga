"""
Single-Audio Feature Extraction Inspector
FPGA-Based Multilingual AI Receptionist

Inspects audio files and displays:
- Waveform duration and sample rate
- MFCC matrix dimensions
- 26-feature fixed representation (13 Means + 13 Standard Deviations)
"""

import os
import sys
import argparse
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import load_audio_signal, extract_26_features, validate_audio_signal, TARGET_SAMPLE_RATE


def inspect_audio_file(audio_path: str):
    print("=" * 60)
    print(f"  FEATURE EXTRACTION INSPECTION: {os.path.basename(audio_path)}")
    print("=" * 60)

    if not os.path.exists(audio_path):
        print(f"[!] Error: File '{audio_path}' does not exist.")
        sys.exit(1)

    waveform, sr = load_audio_signal(audio_path, sr=TARGET_SAMPLE_RATE)
    is_valid, msg = validate_audio_signal(waveform, sr)

    print(f"[*] Input Path       : {audio_path}")
    print(f"[*] Sample Rate      : {sr} Hz")
    print(f"[*] Total Samples    : {len(waveform)}")
    print(f"[*] Duration         : {len(waveform)/sr:.2f} seconds")
    print(f"[*] Signal Valid     : {is_valid} ({msg})")
    print(f"[*] Waveform Range   : [{waveform.min():.4f}, {waveform.max():.4f}]")
    print(f"[*] RMS Energy       : {np.sqrt(np.mean(np.square(waveform))):.6f}")

    if not is_valid:
        print("[!] Audio validation failed. Cannot extract features.")
        return

    features = extract_26_features(waveform, sr=sr)

    print(f"\n[*] Extracted Feature Vector: Shape {features.shape}")
    print("\n--- 13 MFCC Mean Coefficients ---")
    for i in range(13):
        print(f"  MFCC Mean [{i+1:2d}] : {features[i]:>10.4f}")

    print("\n--- 13 MFCC Std Dev Coefficients ---")
    for i in range(13):
        print(f"  MFCC Std  [{i+1:2d}] : {features[13+i]:>10.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Extract and inspect 26 MFCC features from an audio file.")
    parser.add_argument("audio_file", type=str, nargs="?", default="Recording.wav", help="Audio file path (default: Recording.wav)")
    args = parser.parse_args()

    inspect_audio_file(args.audio_file)


if __name__ == "__main__":
    main()
