"""
Reproducible Dataset Preparation and Feature Extraction
FPGA-Based Multilingual AI Receptionist

Extracts canonical 26 MFCC features from raw audio folders or parquet archives:
- Audio Directory Mode: looks in dataset/english, dataset/hindi, dataset/tamil
- Parquet Archive Mode: processes HuggingFace / CommonVoice parquet files
"""

import os
import sys
import io
import argparse
import glob
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import soundfile as sf
import librosa

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import (
    load_audio_signal,
    extract_26_features,
    validate_audio_signal,
    TARGET_SAMPLE_RATE,
    EXPECTED_FEATURE_COUNT
)


def extract_features_from_audio_folders(
    base_dir: str = "dataset",
    output_csv: str = "language_features.csv"
) -> Optional[pd.DataFrame]:
    """
    Extracts 26 features from folder structure:
    base_dir/
      english/*.wav, *.mp3, *.m4a
      hindi/*.wav, *.mp3, *.m4a
      tamil/*.wav, *.mp3, *.m4a
    """
    languages = ["English", "Hindi", "Tamil"]
    all_rows = []

    print(f"[*] Scanning audio directory: {base_dir}")
    total_found = 0

    for lang in languages:
        lang_dir = os.path.join(base_dir, lang.lower())
        if not os.path.exists(lang_dir):
            lang_dir = os.path.join(base_dir, lang)

        if not os.path.exists(lang_dir):
            print(f"[-] Subfolder for {lang} not found at {lang_dir}")
            continue

        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac"]:
            audio_files.extend(glob.glob(os.path.join(lang_dir, ext)))
            audio_files.extend(glob.glob(os.path.join(lang_dir, ext.upper())))

        print(f"[*] Found {len(audio_files)} files for {lang}")
        total_found += len(audio_files)

        for i, fpath in enumerate(audio_files):
            try:
                waveform, sr = load_audio_signal(fpath, sr=TARGET_SAMPLE_RATE)
                is_valid, msg = validate_audio_signal(waveform, sr)
                if not is_valid:
                    continue
                features = extract_26_features(waveform, sr=sr)
                row = list(features) + [lang]
                all_rows.append(row)
            except Exception as e:
                print(f"[!] Warning: Failed to process {fpath}: {e}")

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(audio_files)} files for {lang}")

    if not all_rows:
        print("[-] No audio files found in audio folders.")
        return None

    cols = [f"feature_{i+1}" for i in range(EXPECTED_FEATURE_COUNT)] + ["language"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Extracted {len(df)} samples into {output_csv}")
    print(df["language"].value_counts())
    return df


def extract_features_from_parquets(
    file_map: Dict[str, str],
    output_csv: str = "language_features.csv"
) -> pd.DataFrame:
    """Extracts features from CommonVoice / HuggingFace parquet dataset files."""
    all_rows = []
    for language, file_path in file_map.items():
        expanded = os.path.expanduser(file_path)
        if not os.path.exists(expanded):
            print(f"[!] Parquet file for {language} not found: {expanded}")
            continue

        print(f"[*] Processing {language} from {expanded}...")
        df_parquet = pd.read_parquet(expanded, columns=["audio"])

        for i, audio in enumerate(df_parquet["audio"]):
            try:
                raw_bytes = audio.get("bytes") if isinstance(audio, dict) else audio
                waveform, sr = load_audio_signal(raw_bytes, sr=TARGET_SAMPLE_RATE)
                is_valid, _ = validate_audio_signal(waveform, sr)
                if not is_valid:
                    continue
                features = extract_26_features(waveform, sr=sr)
                all_rows.append(list(features) + [language])
            except Exception as e:
                pass

            if (i + 1) % 500 == 0:
                print(f"    Processed {i + 1} samples for {language}")

    if not all_rows:
        raise ValueError("No samples were successfully extracted from parquet files.")

    cols = [f"feature_{i+1}" for i in range(EXPECTED_FEATURE_COUNT)] + ["language"]
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Dataset creation complete! Saved {len(df)} samples to {output_csv}")
    print(df["language"].value_counts())
    return df


def main():
    parser = argparse.ArgumentParser(description="Create feature dataset from audio directory or parquet archives.")
    parser.add_argument("--audio-dir", type=str, default="dataset", help="Path to base dataset folder with english/hindi/tamil subfolders")
    parser.add_argument("--output", type=str, default="language_features.csv", help="Output CSV path")
    parser.add_argument("--english-parquet", type=str, default=None, help="Optional English parquet file path")
    parser.add_argument("--hindi-parquet", type=str, default=None, help="Optional Hindi parquet file path")
    parser.add_argument("--tamil-parquet", type=str, default=None, help="Optional Tamil parquet file path")
    args = parser.parse_args()

    if args.english_parquet or args.hindi_parquet or args.tamil_parquet:
        file_map = {}
        if args.english_parquet: file_map["English"] = args.english_parquet
        if args.hindi_parquet: file_map["Hindi"] = args.hindi_parquet
        if args.tamil_parquet: file_map["Tamil"] = args.tamil_parquet
        extract_features_from_parquets(file_map, output_csv=args.output)
    else:
        df = extract_features_from_audio_folders(base_dir=args.audio_dir, output_csv=args.output)
        if df is None:
            print("[*] Note: If raw audio folders are empty, use existing 'language_features.csv' or supply parquet paths.")


if __name__ == "__main__":
    main()
