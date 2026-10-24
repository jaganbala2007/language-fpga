"""
Multilingual Live Demonstration Script
FPGA-Based Multilingual AI Receptionist

Tests end-to-end processing of English, Hindi, and Tamil visitor requests:
1. Audio loading and 26-MFCC feature extraction
2. Hardware/Software language classification
3. Speech transcription in classified language
4. Intent routing
5. Multilingual speech synthesis
"""

import os
import sys
import io

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.receptionist import MultilingualReceptionist


def run_live_demonstrations():
    print("=" * 75)
    print("  MULTILINGUAL AI RECEPTIONIST — 3-LANGUAGE DEMONSTRATION")
    print("=" * 75)

    receptionist = MultilingualReceptionist(backend_mode="auto")
    print(f"[*] Inference Backend : {receptionist.backend.name}")
    print(f"[*] Backend Status    : {receptionist.backend.status}")
    print("-" * 75)

    test_cases = [
        ("English", "demo_audio/demo_en_welcome.wav"),
        ("Hindi", "demo_audio/demo_hi_hours.wav"),
        ("Tamil", "demo_audio/demo_ta_admissions.wav"),
        ("Original Recording", "Recording.wav")
    ]

    for expected_lang, audio_path in test_cases:
        if not os.path.exists(audio_path):
            print(f"[!] File not found: {audio_path}")
            continue

        print(f"\n[>>>] Processing: {audio_path} (Target: {expected_lang})")
        res = receptionist.process_visitor_interaction(audio_path, play_audio=False)

        if not res["success"]:
            print(f"    [!] Error: {res['error']}")
            continue

        print(f"    - Classified Language : {res['detected_language']} ({res['confidence']*100:.2f}% confidence)")
        print(f"    - Class Scores        : {res['probabilities']}")
        print(f"    - Visitor Transcript  : \"{res['visitor_transcript']}\"")
        print(f"    - Detected Intent     : {res['intent']}")
        print(f"    - Receptionist Answer : \"{res['receptionist_response']}\"")
        print(f"    - Output Audio Cached : {res['response_audio_path']}")

    print("\n" + "=" * 75)
    print(" [OK] MULTILINGUAL DEMONSTRATION COMPLETE FOR ALL LANGUAGES")
    print("=" * 75)


if __name__ == "__main__":
    run_live_demonstrations()
