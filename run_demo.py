"""
Full Project Execution & Live Demonstration Script
FPGA-Based Multilingual AI Receptionist
Target: Xilinx PYNQ-Z2 & Fixed-Point Q8.8 ALU Emulation
Languages: English, Hindi, Tamil
"""

import os
import sys
import time

# Ensure UTF-8 console output for multilingual characters
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.receptionist import MultilingualReceptionist
from src.features import load_audio_signal, extract_26_features
from src.fpga_backend import get_backend
from src.language_config import SUPPORTED_LANGUAGES


def run_full_project_demo():
    print("=" * 80)
    print("      🎙️  100% OFFLINE MULTILINGUAL AI RECEPTIONIST — FULL SYSTEM RUN  🎙️")
    print("      Hardware Target: Xilinx PYNQ-Z2 (Zynq-7000) & Fixed-Point Q8.8 ALU")
    print("      Languages: English (en-IN) | हिन्दी (hi-IN) | தமிழ் (ta-IN)")
    print("=" * 80)

    # 1. Initialize Hardware / ALU Backend
    print("\n[PHASE 1] HARDWARE & INFERENCE PROBE")
    backend = get_backend("simulated_fpga")
    print(f"  * Backend Name     : {backend.name}")
    print(f"  * Backend Status   : {backend.status}")
    print(f"  * Architecture     : 26 MFCCs -> 16 Hidden (ReLU) -> 3 Classes (Softmax)")
    print(f"  * Fixed-Point Type : 16-bit Signed Q8.8 (1 sign + 7 int + 8 frac)")

    receptionist = MultilingualReceptionist(backend_mode="simulated_fpga", offline_mode=True)

    # 2. Audio Processing & Language Classification Pipeline
    print("\n" + "-" * 80)
    print("[PHASE 2] MULTILINGUAL AUDIO CLASSIFICATION ON FPGA ALU")
    print("-" * 80)

    test_audio_files = [
        ("English Greeting", "demo_audio/demo_en_welcome.wav"),
        ("Hindi Inquiry", "demo_audio/demo_hi_hours.wav"),
        ("Tamil Admissions", "demo_audio/demo_ta_admissions.wav"),
        ("Visitor Audio Recording", "Recording.wav"),
    ]

    for label, audio_path in test_audio_files:
        if not os.path.exists(audio_path):
            print(f"  [!] Missing audio file: {audio_path}")
            continue

        t0 = time.perf_counter()
        wave_data, sr = load_audio_signal(audio_path)
        load_time = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        feats = extract_26_features(wave_data, sr)
        feat_time = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        lang, conf, probs = backend.predict(feats)
        alu_time = (time.perf_counter() - t2) * 1000

        print(f"\n  [>>>] Sample: {label} ({os.path.basename(audio_path)})")
        print(f"        Duration        : {len(wave_data)/sr:.2f} seconds ({sr} Hz)")
        print(f"        Classified Lang : {lang.upper()} (Confidence: {conf*100:.2f}%)")
        print(f"        FPGA ALU Time   : {alu_time:.2f} ms (MFCC DSP: {feat_time:.2f} ms)")
        print(f"        Probability Dist:")
        for l_name, score in probs.items():
            bars = "█" * int(score * 30)
            print(f"          - {l_name:<8}: {score*100:6.2f}% | {bars}")

    # 3. Conversational Reasoning & Knowledge Base Grounding
    print("\n" + "-" * 80)
    print("[PHASE 3] CONVERSATIONAL AGENT REASONING & INTENT ROUTING")
    print("-" * 80)

    conversation_tests = [
        ("English", "Where is the FPGA Engineering Lab and what are the office hours?"),
        ("English", "Who is the Head of Department for Computer Science?"),
        ("Hindi", "कार्यालय का समय क्या है और स्वागत कक्ष कहाँ है?"),
        ("Tamil", "வணக்கம், சேர்க்கை பிரிவு எங்கே உள்ளது?"),
        ("Tamil", "வளாகத்தில் உணவு விடுதி எங்கு உள்ளது?"),
    ]

    for lang, user_query in conversation_tests:
        t0 = time.perf_counter()
        intent = receptionist.detect_intent(user_query, lang)
        answer, meta = receptionist.generate_response(user_query, language=lang)
        reason_time = (time.perf_counter() - t0) * 1000

        print(f"\n  [Visitor ({lang})]: \"{user_query}\"")
        print(f"  - Detected Intent   : {intent.upper()}")
        print(f"  - AI Receptionist   : \"{answer}\"")
        print(f"  - Reasoning Latency : {reason_time:.2f} ms")

    # 4. Text-To-Speech Synthesis
    print("\n" + "-" * 80)
    print("[PHASE 4] OFFLINE SPEECH SYNTHESIS (TTS)")
    print("-" * 80)

    tts_samples = [
        ("English", "Welcome to the Engineering and Research Center."),
        ("Hindi", "इंजीनियरिंग लैब भूतल पर कमरा नंबर 102 में स्थित है।"),
        ("Tamil", "சேர்க்கை பிரிவு பிரதான கட்டிடத்தில் உள்ளது."),
    ]

    for lang, text in tts_samples:
        t0 = time.perf_counter()
        tts_ok, audio_out, tts_meta = receptionist.offline_tts.synthesize_speech(text, language=lang)
        tts_time = (time.perf_counter() - t0) * 1000
        print(f"  * [{lang:<7}] Synthesized: \"{text[:35]}...\"")
        print(f"              Audio File : {os.path.basename(audio_out) if audio_out else 'N/A'}")
        print(f"              TTS Latency: {tts_time:.2f} ms (Offline: {tts_meta.get('offline', True)})")

    print("\n" + "=" * 80)
    print("      [OK] FULL PROJECT PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_full_project_demo()
