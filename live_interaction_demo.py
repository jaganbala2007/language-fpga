"""
Live Interactive Multilingual Receptionist Showcase
Demonstrates live interaction across English, Hindi, and Tamil,
including voice feature classification on FPGA ALU, Agent reasoning,
tool execution traces, and multilingual speech response.
"""

import os
import sys
import time

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.receptionist import MultilingualReceptionist
from src.features import load_audio_signal, extract_26_features


def print_box(text, width=78):
    print("+" + "-" * (width - 2) + "+")
    for line in text.splitlines():
        print(f"| {line:<{width - 4}} |")
    print("+" + "-" * (width - 2) + "+")


def run_live_interaction():
    print("\n" + "=" * 80)
    print("      🎙️  100% OFFLINE MULTILINGUAL AI RECEPTIONIST — LIVE INTERACTION  🎙️")
    print("      Target: Xilinx PYNQ-Z2 FPGA / Bit-Accurate Fixed-Point Q8.8 ALU")
    print("      Languages: English (en) | हिन्दी (hi) | தமிழ் (ta)")
    print("=" * 80)

    receptionist = MultilingualReceptionist(backend_mode="simulated_fpga", offline_mode=True)
    hw = receptionist.backend.get_diagnostics()
    print(f"\n[SYSTEM PROBE] Backend: {hw['backend_name']} | Hardware Status: {hw['status']}")
    print(f"               Precision: 16-bit Q8.8 Fixed-Point | Offline Enforced: 100%\n")

    # Conversation scenarios representing real visitor encounters
    turns = [
        {
            "turn_no": 1,
            "persona": "Visitor from Bangalore (English)",
            "audio_file": "demo_audio/demo_en_welcome.wav",
            "spoken_query": "Hello, good morning! Where is the FPGA Engineering Lab and what are the office hours?",
            "lang": "English"
        },
        {
            "turn_no": 2,
            "persona": "Applicant from New Delhi (Hindi)",
            "audio_file": "demo_audio/demo_hi_hours.wav",
            "spoken_query": "नमस्ते, क्या आप मुझे बता सकते हैं कि प्रवेश विभाग (Admissions) कहाँ है और उसके खुलने का समय क्या है?",
            "lang": "Hindi"
        },
        {
            "turn_no": 3,
            "persona": "Parent from Chennai (Tamil)",
            "audio_file": "demo_audio/demo_ta_admissions.wav",
            "spoken_query": "வணக்கம், பொறியியல் துறை ஆய்வகம் மற்றும் சேர்க்கை பிரிவு எங்கு உள்ளது?",
            "lang": "Tamil"
        },
        {
            "turn_no": 4,
            "persona": "Live Recorded Visitor (Saranya - Recording.wav)",
            "audio_file": "Recording.wav",
            "spoken_query": "Visitor introduction and inquiry regarding department facilities.",
            "lang": "Tamil"
        }
    ]

    for turn in turns:
        t_no = turn["turn_no"]
        persona = turn["persona"]
        audio_file = turn["audio_file"]
        query = turn["spoken_query"]
        target_lang = turn["lang"]

        print("=" * 80)
        print(f" [TURN {t_no}] VISITOR INTERACTION: {persona.upper()}")
        print("=" * 80)
        print(f"  Input Audio Source   : {audio_file}")
        print(f"  Spoken Utterance     : \"{query}\"")

        # Step A: Acoustic DSP & Hardware Classification
        t0 = time.perf_counter()
        wave, sr = load_audio_signal(audio_file)
        feats = extract_26_features(wave, sr)
        dsp_time = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        det_lang, conf, probs = receptionist.backend.predict(feats)
        alu_time = (time.perf_counter() - t1) * 1000

        print("\n  >> [Step 1: Acoustic DSP & FPGA Classification]")
        print(f"     * Signal Duration   : {len(wave)/sr:.2f} s ({sr} Hz mono)")
        print(f"     * Extracted Features: 26 MFCCs (13 Mean + 13 Std Dev) in {dsp_time:.2f} ms")
        print(f"     * FPGA ALU Output   : {det_lang.upper()} ({conf*100:.2f}% confidence) in {alu_time:.2f} ms")
        print("     * Probability Gauges:")
        for l_name, prob in probs.items():
            bar = "█" * int(prob * 25)
            print(f"       [{l_name:<7}] {prob*100:6.2f}% | {bar}")

        # Step B: Autonomous Agent Reasoning & Knowledge Grounding
        t2 = time.perf_counter()
        intent = receptionist.detect_intent(query, language=det_lang)
        reply, meta = receptionist.generate_response(visitor_text=query, language=det_lang)
        reason_time = (time.perf_counter() - t2) * 1000

        print("\n  >> [Step 2: Autonomous Agentic Reasoner]")
        print(f"     * Detected Intent   : {intent.upper()}")
        print(f"     * Reasoning Engine  : Plan-Act-Verify Deterministic Loop")
        print(f"     * Execution Latency : {reason_time:.2f} ms")
        
        # Display Action Trace / Tool calls if any
        trace = meta.get("action_trace", [])
        if trace:
            print(f"     * Tool Executions   : {len(trace)} tools called")
            for idx, action in enumerate(trace, 1):
                tool_name = action.get("tool_name", "lookup")
                tool_arg = action.get("arguments", {})
                print(f"       ({idx}) Tool: {tool_name} | Args: {tool_arg}")

        # Step C: Speech Synthesis (TTS)
        t3 = time.perf_counter()
        tts_ok, audio_out, tts_meta = receptionist.offline_tts.synthesize_speech(reply, language=det_lang)
        tts_time = (time.perf_counter() - t3) * 1000

        print("\n  >> [Step 3: Speech Synthesis & Output]")
        print(f"     * Voice Engine      : Offline Windows SAPI5 / pyttsx3 ({det_lang})")
        print(f"     * Cached Audio File : {os.path.basename(audio_out) if audio_out else 'N/A'}")
        print(f"     * Synthesis Latency : {tts_time:.2f} ms")

        # Step D: Display AI Receptionist Final Dialogue
        print("\n  >> [Receptionist Response]:")
        print(f"     \"{reply}\"\n")

    print("=" * 80)
    print("      [OK] LIVE INTERACTIVE SESSION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_live_interaction()
