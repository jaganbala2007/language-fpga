"""
100% Offline Multilingual AI Receptionist — Interactive Terminal & Voice Console
Target: Xilinx PYNQ-Z2 FPGA / CPU Fixed-Point Simulator
Languages: English (en), Hindi (hi), Tamil (ta)

Zero Cloud Calls | Zero Browser Dependencies | 100% Offline Local Execution
"""

import os
import sys
import argparse
import time
import io

# Ensure UTF-8 console output for multilingual text (Hindi & Tamil Unicode)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.receptionist import MultilingualReceptionist
from src.language_config import SUPPORTED_LANGUAGES, get_language_profile
from src.features import TARGET_SAMPLE_RATE


def print_banner(receptionist: MultilingualReceptionist):
    hw_diag = receptionist.backend.get_diagnostics()
    print("\n" + "=" * 76)
    print("      🎙️  100% OFFLINE MULTILINGUAL AI RECEPTIONIST (LOCAL DESKTOP)  🎙️")
    print("      Target: Xilinx PYNQ-Z2 (Zynq-7000) & Fixed-Point Q8.8 ALU")
    print("      Languages: English (en-IN) | हिन्दी (hi-IN) | தமிழ் (ta-IN)")
    print("=" * 76)
    print(f" [*] Inference Backend   : {receptionist.backend.name.upper()}")
    print(f" [*] Hardware Status     : {receptionist.backend.status}")
    print(f" [*] Offline Mode        : ENFORCED (Zero Cloud / Zero Browser / 100% Local)")
    print(f" [*] Classification Rate : 26 MFCCs -> 16 Hidden (ReLU) -> 3 Classes")
    print("-" * 76)


def print_interaction_summary(res: dict, receptionist: MultilingualReceptionist):
    if not res["success"]:
        print(f"\n [!] Error: {res.get('error', 'Unknown pipeline failure')}")
        return

    lang = res["detected_language"]
    conf = res["confidence"] * 100.0
    scores = res.get("probabilities", {})
    timings = res.get("timings_ms", {})
    transcript = res.get("visitor_transcript", "")
    response_text = res.get("receptionist_response", "")
    audio_path = res.get("response_audio_path")

    print("\n" + "-" * 76)
    print("                      INTERACTION RESULT (100% OFFLINE)")
    print("-" * 76)
    print(f" Detected Language   : {lang.upper()} ({conf:.2f}% confidence)")
    print(f" Visitor Transcript  : \"{transcript}\"")
    print(f" Receptionist Intent : {res.get('intent', 'conversational')}")
    print(f" AI Receptionist     : \"{response_text}\"")

    # Display bar chart in console
    print("\n Class Probability Distribution:")
    for l_name, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bars = "█" * int(prob * 30)
        print(f"   {l_name:<10}: {prob * 100:>6.2f}%  | {bars}")

    # Latency Breakdown
    print("\n Latency Breakdown (Hardware & DSP):")
    for stage, ms in timings.items():
        print(f"   - {stage:<25}: {ms:>7.2f} ms")

    if audio_path and os.path.exists(audio_path):
        print(f"\n Synthesized Audio   : {audio_path}")
        print(" [*] Playing synthesized speech through local speakers...")
        receptionist.offline_tts.play_audio(audio_path, blocking=False)

    print("-" * 76)


def handle_mic_interaction(receptionist: MultilingualReceptionist, duration: float = 4.0):
    print(f"\n[*] Preparing microphone for {duration:.1f}s recording...")
    print("    >> SPEAK NOW IN ENGLISH, HINDI, OR TAMIL <<")
    from src.stt_engine import get_stt_engine
    stt = get_stt_engine()
    ok, wf, wav_path = stt.record_microphone(duration_sec=duration)
    if not ok:
        print(f"[!] Microphone error: {wav_path}")
        return

    print(f"[OK] Audio recorded ({os.path.getsize(wav_path)} bytes). Running local pipeline...")
    res = receptionist.process_visitor_interaction(wav_path, play_audio=False)
    print_interaction_summary(res, receptionist)


def handle_text_interaction(receptionist: MultilingualReceptionist):
    print("\n" + "=" * 50)
    print(" TYPE QUERY IN ENGLISH, HINDI, OR TAMIL")
    print(" Examples:")
    print("  - English : Where is the Engineering Lab?")
    print("  - Hindi   : कार्यालय का समय क्या है?")
    print("  - Tamil   : சேர்க்கை பிரிவு எங்கே இருக்கிறது?")
    print("=" * 50)
    user_text = input("\nVisitor Query > ").strip()
    if not user_text:
        return

    # Select or detect language
    print("\nSelect input language:")
    print(" [1] English (default)")
    print(" [2] Hindi (हिन्दी)")
    print(" [3] Tamil (தமிழ்)")
    choice = input("Choice [1-3, default 1] > ").strip()

    lang_map = {"1": "English", "2": "Hindi", "3": "Tamil", "en": "English", "hi": "Hindi", "ta": "Tamil"}
    active_lang = lang_map.get(choice, "English")

    print(f"\n[*] Reasoning locally in {active_lang}...")
    reply, meta = receptionist.generate_response(visitor_text=user_text, language=active_lang)

    print("\n" + "-" * 76)
    print(f" Language            : {active_lang}")
    print(f" Visitor Said        : \"{user_text}\"")
    print(f" Receptionist Answer : \"{reply}\"")
    print("-" * 76)

    # Synthesize local speech
    tts_ok, audio_path, tts_meta = receptionist.offline_tts.synthesize_speech(reply, language=active_lang)
    if tts_ok and os.path.exists(audio_path):
        print(f"[*] Playing audio ({tts_meta.get('provider', 'Local TTS')})...")
        receptionist.offline_tts.play_audio(audio_path, blocking=False)


def handle_demo_samples(receptionist: MultilingualReceptionist):
    demos = [
        ("1", "English Sample (Welcome to Lab)", "demo_audio/demo_en_welcome.wav"),
        ("2", "Hindi Sample (Office Hours)", "demo_audio/demo_hi_hours.wav"),
        ("3", "Tamil Sample (Admissions)", "demo_audio/demo_ta_admissions.wav"),
        ("4", "Original Recording (Tamil/English)", "Recording.wav")
    ]

    print("\nPreset Audio Demos:")
    for key, name, path in demos:
        status = "Available" if os.path.exists(path) else "Not Found"
        print(f" [{key}] {name:<36} ({status})")

    choice = input("\nSelect sample [1-4] > ").strip()
    selected_path = None
    for key, name, path in demos:
        if choice == key:
            selected_path = path
            break

    if not selected_path or not os.path.exists(selected_path):
        print("[!] Invalid selection or file not found.")
        return

    print(f"\n[*] Processing sample: {selected_path}...")
    res = receptionist.process_visitor_interaction(selected_path, play_audio=False)
    print_interaction_summary(res, receptionist)


def handle_custom_file(receptionist: MultilingualReceptionist):
    path = input("\nEnter path to audio file (.wav, .m4a, .mp3) > ").strip().strip('"').strip("'")
    if not os.path.exists(path):
        print(f"[!] File not found: {path}")
        return

    print(f"\n[*] Processing custom audio: {path}...")
    res = receptionist.process_visitor_interaction(path, play_audio=False)
    print_interaction_summary(res, receptionist)


def display_system_status(receptionist: MultilingualReceptionist):
    stat = receptionist.get_system_status()
    print("\n" + "=" * 76)
    print("                    100% OFFLINE SYSTEM DIAGNOSTICS")
    print("=" * 76)
    print(f" Facility Name     : {stat['knowledge_base']['facility']}")
    print(f" Departments Count : {stat['knowledge_base']['departments_count']}")
    print(f" Inference Backend : {stat['backend']['name']} ({stat['backend']['status']})")
    print(f" Quantization Format: {stat['backend'].get('quantization', 'Q8.8 Fixed-Point')}")
    print(f" Local ASR Engine  : {stat['local_asr']['engine']} ({stat['local_asr']['status']})")
    print(f" Local LLM / Agent : {stat['local_llm']['fallback_agent']}")
    print(f" Local TTS Engine  : {stat['local_tts']['engine']}")
    print(f" Offline Enforced  : {stat['offline_mode_enforced']}")
    print("=" * 76)


def interactive_loop(receptionist: MultilingualReceptionist):
    while True:
        print("\n" + "=" * 45)
        print("       MAIN MENU (100% OFFLINE)")
        print("=" * 45)
        print(" [1] 🎙️  Live Microphone Voice Interaction")
        print(" [2] ⌨️  Type Text Query (English/Hindi/Tamil)")
        print(" [3] 📁 Preset Multilingual Audio Demos")
        print(" [4] 🎵 Process Custom Audio File")
        print(" [5] ⚙️  View Offline Hardware Diagnostics")
        print(" [6] 🔄 Reset Conversation History")
        print(" [0] 🚪 Exit")
        print("-" * 45)

        choice = input("Select an option [0-6] > ").strip()

        if choice == "1":
            dur = input("Recording duration in seconds [default 4.0] > ").strip()
            try:
                duration_val = float(dur) if dur else 4.0
            except ValueError:
                duration_val = 4.0
            handle_mic_interaction(receptionist, duration=duration_val)

        elif choice == "2":
            handle_text_interaction(receptionist)

        elif choice == "3":
            handle_demo_samples(receptionist)

        elif choice == "4":
            handle_custom_file(receptionist)

        elif choice == "5":
            display_system_status(receptionist)

        elif choice == "6":
            receptionist.reset_conversation()
            print("\n[OK] Conversation memory and session history cleared.")

        elif choice == "0" or choice.lower() in ["exit", "quit", "q"]:
            print("\nExiting 100% Offline Receptionist. Goodbye!\n")
            break

        else:
            print("[!] Invalid option. Please enter a number between 0 and 6.")


def main():
    parser = argparse.ArgumentParser(description="100% Offline Multilingual AI Receptionist (Local Desktop Console)")
    parser.add_argument("--mic", action="store_true", help="Directly record from live microphone")
    parser.add_argument("--duration", type=float, default=4.0, help="Microphone recording duration (seconds)")
    parser.add_argument("--audio", type=str, default=None, help="Process single audio file")
    parser.add_argument("--backend", type=str, choices=["cpu", "fpga", "simulated_fpga", "auto"], default="auto", help="Inference backend")
    parser.add_argument("--play", action="store_true", default=True, help="Play synthesized speech")
    args = parser.parse_args()

    receptionist = MultilingualReceptionist(backend_mode=args.backend, offline_mode=True)
    print_banner(receptionist)

    if args.mic:
        handle_mic_interaction(receptionist, duration=args.duration)
    elif args.audio:
        if not os.path.exists(args.audio):
            print(f"[!] Audio file '{args.audio}' not found.")
            return
        res = receptionist.process_visitor_interaction(args.audio, play_audio=args.play)
        print_interaction_summary(res, receptionist)
    else:
        interactive_loop(receptionist)


if __name__ == "__main__":
    main()
