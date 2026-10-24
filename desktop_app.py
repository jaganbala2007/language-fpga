"""
100% Offline Multilingual AI Receptionist — Native Desktop GUI Application
Zero Browser | Zero Cloud APIs | 100% Local Execution

Runs natively on Windows Desktop using Python Tkinter:
- Glowing Animated Voice Orb with dynamic states (Idle, Listening, Processing, Speaking)
- Push-to-Talk Live Microphone Recording
- Language probability distribution meters (English, Hindi, Tamil)
- Multi-turn conversation log
- Hardware & FPGA Fixed-Point status indicators
- Local speech synthesis and playback
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.receptionist import MultilingualReceptionist
from src.language_config import SUPPORTED_LANGUAGES, get_language_profile
from src.features import TARGET_SAMPLE_RATE


class OfflineReceptionistDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("100% Offline Multilingual AI Receptionist | Native Desktop")
        self.root.geometry("980x760")
        self.root.minsize(860, 680)
        self.root.configure(bg="#0f172a")

        # Initialize Offline Receptionist Engine
        self.receptionist = MultilingualReceptionist(backend_mode="auto", offline_mode=True)
        self.is_recording = False
        self.orb_angle = 0
        self.orb_state = "idle"  # idle, listening, processing, speaking

        self._setup_ui()
        self._start_orb_animation()

    def _setup_ui(self):
        # Top Header Frame
        header = tk.Frame(self.root, bg="#1e293b", padx=20, pady=12, relief="flat")
        header.pack(fill="x", padx=14, pady=(14, 8))

        title_lbl = tk.Label(
            header,
            text="🎙️ 100% OFFLINE MULTILINGUAL AI RECEPTIONIST",
            font=("Segoe UI", 16, "bold"),
            fg="#38bdf8",
            bg="#1e293b"
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            header,
            text="Target Hardware: Xilinx PYNQ-Z2 (Zynq-7000) & Fixed-Point ALU | Zero Cloud | Zero Browser",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b"
        )
        sub_lbl.pack(anchor="w")

        # Main 2-Column Split
        main_pane = tk.Frame(self.root, bg="#0f172a")
        main_pane.pack(fill="both", expand=True, padx=14, pady=6)

        # Left Column: Orb, Controls, Hardware Status
        left_frame = tk.Frame(main_pane, bg="#1e293b", padx=16, pady=16, width=380)
        left_frame.pack(side="left", fill="y", padx=(0, 8), pady=0)
        left_frame.pack_propagate(False)

        # Voice Orb Canvas
        self.orb_canvas = tk.Canvas(left_frame, width=170, height=170, bg="#1e293b", highlightthickness=0)
        self.orb_canvas.pack(pady=(5, 10))

        self.status_lbl = tk.Label(
            left_frame,
            text="● SYSTEM READY (100% OFFLINE)",
            font=("Segoe UI", 10, "bold"),
            fg="#34d399",
            bg="#1e293b"
        )
        self.status_lbl.pack(pady=(0, 12))

        # Push-to-Talk Record Button
        self.mic_btn = tk.Button(
            left_frame,
            text="🎙️ Record Voice (3.5s)",
            font=("Segoe UI", 11, "bold"),
            bg="#0284c7",
            fg="white",
            activebackground="#0369a1",
            activeforeground="white",
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
            command=self._on_record_mic_clicked
        )
        self.mic_btn.pack(fill="x", pady=4)

        # File Upload Button
        self.file_btn = tk.Button(
            left_frame,
            text="📁 Select Audio File (.wav/.m4a)",
            font=("Segoe UI", 10),
            bg="#334155",
            fg="#e2e8f0",
            activebackground="#475569",
            activeforeground="white",
            relief="flat",
            padx=10,
            pady=6,
            cursor="hand2",
            command=self._on_select_file_clicked
        )
        self.file_btn.pack(fill="x", pady=4)

        # Preset Audio Demo Buttons Frame
        demo_frame = tk.LabelFrame(left_frame, text="Quick Audio Demos", font=("Segoe UI", 9, "bold"), fg="#94a3b8", bg="#1e293b", padx=8, pady=8)
        demo_frame.pack(fill="x", pady=(10, 6))

        tk.Button(demo_frame, text="🇬🇧 English Demo", font=("Segoe UI", 8), bg="#1e293b", fg="#38bdf8", bd=1, relief="solid", command=lambda: self._process_audio_path("demo_audio/demo_en_welcome.wav")).pack(fill="x", pady=2)
        tk.Button(demo_frame, text="🇮🇳 Hindi Demo (हिन्दी)", font=("Segoe UI", 8), bg="#1e293b", fg="#fb923c", bd=1, relief="solid", command=lambda: self._process_audio_path("demo_audio/demo_hi_hours.wav")).pack(fill="x", pady=2)
        tk.Button(demo_frame, text="🇮🇳 Tamil Demo (தமிழ்)", font=("Segoe UI", 8), bg="#1e293b", fg="#4ade80", bd=1, relief="solid", command=lambda: self._process_audio_path("demo_audio/demo_ta_admissions.wav")).pack(fill="x", pady=2)
        tk.Button(demo_frame, text="🎤 Original Recording", font=("Segoe UI", 8), bg="#1e293b", fg="#c084fc", bd=1, relief="solid", command=lambda: self._process_audio_path("Recording.wav")).pack(fill="x", pady=2)

        # Hardware & Model Status Info
        hw_box = tk.LabelFrame(left_frame, text="Hardware Engine Diagnostics", font=("Segoe UI", 8, "bold"), fg="#94a3b8", bg="#1e293b", padx=8, pady=6)
        hw_box.pack(fill="x", side="bottom", pady=(10, 0))

        hw_text = (
            f"Backend: {self.receptionist.backend.name}\n"
            f"Quantization: Q8.8 Fixed-Point\n"
            f"Features: 26 MFCCs (13 Mean + 13 Std)\n"
            f"Local ASR: Local Whisper / SpeechDecoder\n"
            f"Local TTS: Windows SAPI5 (pyttsx3)"
        )
        tk.Label(hw_box, text=hw_text, font=("Consolas", 8), justify="left", fg="#cbd5e1", bg="#1e293b").pack(anchor="w")

        # Right Column: Live Probability Bars, Conversation History, Text Query Input
        right_frame = tk.Frame(main_pane, bg="#1e293b", padx=16, pady=16)
        right_frame.pack(side="right", fill="both", expand=True)

        # Language Probability Frame
        prob_frame = tk.LabelFrame(right_frame, text="Real-Time Language Classification (FPGA / MLP)", font=("Segoe UI", 10, "bold"), fg="#38bdf8", bg="#1e293b", padx=10, pady=8)
        prob_frame.pack(fill="x", pady=(0, 10))

        self.prob_bars = {}
        self.prob_labels = {}

        for lang, color in [("English", "#38bdf8"), ("Hindi", "#fb923c"), ("Tamil", "#4ade80")]:
            row = tk.Frame(prob_frame, bg="#1e293b")
            row.pack(fill="x", pady=3)

            lbl = tk.Label(row, text=f"{lang:<8}:", font=("Segoe UI", 9, "bold"), fg="#f8fafc", bg="#1e293b", width=10, anchor="w")
            lbl.pack(side="left")

            bar = ttk.Progressbar(row, orient="horizontal", mode="determinate", length=260)
            bar.pack(side="left", fill="x", expand=True, padx=8)

            val_lbl = tk.Label(row, text="0.00%", font=("Consolas", 9), fg=color, bg="#1e293b", width=8, anchor="e")
            val_lbl.pack(side="right")

            self.prob_bars[lang] = bar
            self.prob_labels[lang] = val_lbl

        # Multi-Turn Conversation Log
        chat_frame = tk.LabelFrame(right_frame, text="Multi-Turn Conversation History", font=("Segoe UI", 10, "bold"), fg="#94a3b8", bg="#1e293b", padx=8, pady=8)
        chat_frame.pack(fill="both", expand=True, pady=(0, 8))

        self.chat_box = tk.Text(
            chat_frame,
            font=("Segoe UI", 9),
            bg="#0f172a",
            fg="#f8fafc",
            wrap="word",
            padx=10,
            pady=10,
            relief="flat"
        )
        scroll = ttk.Scrollbar(chat_frame, command=self.chat_box.yview)
        self.chat_box.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.chat_box.pack(side="left", fill="both", expand=True)

        # Initial Welcome Message
        self._log_system_message("AI Receptionist initialized. Speak via microphone or enter a query below.")

        # Text Query Input Frame at Bottom
        input_frame = tk.Frame(right_frame, bg="#1e293b")
        input_frame.pack(fill="x", pady=(4, 0))

        self.text_entry = tk.Entry(
            input_frame,
            font=("Segoe UI", 10),
            bg="#0f172a",
            fg="white",
            insertbackground="white",
            relief="flat",
            bd=5
        )
        self.text_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.text_entry.bind("<Return>", lambda e: self._on_send_text_clicked())

        self.send_btn = tk.Button(
            input_frame,
            text="Send Query",
            font=("Segoe UI", 9, "bold"),
            bg="#0284c7",
            fg="white",
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._on_send_text_clicked
        )
        self.send_btn.pack(side="right")

    def _start_orb_animation(self):
        def animate():
            while True:
                self.orb_angle = (self.orb_angle + 5) % 360
                self._draw_orb()
                time.sleep(0.04)

        t = threading.Thread(target=animate, daemon=True)
        t.start()

    def _draw_orb(self):
        try:
            self.orb_canvas.delete("all")
            cx, cy = 85, 85
            r = 55

            if self.orb_state == "idle":
                color_outer = "#38bdf8"
                color_inner = "#1e40af"
            elif self.orb_state == "listening":
                color_outer = "#f43f5e"
                color_inner = "#ea580c"
            elif self.orb_state == "processing":
                color_outer = "#c084fc"
                color_inner = "#4f46e5"
            else:  # speaking
                color_outer = "#10b981"
                color_inner = "#0284c7"

            # Outer glow
            self.orb_canvas.create_oval(cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8, outline=color_outer, width=2)
            # Inner circle
            self.orb_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color_inner, outline=color_outer, width=3)
            # Center specular highlight
            self.orb_canvas.create_oval(cx - 20, cy - 25, cx + 5, cy - 5, fill="#ffffff", outline="", stipple="gray50")
        except Exception:
            pass

    def _log_system_message(self, text: str):
        self.chat_box.insert("end", f"⚙️ SYSTEM: {text}\n\n")
        self.chat_box.see("end")

    def _log_interaction(self, lang: str, user_text: str, reply: str):
        self.chat_box.insert("end", f"👤 VISITOR ({lang}):\n{user_text}\n\n", "visitor")
        self.chat_box.insert("end", f"🎙️ RECEPTIONIST:\n{reply}\n\n", "receptionist")
        self.chat_box.tag_config("visitor", foreground="#38bdf8", font=("Segoe UI", 9, "bold"))
        self.chat_box.tag_config("receptionist", foreground="#4ade80", font=("Segoe UI", 9))
        self.chat_box.see("end")

    def _on_record_mic_clicked(self):
        if self.is_recording:
            return

        def record_thread():
            self.is_recording = True
            self.orb_state = "listening"
            self.status_lbl.config(text="● LISTENING... (SPEAK NOW)", fg="#f43f5e")
            self.mic_btn.config(state="disabled", text="Recording...")

            try:
                from src.stt_engine import get_stt_engine
                stt = get_stt_engine()
                ok, wf, wav_path = stt.record_microphone(duration_sec=3.5)

                if ok and os.path.exists(wav_path):
                    self.orb_state = "processing"
                    self.status_lbl.config(text="● PROCESSING LOCAL FPGA PIPELINE...", fg="#c084fc")
                    self._process_audio_path(wav_path)
                else:
                    self._log_system_message(f"Microphone recording failed: {wav_path}")
            except Exception as e:
                self._log_system_message(f"Microphone error: {e}")
            finally:
                self.is_recording = False
                self.mic_btn.config(state="normal", text="🎙️ Record Voice (3.5s)")
                self.orb_state = "idle"
                self.status_lbl.config(text="● SYSTEM READY (100% OFFLINE)", fg="#34d399")

        threading.Thread(target=record_thread, daemon=True).start()

    def _on_select_file_clicked(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.m4a *.mp3 *.ogg *.flac"), ("All Files", "*.*")]
        )
        if file_path:
            threading.Thread(target=self._process_audio_path, args=(file_path,), daemon=True).start()

    def _process_audio_path(self, audio_path: str):
        if not os.path.exists(audio_path):
            self._log_system_message(f"File not found: {audio_path}")
            return

        self.orb_state = "processing"
        self.status_lbl.config(text="● COMPUTING 26-MFCC & FPGA CLASSIFICATION...", fg="#c084fc")

        res = self.receptionist.process_visitor_interaction(audio_path, play_audio=False)

        if not res["success"]:
            self._log_system_message(f"Pipeline error: {res.get('error')}")
            self.orb_state = "idle"
            self.status_lbl.config(text="● SYSTEM READY (100% OFFLINE)", fg="#34d399")
            return

        # Update Probability Bars
        scores = res.get("probabilities", {})
        for lang in ["English", "Hindi", "Tamil"]:
            p = scores.get(lang, 0.0)
            self.prob_bars[lang]["value"] = p * 100.0
            self.prob_labels[lang].config(text=f"{p * 100.0:.2f}%")

        # Log conversation
        lang = res["detected_language"]
        transcript = res["visitor_transcript"]
        reply = res["receptionist_response"]
        self._log_interaction(lang, transcript, reply)

        # Synthesized Audio Playback
        audio_out = res.get("response_audio_path")
        if audio_out and os.path.exists(audio_out):
            self.orb_state = "speaking"
            self.status_lbl.config(text=f"● SPEAKING IN {lang.upper()} (SAPI5)...", fg="#10b981")
            self.receptionist.offline_tts.play_audio(audio_out, blocking=False)

        time.sleep(1.5)
        self.orb_state = "idle"
        self.status_lbl.config(text="● SYSTEM READY (100% OFFLINE)", fg="#34d399")

    def _on_send_text_clicked(self):
        query = self.text_entry.get().strip()
        if not query:
            return
        self.text_entry.delete(0, "end")

        def text_thread():
            self.orb_state = "processing"
            self.status_lbl.config(text="● LOCAL AGENT REASONING...", fg="#c084fc")

            # Check if Tamil / Hindi script or English
            if any('\u0b80' <= c <= '\u0bff' for c in query):
                lang = "Tamil"
            elif any('\u0900' <= c <= '\u097f' for c in query):
                lang = "Hindi"
            else:
                lang = "English"

            reply, meta = self.receptionist.generate_response(visitor_text=query, language=lang)
            self._log_interaction(lang, query, reply)

            # Synthesize and play audio
            tts_ok, audio_path, _ = self.receptionist.offline_tts.synthesize_speech(reply, language=lang)
            if tts_ok and os.path.exists(audio_path):
                self.orb_state = "speaking"
                self.status_lbl.config(text=f"● SPEAKING IN {lang.upper()} (SAPI5)...", fg="#10b981")
                self.receptionist.offline_tts.play_audio(audio_path, blocking=False)

            time.sleep(1.2)
            self.orb_state = "idle"
            self.status_lbl.config(text="● SYSTEM READY (100% OFFLINE)", fg="#34d399")

        threading.Thread(target=text_thread, daemon=True).start()


def main():
    root = tk.Tk()
    app = OfflineReceptionistDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
