# 🛡️ 100% Offline Multilingual AI Receptionist Setup & Guide

This project is built to execute **100% locally on your machine with ZERO cloud calls, ZERO external API keys, and NO active internet connection required at runtime**.

---

## 🏛️ Local Offline Pipeline Architecture

```
[ Visitor Microphone / Audio ]
              │
              ▼
[ 1. Local Audio Preprocessor ] ── (16 kHz Mono, VAD, Peak Normalized)
              │
              ▼
[ 2. Local 26-MFCC Extractor ] ── (13 Means + 13 Stds via Librosa/NumPy)
              │
              ▼
[ 3. Local Language Classifier ] ── (MLP 26→16→3 on PYNQ-Z2 FPGA / CPU, 98.17% Acc)
              │
              ▼
[ 4. Local Offline ASR ] ────────── (Faster-Whisper / Whisper Local Models)
              │
              ▼
[ 5. Local Reasoning Engine ] ──── (Deterministic Grounded Agent + Local Ollama LLM)
              │
              ▼
[ 6. Local Knowledge & Tools ] ──── (Verified Directory JSON + 8 Local Typed Tools)
              │
              ▼
[ 7. Local Offline TTS ] ────────── (pyttsx3 Windows SAPI5 / Piper Local Voice)
              │
              ▼
[ Local Speaker Audio Playback ]
```

---

## 🚀 Quickstart: Running 100% Offline

### 1. Launch the Streamlit Web Application
```powershell
.\.venv\Scripts\streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 2. Launch Interactive Command-Line Interface (CLI)
```powershell
# Audio file test mode
.\.venv\Scripts\python src/cli_receptionist.py --mode file --audio demo_audio/demo_en_welcome.wav

# Live microphone mode
.\.venv\Scripts\python src/cli_receptionist.py --mode mic
```

---

## 📦 Optional Local Model Enhancements

During **initial setup**, you can optionally download local models for enhanced offline quality:

### A. Local Faster-Whisper ASR
```powershell
pip install faster-whisper
```
The system will automatically cache `whisper-base` into the `models/` directory.

### B. Local Ollama LLM
1. Install [Ollama for Windows](https://ollama.com/download).
2. Pull a compact multilingual model:
```powershell
ollama run qwen2.5:1.5b
```
*(If Ollama is not running, the system automatically uses the built-in deterministic multilingual grounding agent with Plan-Act-Verify loop).*

### C. Local Text-to-Speech (TTS)
The system uses native **Windows SAPI5 offline voices** via `pyttsx3` with zero external dependencies.

---

## 🔌 Hardware Modes

- **AUTO (Recommended):** Probes for physical Xilinx PYNQ-Z2 board over memory-mapped AXI registers (`0x43C00000`). If not connected, gracefully falls back to host CPU inference.
- **Simulated FPGA:** Bit-accurate Q8.8 fixed-point arithmetic matching FPGA Verilog RTL (`fpga/rtl/mlp_top.v`).
- **CPU:** Standard 64-bit float NumPy/Scikit-Learn inference.
