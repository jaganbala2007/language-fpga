# FPGA-Based Multilingual AI Receptionist (100% Offline Voice-to-Voice AI)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Target FPGA](https://img.shields.io/badge/FPGA-Xilinx%20PYNQ--Z2-orange.svg)](http://www.pynq.io/)
[![Languages](https://img.shields.io/badge/Languages-English%20%7C%20Hindi%20%7C%20Tamil-green.svg)]()
[![Model Accuracy](https://img.shields.io/badge/Accuracy-98.17%25%20(Test)-brightgreen.svg)]()
[![Execution Mode](https://img.shields.io/badge/Mode-100%25%20Offline%20Local-success.svg)]()

An end-to-end hardware-accelerated multilingual AI voice receptionist operating **100% locally with zero cloud dependencies at runtime**. The system identifies whether a visitor is speaking **English**, **Hindi**, or **Tamil** using a 26-feature MFCC pipeline and neural network classifier running on a **Xilinx PYNQ-Z2 FPGA** (with bit-accurate CPU/Simulated fallbacks), transcribes speech with **Local Faster-Whisper / Speech Engine**, generates grounded conversational responses with a **Local Deterministic Agent / Ollama LLM**, and speaks naturally with **Local pyttsx3 Windows SAPI5 / Piper TTS**.

---

## 🎙️ 100% Offline Voice Architecture

```
VISITOR AUDIO ──► 16 kHz Mono Audio ──► 13 MFCCs (Mean + Std = 26 Features)
       │
       ▼
 [LOCAL INFERENCE BACKEND] ──► PYNQ-Z2 FPGA (AXI4-Lite) / CPU Q8.8 ALU
       │
       ▼
 [CLASSIFIED LANGUAGE: English / Hindi / Tamil (98.17% Acc)]
       │
       ▼
 [LOCAL ASR: Faster-Whisper / Local Speech Engine] ──► Native Script Transcript
       │
       ▼
 [LOCAL AGENT REASONING: Plan-Act-Verify + Local Ollama] ──► Verified Fact Retrieval
       │
       ▼
 [LOCAL TTS: pyttsx3 Windows SAPI5 / Piper Local Voice] ──► Spoken Output WAV
       │
       ▼
 [LOCAL SPEAKER PLAYBACK] ──► Continuous Multi-Turn Dialogue
```

---

## 🚀 Quick Start (Windows)

### 1. Launch Voice-First Web Application (100% Offline)
```powershell
.\.venv\Scripts\streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### 2. Interactive CLI Receptionist Mode
- **Test with Audio File:**
  ```powershell
  .\.venv\Scripts\python src/cli_receptionist.py --mode file --audio demo_audio/demo_en_welcome.wav
  ```
- **Live Microphone Mode:**
  ```powershell
  .\.venv\Scripts\python src/cli_receptionist.py --mode mic
  ```

---

## ⚡ Inference Backends

| Backend Option | Description | Target Hardware |
| :--- | :--- | :--- |
| `auto` *(Default)* | Auto-probes PYNQ board; uses FPGA hardware if available, falls back safely to CPU. | Host PC or PYNQ-Z2 |
| `fpga` | Real memory-mapped AXI4-Lite hardware driver for physical PYNQ-Z2 board (`0x43C00000`). | PYNQ-Z2 Board |
| `simulated_fpga` | Bit-accurate Q8.8 signed fixed-point integer arithmetic emulator matching FPGA ALU. | Any CPU |
| `cpu` | Scikit-Learn MLPClassifier software inference. | Any CPU |

---

## 📂 Project Structure

- `app.py`: Streamlit web interface with glowing Voice Orb, multi-turn chat, Agent Trace Inspector, and DSP analyzer.
- `src/`:
  - `offline_asr.py`: 100% Local offline speech-to-text engine (Faster-Whisper / Whisper / SpeechRecognition).
  - `offline_llm.py`: 100% Local reasoning engine (Ollama local models + Grounded Plan-Act-Verify Agent).
  - `offline_tts.py`: 100% Local text-to-speech engine (pyttsx3 Windows SAPI5 / Piper).
  - `agent.py`: Autonomous Receptionist Agent (Goal Understanding & Bounded Plan-Act-Verify Loop).
  - `tools.py`: 8 Typed Local Callable Tools (Departments, Hours, Contacts, Appointments, Directions, Escalation).
  - `features.py`: 10-step audio preprocessing & 26-MFCC feature extraction.
  - `fpga_backend.py`: Unified inference backend (FPGA, CPU, Simulated, AUTO).
  - `fixed_point_nn.py`: Q8.8 quantization engine and bit-accurate hardware emulator.
  - `receptionist.py`: Core Multilingual AI Receptionist coordinator with latency tracking.
  - `knowledge_base.json`: Configurable institution domain facts.
- `fpga/`:
  - `rtl/mlp_top.v`: Synthesizable Verilog neural network accelerator.
  - `rtl/axi_lite_slave.v`: AXI4-Lite bus interface.
  - `scripts/build_bitstream.tcl`: Vivado compilation script.
- `tests/`: Automated unit and integration test suite.
