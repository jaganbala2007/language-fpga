# System Architecture: FPGA-Based Multilingual AI Receptionist

## 1. End-to-End System Pipeline

```
 VISITOR AUDIO
       │
       ▼
 [AUDIO CAPTURE] ──> Microphone / Audio File (.wav, .m4a, .mp3)
       │
       ▼
 [PREPROCESSING] ──> 16,000 Hz Mono Conversion & Signal Validation (RMS / Duration)
       │
       ▼
 [DSP EXTRACTION] ──> 13 MFCCs -> Mean (13) + Std (13) = Exact 26 Features
       │
       ▼
 [STANDARDIZATION] ──> StandardScaler (z = (x - mean) / scale)
       │
       ├───► [FPGA BACKEND (PYNQ-Z2)] ──► AXI4-Lite MMIO Core (Q8.8 Fixed-Point, 45 Cycles @ 100MHz)
       └───► [CPU BACKEND (MLPClassifier)] ──► 26 -> 16 (ReLU) -> 3 (Softmax)
       │
       ▼
 [LANGUAGE DECISION] ──► English / Hindi / Tamil with Confidence Scores
       │
       ▼
 [SPEECH-TO-TEXT] ──► Language-Specific Acoustic Model (en-IN / hi-IN / ta-IN)
       │
       ▼
 [INTENT ROUTING] ──► Multilingual Domain Knowledge Base
       │
       ▼
 [RESPONSE GENERATION] ──► Native Natural Language Response Text
       │
       ▼
 [TEXT-TO-SPEECH] ──► Multilingual TTS (gTTS / pyttsx3)
       │
       ▼
 [SPEAKER PLAYBACK] ──► Pygame Mixer / OS Sound Device
```

---

## 2. Computational Work Distribution (CPU vs FPGA)

| Pipeline Stage | Executed On | Rationale / Implementation Details |
| :--- | :--- | :--- |
| **Audio Acquisition & FFmpeg Format Conversion** | **Host CPU** | OS driver I/O, file decoding, buffer management. |
| **FFT & Filterbank Processing (13 MFCCs)** | **Host CPU** | Standard DSP frontend (librosa / scipy). |
| **Feature Standardization (StandardScaler)** | **CPU or FPGA** | FPGA hardware includes dedicated fixed-point scaling pipeline. |
| **Neural Network Inference (26→16→3 MLP)** | **PYNQ-Z2 FPGA** | Accelerated via custom fixed-point AXI4-Lite IP block in < 0.5 µs. |
| **Speech-to-Text Transcription (STT)** | **Host CPU / Cloud** | Acoustic decoding with language-specific locale models. |
| **Receptionist Intent & Response Logic** | **Host CPU** | Deterministic domain knowledge base matching. |
| **Text-to-Speech Synthesis (TTS)** | **Host CPU** | Audio synthesis engine and speaker output playback. |

---

## 3. Directory Layout

```
├── app.py                         # Interactive Streamlit Web GUI
├── language_classifier.pkl        # Trained 26->16->3 MLPClassifier (98.17% test acc)
├── feature_scaler.pkl             # Trained StandardScaler (26 features)
├── language_features.csv          # 7,089-sample dataset (English, Hindi, Tamil)
├── Recording.wav / Recording.m4a  # Original test audio sample
├── requirements.txt               # Cross-platform dependencies
├── .env.example                   # Environment configuration template
│
├── src/
│   ├── agent.py                   # Autonomous Receptionist Agent (Goal Understanding & Plan-Act-Verify Loop)
│   ├── tools.py                   # Typed Tool Registry & Safety Verification Suite
│   ├── language_config.py         # Central language metadata & voice catalog (en-IN, hi-IN, ta-IN)
│   ├── sarvam_voice_stack.py      # Sarvam Saaras v4 STT, Sarvam-105B LLM, and Bulbul v4 Flash TTS
│   ├── features.py                # Canonical 10-step audio processor & 26-MFCC extractor
│   ├── fpga_backend.py            # Unified backend driver (CPU, FPGA, AUTO, Simulated)
│   ├── fixed_point_nn.py          # Q8.8 fixed-point quantization & bit-accurate emulator
│   ├── receptionist.py            # Multilingual receptionist engine with multi-turn session tracking
│   ├── stt_engine.py              # Multilingual speech recognition (Sarvam Saaras + fallback)
│   ├── tts_engine.py              # Multilingual speech synthesis (Sarvam Bulbul + fallback)
│   ├── predict_language.py        # CLI prediction tool
│   ├── evaluate_model.py          # Standalone model validation tool
│   ├── train_classifier.py        # Robust model training pipeline
│   ├── create_dataset.py          # Reproducible dataset preparation script
│   ├── cli_receptionist.py        # Interactive CLI receptionist interface
│   └── knowledge_base.json        # Configurable reception domain information
│
├── fpga/
│   ├── rtl/                       # Synthesizable Verilog HDL files
│   │   ├── mlp_top.v              # Pipelined neural network core
│   │   ├── fixed_point_alu.v      # Signed Q8.8 MAC and ReLU activation unit
│   │   ├── axi_lite_nn_controller.v # AXI4-Lite slave memory-mapped register interface
│   │   └── tb_mlp_top.v           # Complete Verilog simulation testbench
│   ├── hls/                       # Vivado HLS C++ IP design
│   │   ├── mlp_nn_hls.cpp         # Synthesizable HLS kernel
│   │   ├── mlp_nn_hls.h           # HLS definitions and structs
│   │   └── model_weights_fixed.h  # Exported fixed-point weights & biases
│   ├── weights/                   # Hex memories for Verilog testbenches
│   └── scripts/                   # Vivado project generation TCL scripts
│
└── tests/                         # Automated Pytest Suite (37 Tests)
    ├── test_agent.py              # Plan-act-verify loop, typed tools, and safety tests
    ├── test_audio_processor.py
    ├── test_feature_extraction.py
    ├── test_model_validation.py
    ├── test_fixed_point_nn.py
    ├── test_fpga_backend.py
    ├── test_receptionist.py
    ├── test_language_config.py
    ├── test_sarvam_voice_stack.py
    ├── test_multi_turn_receptionist.py
    └── test_live_demo.py
```

---

## 4. Autonomous Agent Architecture: Plan-Act-Verify Loop

```
[VISITOR SPEECH] ──► [26-MFCC FPGA CLASSIFIER] ──► [ASR TRANSCRIPTION]
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AUTONOMOUS REASONING AGENT                            │
│                                                                             │
│  1. UNDERSTAND GOAL & ENTITIES                                              │
│     • Intent, Visitor Name, Host, Department, Urgency                      │
│                                                                             │
│  2. BOUNDED PLAN-ACT-VERIFY LOOP (max 4 iterations)                         │
│     • Tool Selection:                                                       │
│       ├── lookup_department(dept_name)                                      │
│       ├── check_office_hours(day)                                           │
│       ├── get_contact_details(target)                                       │
│       ├── check_appointment(visitor_name, host)                             │
│       ├── get_campus_directions(from, to_dept)                              │
│       ├── search_knowledge_base(query)                                      │
│       ├── escalate_to_human(reason, urgency)                                │
│       └── schedule_appointment(name, host, time, confirmed) [SAFETY GATED]  │
│     • Result Verification: Compare outputs against verified database        │
│                                                                             │
│  3. SHORT-TERM SESSION MEMORY                                               │
│     • Entity tracking, context carryover, privacy-safe clearable session    │
│                                                                             │
│  4. SAME-LANGUAGE RESPONSE SYNTHESIS                                        │
│     • Strict English, Hindi, or Tamil speech output generation              │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
            [SARVAM BULBUL V4 FLASH / TTS SYNTHESIS]
                               │
                               ▼
                   [NATURAL AUDIO PLAYBACK]
```
