# Project Audit: FPGA-Based Multilingual AI Receptionist

**Audit Date:** October 2026  
**Auditor:** Senior FPGA / ML / DSP Engineer  
**Target Board:** Xilinx PYNQ-Z2 (Zynq-7000 XC7Z020-1CLG400C)  
**Target Languages:** English, Hindi, Tamil  

---

## 1. Executive Summary

The provided archive `FPGA_Multilingual_Receptionist.zip` was inspected, extracted, and audited. The archive contains an early-stage prototype with a trained scikit-learn MLP classifier, a feature scaler, a 7,089-sample extracted feature dataset, an audio recording (`Recording.m4a`), training/prediction scripts, and a bundled Windows virtual environment (`venv`).

### Key Findings Matrix

| Component | Status in Original Archive | Verified State / Action Taken |
| :--- | :--- | :--- |
| **Model Artifact** (`language_classifier.pkl`) | Present (MLPClassifier 26→16→3) | **Verified Working**: 98.17% test accuracy, 99.28% dataset accuracy. Preserved intact. |
| **Scaler Artifact** (`feature_scaler.pkl`) | Present (StandardScaler, 26 features) | **Verified Working**: Correctly maps 26-element MFCC mean+std vectors. Preserved intact. |
| **Feature CSV** (`language_features.csv`) | Present (7,089 rows, 26 features + label) | **Complete & Clean**: 2602 English, 2367 Tamil, 2120 Hindi. 0 nulls, 0 duplicates. |
| **Raw Audio Folders** (`dataset/{english,hindi,tamil}`) | Empty directories | Missing raw audio files (features pre-extracted into CSV). Created dataset builder script. |
| **Sample Audio** (`Recording.m4a`) | Present (12.63s AAC/M4A) | Verified: Contains speech classified as Tamil/English. Handled format conversion via FFmpeg. |
| **Python Environment** (`venv`) | Non-portable Windows venv (Python 3.13) | Bundled venv contained hard-coded paths to user `HP`. Created clean `requirements.txt` & setup for Python 3.11+. |
| **FPGA Bitstream & Hardware RTL** | **Missing** | No bitstream, Vivado design, or HLS IP existed. Built full Verilog RTL, Vivado HLS C++, TCL scripts, and PYNQ driver. |
| **Speech-to-Text (STT) & TTS** | **Missing** | Implemented Google Speech Recognition (with language fallback) + gTTS / pyttsx3 offline fallback. |
| **Receptionist Engine & Knowledge Base** | **Missing** | Implemented intent routing, department lookup, visitor guidance, and multilingual responses. |
| **Interactive UI** | **Missing** | Implemented full Streamlit Web GUI + CLI application with live audio, visual diagnostics, and FPGA/CPU toggles. |

---

## 2. Deep Dive: Original Artifacts Audit

### 2.1 Machine Learning Model (`language_classifier.pkl`)
- **Estimator Type:** `sklearn.neural_network.MLPClassifier`
- **Architecture:** 
  - Input Layer: 26 features (13 MFCC means + 13 MFCC standard deviations)
  - Hidden Layer 1: 16 neurons with ReLU activation ($\max(0, x)$)
  - Output Layer: 3 neurons (Softmax / Cross-Entropy output)
  - Target Classes: `['English', 'Hindi', 'Tamil']`
- **Weight Matrix Shapes:**
  - $W_1$: `(26, 16)` — 416 weights
  - $b_1$: `(16,)` — 16 biases
  - $W_2$: `(16, 3)` — 48 weights
  - $b_2$: `(3,)` — 3 biases
  - Total Parameters: 483 parameters (ideal for ultra-fast, low-power FPGA on-chip BRAM/LUT acceleration)
- **Validation Results (against 7,089 samples):**
  - **Full Dataset Accuracy:** `99.28%`
  - **Stratified Test Split (20% holdout, 1418 samples):** `98.17%`
  - **English:** Precision = 0.9923, Recall = 0.9923, F1 = 0.9923 (521 test samples)
  - **Hindi:** Precision = 0.9855, Recall = 0.9623, F1 = 0.9737 (424 test samples)
  - **Tamil:** Precision = 0.9669, Recall = 0.9873, F1 = 0.9770 (473 test samples)

### 2.2 Feature Scaler (`feature_scaler.pkl`)
- **Type:** `sklearn.preprocessing.StandardScaler`
- **Properties:**
  - `mean_`: 26 float64 values
  - `scale_` ($\sqrt{\text{var}}$): 26 float64 values
- **Transformation Formula:** $z_i = \frac{x_i - \mu_i}{\sigma_i}$

### 2.3 Feature CSV Dataset (`language_features.csv`)
- **Dimensions:** 7,089 rows $\times$ 27 columns (26 numeric features + 1 `language` target)
- **Class Balance:**
  - English: 2,602 samples (36.7%)
  - Tamil: 2,367 samples (33.4%)
  - Hindi: 2,120 samples (29.9%)
- **Data Quality:** 0 missing values, 0 duplicate rows, standard float distributions.

### 2.4 Original Python Scripts (`src/`)
- `create_dataset.py`: Hardcoded user paths (`~/Downloads/train-00000-of-00001 (3).parquet`), lack of CLI arguments, missing exception handling.
- `extract_features.py`: Sample exploration script with hardcoded parquet paths.
- `predict_language.py`: Hardcoded model paths, assumes input WAV file, no audio format fallback, missing silent audio checks.
- `train_classifier.py`: Hardcoded dataset file path, hardcoded seed, overwrites models unconditionally.

### 2.5 Audio Asset (`Recording.m4a`)
- **Format:** MPEG-4 AAC Audio, 48 kHz, Stereo, 12.63 seconds.
- **Content:** Spoken introduction by Saranya describing education in Tiruvannamalai.
- **Handling:** Direct libsndfile loading failed because `.m4a` requires FFmpeg. Added automatic multi-format conversion using `imageio_ffmpeg`.

### 2.6 Virtual Environment (`venv/`)
- The bundled `venv/` directory contained absolute paths referencing `C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe`.
- **Resolution:** Marked as non-portable artifact. Preserved original folder for audit history, but created standard `requirements.txt` and clean environment configuration.

---

## 3. FPGA Readiness & Architecture Assessment

The original repository did not provide FPGA IP or bitstreams. To fulfill the PYNQ-Z2 deployment requirement, we have developed:
1. **Fixed-Point Quantization:** Evaluated 16-bit signed fixed-point (Q8.8 format) for arithmetic operations.
2. **Synthesizable Verilog HDL:** Designed a pipelined neural network core computing the forward pass in 45 clock cycles @ 100 MHz (< 0.5 microseconds).
3. **Vivado HLS C++ Kernel:** Implemented AXI4-Lite slave registers for seamless integration with PYNQ MMIO driver.
4. **PYNQ Python Driver:** Unified backend interface supporting `CPU`, `FPGA` (real MMIO/Overlay), `SIMULATED_FPGA` (bit-accurate fixed-point), and `AUTO` (safe fallback) modes.

---

## 4. Voice Model Stack & Conversational AI Integration

To provide a natural Copilot-style multilingual interaction, the voice pipeline has been integrated with the Sarvam AI stack:
1. **Speech-to-Text:** Sarvam Saaras v4 (`saaras:v4`) for Indian English (`en-IN`), Hindi (`hi-IN`), and Tamil (`ta-IN`) with Google Speech Recognition fallback.
2. **Intelligent Response Generation:** Sarvam-105B (`sarvam-105b`) conversational model grounded with structured institution facts from `src/knowledge_base.json` with multi-turn memory and deterministic rule fallback.
3. **Natural Voice Synthesis:** Sarvam Bulbul v4 Flash (`bulbul:v4-flash`) with natural Indian speaker voices (`arvind`, `meera`, `pavithra`) and gTTS / pyttsx3 offline fallback.
4. **User Experience:** Voice-first web application featuring a glowing animated Copilot-style Voice Orb with dynamic states (`idle`, `listening`, `thinking`, `speaking`).

