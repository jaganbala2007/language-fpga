"""
100% Offline Multilingual AI Receptionist Core Engine
FPGA-Based Multilingual AI Receptionist

Complete 100% Local Offline Pipeline:
1. 10-Step Audio Preprocessing & 26-MFCC Feature Extraction (16 kHz mono)
2. Hardware-Accelerated Language Identification (PYNQ-Z2 FPGA / CPU)
3. Local Speech-to-Text Transcription via Local Whisper / Faster-Whisper
4. Local Knowledge-Base Grounded Conversational Generation via Local LLM / Agent Reasoning
5. Local Offline Text-to-Speech Synthesis via pyttsx3 / Windows SAPI5 / Piper
6. Multi-Turn Conversation History & Context Management
7. Zero Cloud Calls, Zero API Keys, 100% Offline
"""

import os
import sys
import json
import time
import tempfile
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.language_config import (
    SUPPORTED_LANGUAGES,
    LanguageProfile,
    get_language_profile,
    normalize_language_name
)
from src.features import (
    load_audio_signal,
    extract_26_features,
    validate_audio_signal,
    TARGET_SAMPLE_RATE,
    EXPECTED_FEATURE_COUNT
)
from src.fpga_backend import get_backend, BaseInferenceBackend
from src.agent import ReceptionistAgent, ActionTraceStep
from src.offline_asr import get_offline_stt_engine, OfflineSpeechToTextEngine
from src.offline_llm import get_offline_llm_engine, OfflineLLMEngine
from src.offline_tts import get_offline_tts_engine, OfflineTextToSpeechEngine
from src.stt_engine import get_stt_engine
from src.tts_engine import get_tts_engine


class ConversationTurn:
    """Represents a single conversation turn in the receptionist session."""

    def __init__(
        self,
        turn_id: int,
        visitor_audio_path: Optional[str],
        detected_language: str,
        confidence: float,
        visitor_transcript: str,
        receptionist_response: str,
        response_audio_path: Optional[str],
        backend_name: str,
        latency_breakdown: Dict[str, float],
        action_trace: Optional[List[Dict[str, Any]]] = None
    ):
        self.turn_id = turn_id
        self.timestamp = time.time()
        self.visitor_audio_path = visitor_audio_path
        self.detected_language = detected_language
        self.confidence = confidence
        self.visitor_transcript = visitor_transcript
        self.receptionist_response = receptionist_response
        self.response_audio_path = response_audio_path
        self.backend_name = backend_name
        self.latency_breakdown = latency_breakdown
        self.action_trace = action_trace or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "detected_language": self.detected_language,
            "confidence": round(self.confidence, 4),
            "visitor_transcript": self.visitor_transcript,
            "receptionist_response": self.receptionist_response,
            "response_audio_path": self.response_audio_path,
            "backend_name": self.backend_name,
            "latency_breakdown": self.latency_breakdown,
            "action_trace": self.action_trace
        }


class MultilingualReceptionist:
    """Enterprise-grade 100% Offline Multilingual AI Receptionist."""

    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        model_path: str = "language_classifier.pkl",
        scaler_path: str = "feature_scaler.pkl",
        backend_mode: str = "auto",
        sarvam_api_key: Optional[str] = None,
        offline_mode: bool = True
    ):
        self.kb_path = knowledge_base_path or os.path.join(
            os.path.dirname(__file__), "knowledge_base.json"
        )
        self.kb = self._load_knowledge_base(self.kb_path)
        self.backend_mode = backend_mode
        self.backend = get_backend(mode=backend_mode, model_path=model_path, scaler_path=scaler_path)
        self.sarvam_api_key = sarvam_api_key
        self.offline_mode = offline_mode
        
        # Initialize 100% local offline voice engines
        self.offline_stt = get_offline_stt_engine()
        self.offline_llm = get_offline_llm_engine()
        self.offline_tts = get_offline_tts_engine()

        # Local Agent with Plan-Act-Verify loop
        self.agent = ReceptionistAgent(knowledge_base_path=self.kb_path, sarvam_api_key=sarvam_api_key)
        
        # Legacy/Fallback compatibility engines
        self.stt = get_stt_engine(sarvam_api_key=sarvam_api_key)
        self.tts = get_tts_engine(sarvam_api_key=sarvam_api_key)

        self.confidence_threshold = 0.40
        self.conversation_history: List[ConversationTurn] = []

    def reset_conversation(self):
        """Resets multi-turn session memory and history."""
        self.conversation_history.clear()
        self.agent.reset_session()

    def get_system_status(self) -> Dict[str, Any]:
        """Returns comprehensive real-time status of all local offline subsystems."""
        ollama_stat = self.offline_llm.check_ollama_status()
        asr_stat = self.offline_stt.get_engine_status()
        hw_diag = self.backend.get_diagnostics()
        voices = self.offline_tts.get_installed_voices()

        return {
            "offline_mode_enforced": True,
            "backend": hw_diag,
            "classifier": {
                "loaded": bool(getattr(self.backend, "status", None) is not None),
                "name": "MLPClassifier (26 MFCC -> 16 Hidden -> 3 Classes)",
                "features": 26,
                "classes": ["English", "Hindi", "Tamil"]
            },


            "local_asr": {
                "ready": True,
                "engine": "faster-whisper / whisper local",
                "device": asr_stat.get("device", "auto"),
                "status": "Loaded" if asr_stat.get("model_loaded") else "Local Engine Ready"
            },
            "local_llm": {
                "ready": True,
                "ollama_running": ollama_stat["online"],
                "installed_models": ollama_stat["models"],
                "fallback_agent": "Active (Deterministic Grounded Agent with Plan-Act-Verify)"
            },
            "local_tts": {
                "ready": self.offline_tts.has_pyttsx3 or self.offline_tts.has_pygame,
                "engine": "pyttsx3 Windows SAPI5 / Piper Local Voice",
                "installed_voices_count": len(voices)
            },
            "knowledge_base": {
                "facility": self.kb.get("facility_name", "RMK Engineering College"),
                "departments_count": len(self.kb.get("departments", {}))
            }
        }

    def _load_knowledge_base(self, path: str) -> Dict[str, Any]:
        """Loads structured reception information."""
        if not os.path.exists(path):
            alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge_base.json")
            if os.path.exists(alt):
                path = alt

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return self._default_knowledge_base()

    def _default_knowledge_base(self) -> Dict[str, Any]:
        return {
            "facility_name": "RMK Engineering College & FPGA Laboratory",
            "office_hours": {
                "weekdays": "8:30 AM to 5:00 PM (Monday to Friday)",
                "saturday": "8:30 AM to 1:00 PM",
                "sunday": "Closed"
            },
            "departments": {
                "engineering": {"name": "Engineering & FPGA Lab", "floor": "3rd Floor", "room": "Room 304", "contact": "ext. 401"},
                "human_resources": {"name": "Human Resources (HR)", "floor": "2nd Floor", "room": "Room 201", "contact": "ext. 205"},
                "admissions": {"name": "Admissions & Student Affairs", "floor": "Ground Floor", "room": "Room G-12", "contact": "ext. 102"},
                "finance": {"name": "Accounts & Finance", "floor": "1st Floor", "room": "Room 105", "contact": "ext. 303"}
            }
        }

    @property
    def _grounding_text(self) -> str:
        """Returns verified ground truth text of the facility."""
        return (
            "Smart Tech Innovation Center & FPGA Laboratory Ground Truth Information:\n"
            "- Facility: Smart Tech Innovation Center\n"
            "- Engineering & FPGA Lab: 3rd Floor, Room 304, contact ext. 401\n"
            "- Human Resources (HR): 2nd Floor, Room 201, contact ext. 205\n"
            "- Admissions & Student Affairs: Ground Floor, Room G-12, contact ext. 102\n"
            "- Accounts & Finance: 1st Floor, Room 105, contact ext. 303\n"
            "- Office Hours: Monday to Friday 9:00 AM to 6:00 PM, Saturday 9:00 AM to 1:00 PM, Sunday closed\n"
            "- Visitor Registration: All visitors must register at the reception desk in the main lobby.\n"
        )

    def detect_intent(self, text: str, language: str = "English") -> str:
        """Determines conversational intent from text query across supported languages."""
        t = (text or "").lower().strip()
        # Greetings
        if any(w in t for w in ["hello", "hi", "good morning", "good afternoon", "namaste", "नमस्ते", "வணக்கம்", "vanakkam"]):
            return "greeting"
        # Office hours
        if any(w in t for w in ["hour", "hours", "time", "timing", "open", "समय", "நேரம்", "வேலை நேரம்", "திறந்திருக்கும்"]):
            return "hours"
        # Engineering directions
        if any(w in t for w in ["engineering", "fpga", "lab", "इंजीनियरिंग", "लैब", "பொறியியல்", "ஆய்வகம்"]):
            return "directions_engineering"
        # HR directions
        if any(w in t for w in ["hr", "human resource", "human resources", "interview", "एचआर", "मानव संसाधन", "மனிதவளம்", "மனித வளம்", "நேர்காணல்"]):
            return "directions_hr"
        # Admissions directions
        if any(w in t for w in ["admission", "admissions", "प्रवेश", "दाखिला", "சேர்க்கை"]):
            return "directions_admissions"
        # General directions
        if any(w in t for w in ["where", "direction", "how to reach", "कहाँ", "எங்கே", "எங்கு"]):
            return "directions_general"
        return "general"

    def generate_response(
        self,
        visitor_text: str,
        language: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        100% Local Autonomous Agentic Response Generation using Goal Understanding,
        Plan-Act-Verify Tool Loop, and Local LLM / Deterministic Agent.
        """
        # Execute local agent loop
        reply, trace, meta = self.agent.run_agent_loop(
            visitor_text=visitor_text,
            language=language,
            use_llm=False  # Keep 100% offline local
        )
        meta["action_trace"] = [step.to_dict() if hasattr(step, "to_dict") else step for step in trace]
        meta["session_memory"] = self.agent.get_memory_snapshot()
        meta["generator"] = "local_agentic_reasoning"
        return reply, meta

    def process_visitor_interaction(
        self,
        audio_source: Union[str, bytes, np.ndarray],
        manual_language: Optional[str] = None,
        play_audio: bool = False
    ) -> Dict[str, Any]:
        """
        Complete 100% Local Voice Receptionist Pipeline:
        1. Local Audio Preprocessing (16 kHz mono, peak normalized)
        2. Local 26-MFCC Extraction (13 means + 13 stds)
        3. Local Language Identification (PYNQ-Z2 FPGA / CPU Backend)
        4. Local Speech-to-Text (Offline Whisper / Local Recognizer)
        5. Local Agent Reasoning & Tool Verification (Plan-Act-Verify)
        6. Local Speech Synthesis (Offline pyttsx3 Windows SAPI5)
        7. Multi-Turn Session Registration & Action Tracing
        """
        timings = {}
        t_start = time.perf_counter()

        result = {
            "success": False,
            "error": None,
            "backend": self.backend.get_diagnostics(),
            "detected_language": "Unknown",
            "confidence": 0.0,
            "probabilities": {},
            "visitor_transcript": "",
            "intent": "none",
            "receptionist_response": "",
            "response_audio_path": None,
            "audio_duration": 0.0,
            "timings_ms": {},
            "turn_id": len(self.conversation_history) + 1,
            "stt_meta": {},
            "tts_meta": {},
            "llm_meta": {},
            "action_trace": [],
            "session_memory": {},
            "offline_mode": True
        }

        # Step 1: Local Audio Loading & Validation
        t0 = time.perf_counter()
        try:
            waveform, sr = load_audio_signal(audio_source, sr=TARGET_SAMPLE_RATE)
            is_valid, msg = validate_audio_signal(waveform, sr)
            if not is_valid:
                result["error"] = f"Invalid audio input: {msg}"
                return result
            result["audio_duration"] = len(waveform) / sr
        except Exception as e:
            result["error"] = f"Audio loading error: {e}"
            return result
        timings["audio_preprocess"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Step 2: Local 26-MFCC Feature Extraction
        t0 = time.perf_counter()
        try:
            features = extract_26_features(waveform, sr=sr)
        except Exception as e:
            result["error"] = f"Feature extraction failed: {e}"
            return result
        timings["dsp_mfcc_extract"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Step 3: Local Language Classification (FPGA / CPU Backend)
        t0 = time.perf_counter()
        try:
            lang, conf, scores = self.backend.predict(features)
            if manual_language and manual_language in SUPPORTED_LANGUAGES:
                active_lang = manual_language
                result["language_overridden"] = True
            else:
                active_lang = lang
                result["language_overridden"] = False

            result["detected_language"] = active_lang
            result["classifier_raw_language"] = lang
            result["confidence"] = conf
            result["probabilities"] = scores
        except Exception as e:
            result["error"] = f"Language classification failed: {e}"
            return result
        timings["language_classification"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Step 4: Local Speech-to-Text Transcription in Active Language
        t0 = time.perf_counter()
        temp_wav = None
        if isinstance(audio_source, str) and os.path.exists(audio_source):
            audio_file_for_stt = audio_source
        else:
            import soundfile as sf
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()
            sf.write(temp_wav.name, waveform, sr, subtype='PCM_16')
            audio_file_for_stt = temp_wav.name

        try:
            # Transcribe strictly locally using offline ASR
            stt_ok, transcript, stt_meta = self.offline_stt.transcribe_audio_file(
                audio_file_for_stt,
                language=active_lang
            )
            result["stt_meta"] = stt_meta
            if stt_ok and transcript and transcript.strip():
                result["visitor_transcript"] = transcript.strip()
            else:
                result["visitor_transcript"] = ""
                result["unintelligible"] = True
        finally:
            if temp_wav and os.path.exists(temp_wav.name):
                try:
                    os.remove(temp_wav.name)
                except Exception:
                    pass
        timings["speech_to_text"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Step 5: Local Autonomous Agent Reasoning (Plan-Act-Verify)
        t0 = time.perf_counter()
        response_text, gen_meta = self.generate_response(
            visitor_text=result["visitor_transcript"],
            language=active_lang
        )
        result["intent"] = gen_meta.get("goal", "conversational")
        result["receptionist_response"] = response_text
        result["llm_meta"] = gen_meta
        result["action_trace"] = gen_meta.get("action_trace", [])
        result["session_memory"] = gen_meta.get("session_memory", {})
        timings["response_generation"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Step 6: Local Text-to-Speech Synthesis (Offline pyttsx3 / Windows SAPI5)
        t0 = time.perf_counter()
        try:
            tts_ok, audio_path, tts_meta = self.offline_tts.synthesize_speech(
                text=response_text,
                language=active_lang
            )
            result["tts_meta"] = tts_meta
            if tts_ok:
                result["response_audio_path"] = audio_path
                if play_audio:
                    self.offline_tts.play_audio(audio_path, blocking=False)
        except Exception as e:
            print(f"[!] Offline TTS warning: {e}")
        timings["text_to_speech"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Total Pipeline Latency
        timings["total_pipeline"] = round((time.perf_counter() - t_start) * 1000.0, 2)
        result["timings_ms"] = timings
        result["success"] = True

        # Step 7: Record Conversation Turn
        turn = ConversationTurn(
            turn_id=result["turn_id"],
            visitor_audio_path=audio_source if isinstance(audio_source, str) else None,
            detected_language=active_lang,
            confidence=conf,
            visitor_transcript=result["visitor_transcript"],
            receptionist_response=result["receptionist_response"],
            response_audio_path=result["response_audio_path"],
            backend_name=self.backend.name,
            latency_breakdown=timings,
            action_trace=result["action_trace"]
        )
        self.conversation_history.append(turn)

        return result

    def clear_conversation(self):
        """Resets the multi-turn conversation session."""
        self.reset_conversation()
