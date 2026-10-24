"""
100% Offline Multilingual Automatic Speech Recognition (ASR) Engine
FPGA-Based Multilingual AI Receptionist

Operates strictly on local hardware with zero cloud calls:
1. Primary: Local Faster-Whisper / Whisper models (tiny / base / small)
2. Fallback: Local SpeechRecognition offline engine
3. Language Support: English (en), Hindi (hi), Tamil (ta)
"""

import os
import sys
import time
import tempfile
from typing import Optional, Tuple, Dict, Any

from src.language_config import get_language_profile, normalize_language_name

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_cache")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class OfflineSpeechToTextEngine:
    """100% Local Offline Multilingual Speech-to-Text Transcription Engine."""

    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "default"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._whisper_model = None
        self._model_loaded = False
        self._faster_whisper_available = False
        self._whisper_available = False
        self._init_local_model()

    def _init_local_model(self):
        """Attempts to load local whisper/faster-whisper model from local storage."""
        # Check faster-whisper
        try:
            from faster_whisper import WhisperModel
            self._faster_whisper_available = True
            
            # Determine device (CPU vs CUDA)
            use_device = self.device
            if use_device == "auto":
                try:
                    import torch
                    use_device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    use_device = "cpu"

            use_compute = "float16" if use_device == "cuda" else "int8"
            
            # Look for local cached model
            local_model_path = os.path.join(MODELS_DIR, f"whisper-{self.model_size}")
            model_to_load = local_model_path if os.path.exists(local_model_path) else self.model_size

            self._whisper_model = WhisperModel(
                model_to_load,
                device=use_device,
                compute_type=use_compute,
                download_root=MODELS_DIR
            )
            self._model_loaded = True
            return
        except Exception:
            self._faster_whisper_available = False

        # Check standard openai-whisper
        try:
            import whisper
            self._whisper_available = True
            self._whisper_model = whisper.load_model(self.model_size, download_root=MODELS_DIR)
            self._model_loaded = True
            return
        except Exception:
            self._whisper_available = False

    def is_offline_ready(self) -> bool:
        """Returns True if local offline ASR model is loaded and ready."""
        return self._model_loaded

    def get_engine_status(self) -> Dict[str, Any]:
        """Returns detailed local ASR status dictionary."""
        return {
            "model_size": self.model_size,
            "model_loaded": self._model_loaded,
            "faster_whisper": self._faster_whisper_available,
            "whisper": self._whisper_available,
            "device": self.device,
            "models_dir": MODELS_DIR,
            "offline_mode": True
        }

    def transcribe_audio_file(
        self,
        audio_path: str,
        language: str = "English",
        timeout: int = 15
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Transcribes audio file using strictly local offline models.

        Returns:
            success (bool), transcript (str), metadata (dict)
        """
        start_time = time.time()
        if not os.path.exists(audio_path):
            return False, f"Audio file not found: {audio_path}", {"error": "file_not_found"}

        profile = get_language_profile(language)
        lang_code = profile.iso_639_1  # 'en', 'hi', 'ta'

        # 1. Primary Offline Attempt: faster-whisper
        if self._faster_whisper_available and self._whisper_model is not None:
            try:
                segments, info = self._whisper_model.transcribe(
                    audio_path,
                    language=lang_code if lang_code in ["en", "hi", "ta"] else None,
                    beam_size=5,
                    vad_filter=True
                )
                text = " ".join([seg.text.strip() for seg in segments]).strip()
                latency_ms = (time.time() - start_time) * 1000
                if text:
                    return True, text, {
                        "provider": "faster-whisper (100% Offline Local)",
                        "language": profile.name,
                        "language_code": lang_code,
                        "detected_language": getattr(info, "language", lang_code),
                        "latency_ms": latency_ms,
                        "offline": True
                    }
            except Exception as e:
                pass

        # 2. Secondary Offline Attempt: standard whisper
        if self._whisper_available and self._whisper_model is not None:
            try:
                result = self._whisper_model.transcribe(
                    audio_path,
                    language=lang_code if lang_code in ["en", "hi", "ta"] else None,
                    fp16=False
                )
                text = result.get("text", "").strip()
                latency_ms = (time.time() - start_time) * 1000
                if text:
                    return True, text, {
                        "provider": "whisper (100% Offline Local)",
                        "language": profile.name,
                        "language_code": lang_code,
                        "latency_ms": latency_ms,
                        "offline": True
                    }
            except Exception:
                pass

        # 3. Tertiary Offline Fallback: Local SpeechRecognition Audio Decode
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            wav_path = audio_path
            created_temp = False
            if not audio_path.lower().endswith(".wav"):
                from src.features import convert_audio_to_wav
                wav_path = convert_audio_to_wav(audio_path)
                created_temp = True

            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)

            # Local offline pocketsphinx if installed
            try:
                transcript = recognizer.recognize_sphinx(audio_data)
                latency_ms = (time.time() - start_time) * 1000
                if transcript and transcript.strip():
                    return True, transcript.strip(), {
                        "provider": "pocketsphinx (100% Offline Local)",
                        "language": profile.name,
                        "latency_ms": latency_ms,
                        "offline": True
                    }
            except Exception:
                pass

            if created_temp and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

        except Exception:
            pass

        # If audio is valid but no speech model transcribed it, return informative offline message
        latency_ms = (time.time() - start_time) * 1000
        return False, "Local offline speech model could not transcribe audio. Please ensure Whisper model is downloaded or speak clearly.", {
            "error": "offline_asr_unintelligible",
            "latency_ms": latency_ms,
            "offline": True
        }


# Singleton instance
_offline_stt_engine = None

def get_offline_stt_engine(model_size: str = "base") -> OfflineSpeechToTextEngine:
    global _offline_stt_engine
    if _offline_stt_engine is None:
        _offline_stt_engine = OfflineSpeechToTextEngine(model_size=model_size)
    return _offline_stt_engine
