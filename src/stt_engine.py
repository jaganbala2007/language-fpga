"""
Multilingual Speech Recognition (STT) Engine
FPGA-Based Multilingual AI Receptionist

Primary: Sarvam Saaras v4 (saaras:v4) — Indian language acoustic model
Fallback: Google Speech Recognition & Sphinx
Languages: English (en-IN), Hindi (hi-IN), Tamil (ta-IN)
"""

import os
import sys
import tempfile
from typing import Optional, Tuple, Dict, Any
import numpy as np
import speech_recognition as sr

from src.language_config import get_language_profile, normalize_language_name
from src.sarvam_voice_stack import SarvamSaarasSTT, get_sarvam_api_key


class SpeechToTextEngine:
    """Multilingual Speech-to-Text transcription engine with Sarvam Saaras + Offline Fallback."""

    def __init__(self, sarvam_api_key: Optional[str] = None):
        self.sarvam_stt = SarvamSaarasSTT(api_key=sarvam_api_key)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def set_api_key(self, api_key: str):
        self.sarvam_stt.api_key = api_key

    def transcribe_audio_file(
        self,
        audio_path: str,
        language: str = "English",
        prefer_sarvam: bool = True,
        timeout: int = 12
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Transcribes an audio file into text using Sarvam Saaras v4 or Google STT fallback.

        Returns:
            success (bool), transcript (str), metadata (dict)
        """
        if not os.path.exists(audio_path):
            return False, f"Audio file not found: {audio_path}", {"error": "file_not_found"}

        profile = get_language_profile(language)

        # 1. Primary Attempt: Sarvam Saaras v4
        if prefer_sarvam and get_sarvam_api_key(self.sarvam_stt.api_key):
            sarvam_ok, transcript, meta = self.sarvam_stt.transcribe(
                audio_path,
                language=profile.name,
                model="saaras:v4",
                timeout=timeout
            )
            if sarvam_ok and transcript.strip():
                return True, transcript, meta

        # 2. Convert to standard 16-bit PCM WAV if needed for local recognizer
        wav_path = audio_path
        created_temp = False
        if not audio_path.lower().endswith(".wav"):
            from src.features import convert_audio_to_wav
            wav_path = convert_audio_to_wav(audio_path)
            created_temp = True

        # 3. Secondary Fallback: Google Speech Recognition
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)

            transcript = self.recognizer.recognize_google(
                audio_data,
                language=profile.google_stt_locale,
                show_all=False
            )
            return True, str(transcript), {
                "provider": "google_speech_recognition",
                "language": profile.name,
                "locale": profile.google_stt_locale,
                "fallback": True
            }

        except sr.UnknownValueError:
            return False, "Speech was unclear or unintelligible.", {
                "error": "unintelligible_speech",
                "locale": profile.google_stt_locale
            }
        except sr.RequestError as e:
            return False, f"Speech recognition service unavailable: {e}", {
                "error": "network_or_api_error"
            }
        except Exception as e:
            return False, f"Transcription failed: {e}", {"error": str(e)}
        finally:
            if created_temp and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    def record_microphone(
        self,
        duration_sec: float = 4.0,
        sample_rate: int = 16000
    ) -> Tuple[bool, np.ndarray, str]:
        """Captures audio from default microphone using sounddevice."""
        try:
            import sounddevice as sd
        except ImportError:
            return False, np.array([]), "sounddevice library not installed."

        try:
            frames = int(duration_sec * sample_rate)
            audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            waveform = audio.flatten()

            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()

            import soundfile as sf
            sf.write(temp_wav.name, waveform, sample_rate, subtype='PCM_16')
            return True, waveform, temp_wav.name
        except Exception as e:
            return False, np.array([]), f"Microphone recording failed: {e}"


# Singleton instance
_stt_engine = None

def get_stt_engine(sarvam_api_key: Optional[str] = None) -> SpeechToTextEngine:
    global _stt_engine
    if _stt_engine is None:
        _stt_engine = SpeechToTextEngine(sarvam_api_key=sarvam_api_key)
    elif sarvam_api_key:
        _stt_engine.set_api_key(sarvam_api_key)
    return _stt_engine
