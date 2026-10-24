"""
Multilingual Text-To-Speech (TTS) and Natural Voice Playback Engine
FPGA-Based Multilingual AI Receptionist

Primary: Sarvam Bulbul v4 Flash (bulbul:v4-flash) — Natural Indian Voices
Fallback: gTTS (Google TTS) & pyttsx3 (Offline Speech Synthesizer)
Languages: English (en-IN), Hindi (hi-IN), Tamil (ta-IN)
"""

import os
import sys
import hashlib
from typing import Optional, Tuple, Dict, Any

from src.language_config import get_language_profile, normalize_language_name
from src.sarvam_voice_stack import SarvamBulbulTTS, get_sarvam_api_key

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


class TextToSpeechEngine:
    """Multilingual Text-to-Speech engine with Sarvam Bulbul + Offline Fallback."""

    def __init__(self, sarvam_api_key: Optional[str] = None):
        self.sarvam_tts = SarvamBulbulTTS(api_key=sarvam_api_key)
        self._init_pygame()
        self._offline_tts = None

    def set_api_key(self, api_key: str):
        self.sarvam_tts.api_key = api_key

    def _init_pygame(self):
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=512)
            self.has_pygame = True
        except Exception:
            self.has_pygame = False

    def _get_offline_engine(self):
        if self._offline_tts is None:
            try:
                import pyttsx3
                self._offline_tts = pyttsx3.init()
            except Exception:
                self._offline_tts = False
        return self._offline_tts

    def synthesize_speech(
        self,
        text: str,
        language: str = "English",
        speaker: Optional[str] = None,
        prefer_sarvam: bool = True,
        output_file: Optional[str] = None,
        pace: float = 1.0
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Synthesizes text into high-quality spoken audio using Sarvam Bulbul or fallbacks.

        Returns:
            success (bool), file_path (str), metadata (dict)
        """
        if not text or not text.strip():
            return False, "", {"error": "empty_text"}

        clean_text = text.strip()
        profile = get_language_profile(language)

        # 1. Primary Attempt: Sarvam Bulbul v4 Flash
        if prefer_sarvam and get_sarvam_api_key(self.sarvam_tts.api_key):
            sarvam_ok, audio_path, meta = self.sarvam_tts.synthesize(
                clean_text,
                language=profile.name,
                speaker=speaker or profile.default_speaker,
                model=profile.tts_model,
                output_path=output_file,
                pace=pace
            )
            if sarvam_ok and os.path.exists(audio_path):
                return True, audio_path, meta

        # 2. Secondary Fallback: gTTS
        cache_key = hashlib.md5(f"gtts_{profile.gtts_code}_{clean_text}".encode("utf-8")).hexdigest()
        gtts_output = output_file or os.path.join(CACHE_DIR, f"gtts_{profile.gtts_code}_{cache_key[:12]}.mp3")

        if os.path.exists(gtts_output) and os.path.getsize(gtts_output) > 200:
            return True, gtts_output, {
                "provider": "gTTS_cached",
                "language": profile.name,
                "fallback": True
            }

        try:
            from gtts import gTTS
            tts = gTTS(text=clean_text, lang=profile.gtts_code, slow=False)
            tts.save(gtts_output)
            return True, gtts_output, {
                "provider": "gTTS",
                "language": profile.name,
                "fallback": True
            }
        except Exception as gtts_err:
            # 3. Tertiary Fallback: pyttsx3 offline synthesizer
            offline = self._get_offline_engine()
            if offline:
                try:
                    wav_fallback = gtts_output.replace(".mp3", ".wav")
                    offline.save_to_file(clean_text, wav_fallback)
                    offline.runAndWait()
                    return True, wav_fallback, {
                        "provider": "pyttsx3_offline",
                        "language": profile.name,
                        "fallback": True,
                        "warning": str(gtts_err)
                    }
                except Exception as offline_err:
                    return False, "", {"error": f"gTTS failed ({gtts_err}) & pyttsx3 failed ({offline_err})"}
            return False, "", {"error": f"All TTS providers failed: {gtts_err}"}

    def play_audio(self, audio_file: str, blocking: bool = False) -> bool:
        """Plays the audio file through the speaker device."""
        if not os.path.exists(audio_file):
            return False

        if self.has_pygame:
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                if blocking:
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                return True
            except Exception as e:
                print(f"[!] Pygame playback failed: {e}")

        # Windows system fallback
        try:
            if sys.platform == "win32":
                import winsound
                if audio_file.endswith(".wav"):
                    flag = winsound.SND_FILENAME if blocking else winsound.SND_FILENAME | winsound.SND_ASYNC
                    winsound.PlaySound(audio_file, flag)
                    return True
        except Exception:
            pass

        return False

    def stop_audio(self):
        """Stops active audio playback immediately."""
        if self.has_pygame:
            try:
                import pygame
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
            except Exception:
                pass


# Singleton instance
_tts_engine = None

def get_tts_engine(sarvam_api_key: Optional[str] = None) -> TextToSpeechEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TextToSpeechEngine(sarvam_api_key=sarvam_api_key)
    elif sarvam_api_key:
        _tts_engine.set_api_key(sarvam_api_key)
    return _tts_engine
