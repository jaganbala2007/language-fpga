"""
Sarvam AI Voice Model Stack Integration
FPGA-Based Multilingual AI Receptionist

Integrates:
1. Speech-to-Text: Sarvam Saaras v4 (saaras:v4)
2. Response Generation: Sarvam-105B Conversational LLM (sarvam-105b)
3. Text-to-Speech: Sarvam Bulbul v4 Flash (bulbul:v4-flash)

Supports:
- English (en-IN), Hindi (hi-IN), Tamil (ta-IN)
- Automatic API key resolution (Environment / Session / Argument)
- Graceful error handling and offline fallbacks
"""

import os
import sys
import json
import base64
import hashlib
import time
import tempfile
from typing import Optional, Dict, Any, Tuple, List
import requests

from src.language_config import get_language_profile, normalize_language_name

# Default API endpoints
SARVAM_BASE_URL = "https://api.sarvam.ai"
SARVAM_STT_ENDPOINT = f"{SARVAM_BASE_URL}/speech-to-text"
SARVAM_CHAT_ENDPOINT = f"{SARVAM_BASE_URL}/v1/chat/completions"
SARVAM_TTS_ENDPOINT = f"{SARVAM_BASE_URL}/text-to-speech"

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def get_sarvam_api_key(override_key: Optional[str] = None) -> Optional[str]:
    """Resolves Sarvam API key from parameter, environment, or .env file."""
    if override_key and override_key.strip():
        return override_key.strip()

    # Check environment variable
    env_key = os.environ.get("SARVAM_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # Check .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("SARVAM_API_KEY=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val and "your_sarvam_api_key" not in val:
                            return val
        except Exception:
            pass

    return None


class SarvamSaarasSTT:
    """Sarvam Saaras Speech-to-Text Client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def transcribe(
        self,
        audio_file_path: str,
        language: str = "English",
        model: str = "saaras:v4",
        timeout: int = 15
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Transcribes audio using Sarvam Saaras API.

        Returns:
            success (bool), transcript (str), metadata (dict)
        """
        resolved_key = get_sarvam_api_key(self.api_key)
        if not resolved_key:
            return False, "", {"error": "missing_api_key", "message": "SARVAM_API_KEY not configured"}

        if not os.path.exists(audio_file_path):
            return False, "", {"error": "file_not_found", "message": f"File not found: {audio_file_path}"}

        profile = get_language_profile(language)
        headers = {
            "api-subscription-key": resolved_key
        }

        try:
            with open(audio_file_path, "rb") as audio_file:
                files = {
                    "file": (os.path.basename(audio_file_path), audio_file, "audio/wav")
                }
                data = {
                    "model": model,
                    "language_code": profile.iso_code,
                    "with_diarization": "false"
                }

                t0 = time.perf_counter()
                response = requests.post(
                    SARVAM_STT_ENDPOINT,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=timeout
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0

            if response.status_code == 200:
                res_json = response.json()
                transcript = res_json.get("transcript", "").strip()
                return True, transcript, {
                    "provider": "sarvam_saaras",
                    "model": model,
                    "language": profile.name,
                    "iso_code": profile.iso_code,
                    "latency_ms": round(latency_ms, 2),
                    "raw_response": res_json
                }
            elif response.status_code == 404 or "model" in response.text.lower():
                # Fallback to saaras:v1 if v4 is specific to enterprise tier
                if model != "saaras:v1":
                    return self.transcribe(audio_file_path, language=language, model="saaras:v1", timeout=timeout)
                return False, "", {"error": "api_error", "status_code": response.status_code, "message": response.text}
            else:
                return False, "", {
                    "error": "api_error",
                    "status_code": response.status_code,
                    "message": response.text
                }

        except requests.exceptions.Timeout:
            return False, "", {"error": "timeout", "message": f"Sarvam STT timed out after {timeout}s"}
        except Exception as e:
            return False, "", {"error": "network_exception", "message": str(e)}


class Sarvam105BLLM:
    """Sarvam-105B Conversational Chat Client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def generate_receptionist_reply(
        self,
        visitor_message: str,
        language: str,
        knowledge_base_context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: str = "sarvam-105b",
        timeout: int = 15
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Generates conversational grounded response in the visitor's native language.

        Returns:
            success (bool), reply_text (str), metadata (dict)
        """
        resolved_key = get_sarvam_api_key(self.api_key)
        if not resolved_key:
            return False, "", {"error": "missing_api_key", "message": "SARVAM_API_KEY not configured"}

        profile = get_language_profile(language)

        system_prompt = f"""You are a warm, polite, and efficient AI Receptionist at the front desk.
You are conversing with a visitor speaking in {profile.name} ({profile.native_name}).

CRITICAL INSTRUCTIONS:
1. Respond EXCLUSIVELY in natural, fluent {profile.name} using the proper native script ({profile.script}).
2. Be concise (1 to 3 friendly sentences), clear, and professional.
3. Use the verified INSTITUTION KNOWLEDGE BASE below as your single source of truth for facts.
4. If the visitor asks about a department, room, timing, or person NOT in the knowledge base, politely state that you do not have that specific information configured and direct them to front desk staff. NEVER invent or hallucinate unverified details.
5. Provide helpful directions and context when relevant.

--- VERIFIED INSTITUTION KNOWLEDGE BASE ---
{knowledge_base_context}
------------------------------------------
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Append previous conversation history for multi-turn continuity
        if conversation_history:
            for turn in conversation_history[-6:]:  # Keep last 6 messages
                role = "user" if turn.get("role") in ["user", "visitor"] else "assistant"
                messages.append({"role": role, "content": turn.get("content", "")})

        messages.append({"role": "user", "content": visitor_message})

        headers = {
            "api-subscription-key": resolved_key,
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 300
        }

        try:
            t0 = time.perf_counter()
            response = requests.post(
                SARVAM_CHAT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if response.status_code == 200:
                res_json = response.json()
                reply = res_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return True, reply, {
                    "provider": "sarvam_llm",
                    "model": model,
                    "language": profile.name,
                    "latency_ms": round(latency_ms, 2)
                }
            elif response.status_code == 404 or "model" in response.text.lower():
                # Fallback to sarvam-2b if 105b is unavailable
                if model != "sarvam-2b":
                    return self.generate_receptionist_reply(
                        visitor_message, language, knowledge_base_context,
                        conversation_history=conversation_history,
                        model="sarvam-2b", timeout=timeout
                    )
                return False, "", {"error": "api_error", "status_code": response.status_code, "message": response.text}
            else:
                return False, "", {
                    "error": "api_error",
                    "status_code": response.status_code,
                    "message": response.text
                }

        except requests.exceptions.Timeout:
            return False, "", {"error": "timeout", "message": f"Sarvam-105B LLM timed out after {timeout}s"}
        except Exception as e:
            return False, "", {"error": "network_exception", "message": str(e)}


class SarvamBulbulTTS:
    """Sarvam Bulbul Text-to-Speech Client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def synthesize(
        self,
        text: str,
        language: str = "English",
        speaker: Optional[str] = None,
        model: str = "bulbul:v4-flash",
        output_path: Optional[str] = None,
        pace: float = 1.0,
        timeout: int = 15
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Synthesizes text into natural Indian voice audio using Sarvam Bulbul API.

        Returns:
            success (bool), audio_file_path (str), metadata (dict)
        """
        if not text or not text.strip():
            return False, "", {"error": "empty_text"}

        clean_text = text.strip()
        profile = get_language_profile(language)
        chosen_speaker = speaker or profile.default_speaker

        # Check local audio cache first
        cache_key = hashlib.md5(f"sarvam_{profile.iso_code}_{chosen_speaker}_{clean_text}".encode("utf-8")).hexdigest()
        if output_path is None:
            output_path = os.path.join(CACHE_DIR, f"sarvam_tts_{profile.iso_code}_{cache_key[:12]}.wav")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
            return True, output_path, {
                "provider": "sarvam_bulbul",
                "cached": True,
                "speaker": chosen_speaker,
                "language": profile.name
            }

        resolved_key = get_sarvam_api_key(self.api_key)
        if not resolved_key:
            return False, "", {"error": "missing_api_key", "message": "SARVAM_API_KEY not configured"}

        headers = {
            "api-subscription-key": resolved_key,
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": [clean_text],
            "target_language_code": profile.iso_code,
            "speaker": chosen_speaker,
            "model": model,
            "pace": pace,
            "speech_sample_rate": 24000,
            "enable_preprocessing": True
        }

        try:
            t0 = time.perf_counter()
            response = requests.post(
                SARVAM_TTS_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if response.status_code == 200:
                res_json = response.json()
                audios = res_json.get("audios", [])
                if not audios:
                    return False, "", {"error": "no_audio_in_response", "message": "Empty audio array returned"}

                raw_b64 = audios[0]
                audio_bytes = base64.b64decode(raw_b64)

                with open(output_path, "wb") as f:
                    f.write(audio_bytes)

                return True, output_path, {
                    "provider": "sarvam_bulbul",
                    "model": model,
                    "speaker": chosen_speaker,
                    "language": profile.name,
                    "iso_code": profile.iso_code,
                    "latency_ms": round(latency_ms, 2),
                    "cached": False
                }
            elif response.status_code == 404 or "model" in response.text.lower():
                # Fallback to bulbul:v1 or bulbul:v2
                if model != "bulbul:v1":
                    return self.synthesize(
                        text, language=language, speaker=speaker,
                        model="bulbul:v1", output_path=output_path, pace=pace, timeout=timeout
                    )
                return False, "", {"error": "api_error", "status_code": response.status_code, "message": response.text}
            else:
                return False, "", {
                    "error": "api_error",
                    "status_code": response.status_code,
                    "message": response.text
                }

        except requests.exceptions.Timeout:
            return False, "", {"error": "timeout", "message": f"Sarvam Bulbul TTS timed out after {timeout}s"}
        except Exception as e:
            return False, "", {"error": "network_exception", "message": str(e)}
