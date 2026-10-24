"""
Unit Tests: Sarvam AI Voice Model Stack Integration
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sarvam_voice_stack import (
    SarvamSaarasSTT,
    Sarvam105BLLM,
    SarvamBulbulTTS,
    get_sarvam_api_key
)


def test_api_key_resolution():
    with patch.dict(os.environ, {"SARVAM_API_KEY": "test_env_key_12345"}):
        assert get_sarvam_api_key() == "test_env_key_12345"
        assert get_sarvam_api_key("override_key_999") == "override_key_999"


def test_sarvam_stt_missing_key():
    stt = SarvamSaarasSTT(api_key=None)
    with patch.dict(os.environ, {}, clear=True):
        ok, text, meta = stt.transcribe("non_existent.wav", language="English")
        assert not ok
        assert meta.get("error") == "missing_api_key"


def test_sarvam_llm_mock_generation():
    llm = Sarvam105BLLM(api_key="mock_key")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "Hello! The Engineering Lab is on the 3rd Floor, Room 304."}}
        ]
    }

    with patch("requests.post", return_value=mock_response):
        ok, reply, meta = llm.generate_receptionist_reply(
            visitor_message="Where is the engineering lab?",
            language="English",
            knowledge_base_context="Engineering Lab: 3rd Floor, Room 304."
        )
        assert ok
        assert "Engineering Lab" in reply
        assert meta["provider"] == "sarvam_llm"


def test_sarvam_tts_mock_synthesis(tmp_path):
    tts = SarvamBulbulTTS(api_key="mock_key")
    out_file = str(tmp_path / "test_synth.wav")

    # Base64 encoded fake WAV payload
    fake_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "audios": [fake_b64]
    }

    with patch("requests.post", return_value=mock_response):
        ok, audio_path, meta = tts.synthesize(
            text="Welcome to Smart Tech Center",
            language="English",
            speaker="arvind",
            output_path=out_file
        )
        assert ok
        assert os.path.exists(audio_path)
        assert meta["provider"] == "sarvam_bulbul"
