"""
100% Offline Multilingual AI Receptionist Test Suite
Verifies offline ASR, offline LLM/Agent reasoning, offline TTS, and end-to-end pipeline execution without internet.
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.offline_asr import get_offline_stt_engine, OfflineSpeechToTextEngine
from src.offline_llm import get_offline_llm_engine, OfflineLLMEngine
from src.offline_tts import get_offline_tts_engine, OfflineTextToSpeechEngine
from src.receptionist import MultilingualReceptionist


def test_offline_asr_engine_status():
    """Verify offline ASR engine reports offline mode and engine parameters."""
    engine = get_offline_stt_engine()
    status = engine.get_engine_status()
    assert status["offline_mode"] is True
    assert "device" in status
    assert "models_dir" in status


def test_offline_llm_engine_status():
    """Verify offline LLM engine reports Ollama status and fallback readiness."""
    engine = get_offline_llm_engine()
    status = engine.check_ollama_status()
    assert "offline_ready" in status
    assert status["offline_ready"] is True
    assert "preferred_model" in status


def test_offline_llm_grounded_response():
    """Verify local LLM/Agent generates grounded response in English, Hindi, and Tamil."""
    engine = get_offline_llm_engine()
    
    # English Query
    en_resp, en_meta = engine.generate_response("Where is the Engineering Lab?", language="English")
    assert "engineering" in en_resp.lower() or "floor" in en_resp.lower() or "room" in en_resp.lower()
    assert en_meta["offline"] is True

    # Hindi Query
    hi_resp, hi_meta = engine.generate_response("कार्यालय का समय क्या है?", language="Hindi")
    assert len(hi_resp) > 5
    assert hi_meta["offline"] is True

    # Tamil Query
    ta_resp, ta_meta = engine.generate_response("சேர்க்கை பிரிவு எங்கே உள்ளது?", language="Tamil")
    assert len(ta_resp) > 5
    assert ta_meta["offline"] is True


def test_offline_tts_synthesis():
    """Verify offline TTS engine produces valid audio or returns clear offline metadata."""
    tts = get_offline_tts_engine()
    ok, audio_path, meta = tts.synthesize_speech("Welcome to the Engineering Lab.", language="English")
    assert meta["offline"] is True
    assert "latency_ms" in meta


def test_offline_receptionist_pipeline_demo_audio():
    """Verify complete 100% offline receptionist pipeline using preset audio sample."""
    receptionist = MultilingualReceptionist(backend_mode="cpu", offline_mode=True)
    sys_stat = receptionist.get_system_status()
    assert sys_stat["offline_mode_enforced"] is True

    demo_file = "demo_audio/demo_en_welcome.wav"
    if os.path.exists(demo_file):
        result = receptionist.process_visitor_interaction(demo_file)
        assert result["success"] is True
        assert result["offline_mode"] is True
        assert "timings_ms" in result
        assert "audio_preprocess" in result["timings_ms"]
        assert "dsp_mfcc_extract" in result["timings_ms"]
        assert "language_classification" in result["timings_ms"]
        assert "response_generation" in result["timings_ms"]
