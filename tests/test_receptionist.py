"""
Unit and Integration Tests: Receptionist Engine and Multilingual Interaction
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.receptionist import MultilingualReceptionist


def test_intent_detection():
    receptionist = MultilingualReceptionist(backend_mode="cpu")

    # English intents
    assert receptionist.detect_intent("Hello good morning", "English") == "greeting"
    assert receptionist.detect_intent("What are your office hours?", "English") == "hours"
    assert receptionist.detect_intent("Where is the engineering and fpga lab?", "English") == "directions_engineering"
    assert receptionist.detect_intent("I need to visit HR for an interview", "English") == "directions_hr"

    # Hindi intents
    assert receptionist.detect_intent("नमस्ते", "Hindi") == "greeting"
    assert receptionist.detect_intent("कार्यालय का समय क्या है", "Hindi") == "hours"
    assert receptionist.detect_intent("इंजीनियरिंग लैब कहाँ है", "Hindi") == "directions_engineering"

    # Tamil intents
    assert receptionist.detect_intent("வணக்கம்", "Tamil") == "greeting"
    assert receptionist.detect_intent("அலுவலக நேரம் என்ன", "Tamil") == "hours"
    assert receptionist.detect_intent("பொறியியல் ஆய்வகம் எங்கு உள்ளது", "Tamil") == "directions_engineering"


def test_receptionist_response_generation():
    receptionist = MultilingualReceptionist(backend_mode="cpu")

    resp_en, _ = receptionist.generate_response("hours", "English")
    assert "9:00 AM" in resp_en or "Monday" in resp_en

    resp_hi, _ = receptionist.generate_response("hours", "Hindi")
    assert "कार्यालय" in resp_hi or "सोमवार" in resp_hi

    resp_ta, _ = receptionist.generate_response("hours", "Tamil")
    assert "அலுவலகம்" in resp_ta or "திங்கள்" in resp_ta


def test_receptionist_end_to_end_on_recording():
    audio_path = "Recording.wav" if os.path.exists("Recording.wav") else "FPGA_Multilingual_Receptionist/Recording.wav"
    if not os.path.exists(audio_path):
        pytest.skip("Recording.wav not found for end-to-end test")

    receptionist = MultilingualReceptionist(backend_mode="auto")
    res = receptionist.process_visitor_interaction(audio_path, play_audio=False)

    assert res["success"]
    assert res["detected_language"] in ["English", "Hindi", "Tamil"]
    assert res["confidence"] > 0.50
    assert len(res["receptionist_response"]) > 0
