"""
Integration Tests: Multi-Turn Conversation History & Context
"""

import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.receptionist import MultilingualReceptionist


def test_multi_turn_session_tracking():
    receptionist = MultilingualReceptionist(backend_mode="cpu")
    receptionist.clear_conversation()

    assert len(receptionist.conversation_history) == 0

    # Turn 1: Process Greeting
    audio_path = "demo_audio/demo_en_welcome.wav"
    if os.path.exists(audio_path):
        res1 = receptionist.process_visitor_interaction(audio_path, play_audio=False)
        assert res1["success"]
        assert len(receptionist.conversation_history) == 1
        assert receptionist.conversation_history[0].turn_id == 1

        # Turn 2: Process Hours
        res2 = receptionist.process_visitor_interaction(audio_path, play_audio=False)
        assert res2["success"]
        assert len(receptionist.conversation_history) == 2
        assert receptionist.conversation_history[1].turn_id == 2

        # Clear session
        receptionist.clear_conversation()
        assert len(receptionist.conversation_history) == 0


def test_grounding_text_contains_core_facts():
    receptionist = MultilingualReceptionist(backend_mode="cpu")
    grounding = receptionist._grounding_text

    assert "Smart Tech Innovation Center" in grounding
    assert "Engineering & FPGA Lab" in grounding
    assert "Human Resources" in grounding
    assert "Visitor Registration" in grounding
