"""
Unit Tests: Central Language Configuration Module
"""

import os
import sys
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.language_config import (
    SUPPORTED_LANGUAGES,
    LANGUAGE_LIST,
    get_language_profile,
    normalize_language_name
)


def test_language_profiles_exist():
    assert "English" in SUPPORTED_LANGUAGES
    assert "Hindi" in SUPPORTED_LANGUAGES
    assert "Tamil" in SUPPORTED_LANGUAGES
    assert len(LANGUAGE_LIST) == 3


def test_iso_locales_and_scripts():
    en = get_language_profile("English")
    hi = get_language_profile("Hindi")
    ta = get_language_profile("Tamil")

    assert en.iso_code == "en-IN"
    assert hi.iso_code == "hi-IN"
    assert ta.iso_code == "ta-IN"

    assert en.script == "Latin"
    assert hi.script == "Devanagari"
    assert ta.script == "Tamil"


def test_sarvam_model_and_speaker_config():
    for lang in ["English", "Hindi", "Tamil"]:
        prof = get_language_profile(lang)
        assert prof.stt_model == "saaras:v4"
        assert prof.tts_model == "bulbul:v4-flash"
        assert len(prof.default_speaker) > 0
        assert len(prof.available_speakers) >= 2


def test_alias_and_case_insensitive_lookup():
    assert normalize_language_name("en-in") == "English"
    assert normalize_language_name("HI-IN") == "Hindi"
    assert normalize_language_name("ta-in") == "Tamil"
    assert normalize_language_name("english") == "English"
    assert normalize_language_name("tamil") == "Tamil"
