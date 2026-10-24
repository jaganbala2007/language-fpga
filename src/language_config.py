"""
Central Language and Voice Configuration Module
FPGA-Based Multilingual AI Receptionist

Maintains unified mapping for:
- Supported languages: English, Hindi, Tamil
- ISO Locales: en-IN, hi-IN, ta-IN
- Native scripts & display labels
- Sarvam AI Speech-to-Text (Saaras v4) models & options
- Sarvam AI Text-to-Speech (Bulbul v4 Flash) voice catalog & settings
- Fallback locale configurations
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class VoiceOption:
    speaker_id: str
    gender: str
    description: str
    is_default: bool = False


@dataclass
class LanguageProfile:
    name: str
    native_name: str
    iso_code: str
    script: str
    class_index: int  # 0: English, 1: Hindi, 2: Tamil
    stt_model: str = "saaras:v4"
    tts_model: str = "bulbul:v4-flash"
    default_speaker: str = "meera"
    available_speakers: List[VoiceOption] = field(default_factory=list)
    gtts_code: str = "en"
    google_stt_locale: str = "en-IN"
    sample_phrases: List[str] = field(default_factory=list)

    @property
    def iso_639_1(self) -> str:
        return self.gtts_code

    @property
    def code(self) -> str:
        return self.gtts_code



# Central Multilingual Configuration
SUPPORTED_LANGUAGES: Dict[str, LanguageProfile] = {
    "English": LanguageProfile(
        name="English",
        native_name="English (India)",
        iso_code="en-IN",
        script="Latin",
        class_index=0,
        stt_model="saaras:v4",
        tts_model="bulbul:v4-flash",
        default_speaker="arvind",
        available_speakers=[
            VoiceOption(speaker_id="arvind", gender="Male", description="Natural Indian English (Male)", is_default=True),
            VoiceOption(speaker_id="meera", gender="Female", description="Polite Indian English (Female)"),
            VoiceOption(speaker_id="amartya", gender="Male", description="Deep Professional English (Male)"),
            VoiceOption(speaker_id="pavithra", gender="Female", description="Warm Receptionist Voice (Female)"),
        ],
        gtts_code="en",
        google_stt_locale="en-IN",
        sample_phrases=[
            "Hello, where is the Engineering and FPGA Lab?",
            "What are the office visiting hours today?",
            "I have an appointment with Human Resources.",
            "Where can I get a visitor pass?"
        ]
    ),
    "Hindi": LanguageProfile(
        name="Hindi",
        native_name="हिन्दी",
        iso_code="hi-IN",
        script="Devanagari",
        class_index=1,
        stt_model="saaras:v4",
        tts_model="bulbul:v4-flash",
        default_speaker="meera",
        available_speakers=[
            VoiceOption(speaker_id="meera", gender="Female", description="प्राकृतिक स्पष्ट आवाज़ (महिला)", is_default=True),
            VoiceOption(speaker_id="arvind", gender="Male", description="पेशेवर गंभीर आवाज़ (पुरुष)"),
            VoiceOption(speaker_id="maithili", gender="Female", description="मधुर स्वागत आवाज़ (महिला)"),
            VoiceOption(speaker_id="amartya", gender="Male", description="औपचारिक आवाज़ (पुरुष)"),
        ],
        gtts_code="hi",
        google_stt_locale="hi-IN",
        sample_phrases=[
            "नमस्ते, इंजीनियरिंग और एफपीजीए लैब कहाँ है?",
            "कार्यालय का समय क्या है?",
            "मुझे एचआर विभाग में जाना है।",
            "विज़िटर पास कहाँ से मिलेगा?"
        ]
    ),
    "Tamil": LanguageProfile(
        name="Tamil",
        native_name="தமிழ்",
        iso_code="ta-IN",
        script="Tamil",
        class_index=2,
        stt_model="saaras:v4",
        tts_model="bulbul:v4-flash",
        default_speaker="pavithra",
        available_speakers=[
            VoiceOption(speaker_id="pavithra", gender="Female", description="தெளிவான தமிழ் வரவேற்பாளர் (பெண்)", is_default=True),
            VoiceOption(speaker_id="arvind", gender="Male", description="இயற்கையான தமிழ் குரல் (ஆண்)"),
            VoiceOption(speaker_id="meera", gender="Female", description="மரியாதையான வரவேற்பு குரல் (பெண்)"),
        ],
        gtts_code="ta",
        google_stt_locale="ta-IN",
        sample_phrases=[
            "வணக்கம், பொறியியல் மற்றும் FPGA ஆய்வகம் எங்கு உள்ளது?",
            "அலுவலக வேலை நேரம் என்ன?",
            "சேர்க்கை பிரிவு (Admissions) எங்கே இருக்கிறது?",
            "பார்வையாளர் அனுமதி அட்டை எங்கு பெற வேண்டும்?"
        ]
    )
}

LANGUAGE_LIST = ["English", "Hindi", "Tamil"]
INDEX_TO_LANGUAGE = {0: "English", 1: "Hindi", 2: "Tamil"}
LANGUAGE_TO_INDEX = {"English": 0, "Hindi": 1, "Tamil": 2}


def get_language_profile(lang_key: str) -> LanguageProfile:
    """Retrieves the LanguageProfile by name, ISO code, or alias."""
    if not lang_key:
        return SUPPORTED_LANGUAGES["English"]

    k = lang_key.strip()
    if k in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[k]

    for prof in SUPPORTED_LANGUAGES.values():
        if k.lower() == prof.name.lower() or k.lower() == prof.iso_code.lower() or k.lower() == prof.gtts_code.lower():
            return prof

    return SUPPORTED_LANGUAGES["English"]


def normalize_language_name(lang_key: str) -> str:
    """Returns canonical language name: 'English', 'Hindi', or 'Tamil'."""
    return get_language_profile(lang_key).name
