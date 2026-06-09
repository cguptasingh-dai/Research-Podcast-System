"""
Sarvam AI Configuration Settings
Customize voice, speed, quality, and other audio properties here
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Try multiple locations for .env file
possible_paths = [
    Path(__file__).parent.parent.parent / ".env",  # Podcast_agent/agents/.. (project root)
    Path.cwd() / ".env",  # Current working directory
    Path.home() / ".env",  # Home directory
]

env_found = False
for env_path in possible_paths:
    if env_path.exists():
        load_dotenv(env_path)
        env_found = True
        break

# If no .env file found, just try loading from environment
if not env_found:
    load_dotenv()


class SarvamConfig:
    """Sarvam AI Text-to-Speech Configuration"""

    # ========================================================================
    # API SETTINGS
    # ========================================================================

    API_URL = "https://api.sarvam.ai/text-to-speech/stream"
    API_KEY = os.getenv("SARVAM_API_KEY", "")
    TIMEOUT = 60  # seconds

    # ========================================================================
    # VOICE SETTINGS
    # ========================================================================

    # Language code (en-IN for Indian English)
    TARGET_LANGUAGE = "en-IN"

    # Single voice mode
    # Available speakers:
    # - "shubh"  : Male, natural Indian English (RECOMMENDED)
    # - "meera"  : Female, natural Indian English
    # - "kavya"  : Female, emotional/expressive
    SPEAKER = "shubh"

    # DUAL VOICE MODE (for audio_sarvam_dual.py)
    # Different voices for questions and answers
    SPEAKER_QUESTIONS = "shubh"  # Male voice for questions
    SPEAKER_ANSWERS = "meera"    # Female voice for answers (clear, friendly Indian English)

    # ========================================================================
    # MODEL SETTINGS
    # ========================================================================

    # Model versions:
    # - "bulbul:v3" : Latest, best quality (RECOMMENDED)
    # - "bulbul:v2" : Older version
    MODEL = "bulbul:v3"

    # ========================================================================
    # AUDIO PROPERTIES
    # ========================================================================

    # Speech pace (speed multiplier)
    # 0.8 = 20% slower
    # 1.0 = Natural speed (RECOMMENDED)
    # 1.1 = 10% faster
    # 1.3 = 30% faster
    PACE = 1.0

    # Sample rate in Hz
    # 16000 = Lower quality, faster
    # 22050 = Standard quality (RECOMMENDED)
    # 44100 = Higher quality, larger files
    SPEECH_SAMPLE_RATE = 22050

    # Output codec
    # "mp3"  = Direct MP3 (RECOMMENDED, no conversion)
    # "wav"  = WAV format (requires conversion to MP3)
    OUTPUT_CODEC = "mp3"

    # ========================================================================
    # PROCESSING OPTIONS
    # ========================================================================

    # Auto-clean/preprocess text
    # True = Sarvam cleans text (RECOMMENDED)
    # False = Use text as-is
    ENABLE_PREPROCESSING = True

    # ========================================================================
    # QUALITY PRESETS (Choose one)
    # ========================================================================

    # PRESET: HIGH QUALITY (Larger files, slower)
    HIGH_QUALITY = {
        "pace": 1.0,
        "speech_sample_rate": 44100,
        "enable_preprocessing": True,
    }

    # PRESET: BALANCED (RECOMMENDED)
    BALANCED = {
        "pace": 1.0,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,
    }

    # PRESET: FAST (Smaller files, lower quality)
    FAST = {
        "pace": 1.2,
        "speech_sample_rate": 16000,
        "enable_preprocessing": True,
    }

    # ========================================================================
    # SPEAKER PRESETS
    # ========================================================================

    SPEAKERS = {
        "shubh": {
            "name": "Shubh",
            "gender": "Male",
            "language": "Indian English",
            "quality": "Professional, natural",
            "use_case": "Podcasts, broadcasts"
        },
        "meera": {
            "name": "Meera",
            "gender": "Female",
            "language": "Indian English",
            "quality": "Clear, friendly",
            "use_case": "Educational, friendly tone"
        },
        "kavya": {
            "name": "Kavya",
            "gender": "Female",
            "language": "Indian English",
            "quality": "Emotional, expressive",
            "use_case": "Storytelling, emotional content"
        }
    }

    # ========================================================================
    # BUILD PAYLOAD (Called by audio_sarvam.py)
    # ========================================================================

    @classmethod
    def build_payload(cls, text: str, quality_preset: str = "BALANCED") -> dict:
        """
        Build Sarvam API payload with given text and quality preset

        Args:
            text: The text to convert to speech
            quality_preset: "HIGH_QUALITY", "BALANCED", or "FAST"

        Returns:
            Dictionary ready for Sarvam API POST request
        """

        # Get quality preset
        preset = getattr(cls, quality_preset, cls.BALANCED)

        # Build payload
        payload = {
            "text": text,
            "target_language_code": cls.TARGET_LANGUAGE,
            "speaker": cls.SPEAKER,
            "model": cls.MODEL,
            "pace": preset.get("pace", cls.PACE),
            "speech_sample_rate": preset.get("speech_sample_rate", cls.SPEECH_SAMPLE_RATE),
            "output_audio_codec": cls.OUTPUT_CODEC,
            "enable_preprocessing": preset.get("enable_preprocessing", cls.ENABLE_PREPROCESSING),
        }

        return payload

    @classmethod
    def get_api_headers(cls) -> dict:
        """Get HTTP headers for Sarvam API request"""
        if not cls.API_KEY:
            raise ValueError("SARVAM_API_KEY not configured in environment!")

        return {
            "api-subscription-key": cls.API_KEY,
            "Content-Type": "application/json"
        }

    @classmethod
    def validate(cls) -> bool:
        """Validate Sarvam configuration"""
        errors = []

        if not cls.API_KEY:
            errors.append("SARVAM_API_KEY not set in environment")

        if cls.TARGET_LANGUAGE not in ["en-IN", "en", "hi", "ta", "te", "ml"]:
            errors.append(f"Invalid language: {cls.TARGET_LANGUAGE}")

        if cls.SPEAKER not in cls.SPEAKERS:
            errors.append(f"Invalid speaker: {cls.SPEAKER}")

        if cls.PACE < 0.5 or cls.PACE > 2.0:
            errors.append(f"Invalid pace: {cls.PACE} (must be 0.5-2.0)")

        if cls.SPEECH_SAMPLE_RATE not in [16000, 22050, 44100]:
            errors.append(f"Invalid sample rate: {cls.SPEECH_SAMPLE_RATE}")

        if errors:
            print("Sarvam Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*70)
        print("  SARVAM AI CONFIGURATION")
        print("="*70)
        print(f"\n  API Endpoint: {cls.API_URL}")
        print(f"  API Key: {cls.API_KEY[:20]}...*** (hidden)")
        print(f"\n  Language: {cls.TARGET_LANGUAGE}")
        print(f"  Speaker: {cls.SPEAKER} (Male, Natural)")
        print(f"  Model: {cls.MODEL}")
        print(f"\n  Pace: {cls.PACE}x (Normal speed)")
        print(f"  Sample Rate: {cls.SPEECH_SAMPLE_RATE} Hz")
        print(f"  Output: {cls.OUTPUT_CODEC.upper()}")
        print(f"  Preprocessing: {'Enabled' if cls.ENABLE_PREPROCESSING else 'Disabled'}")
        print(f"\n" + "="*70 + "\n")


# Test configuration on import
if __name__ == "__main__":
    print("Sarvam AI Configuration Validator\n")

    if SarvamConfig.validate():
        print("SUCCESS: Sarvam configuration is valid!\n")
        SarvamConfig.print_config()
    else:
        print("\nERROR: Sarvam configuration is invalid!")
        print("Please fix the errors above and try again.")
