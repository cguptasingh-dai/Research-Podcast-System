"""
Shared configuration for both Researcher and Podcast agents.
Ensures consistent API key and LLM settings across all agents.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APIConfig:
    """Centralized API configuration for both agents."""

    @staticmethod
    def get_nvidia_api_key():
        """Get NVIDIA API key - tries multiple env var names for compatibility."""
        key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
        if not key:
            raise ValueError(
                "[ERROR] NVIDIA API key not found!\n"
                "   Set NVIDIA_NIM_API_KEY or NVIDIA_API_KEY in .env file\n"
                "   Example: NVIDIA_NIM_API_KEY=\"nvapi-...\""
            )
        return key

    @staticmethod
    def get_serper_api_key():
        """Get SERPER API key for web search."""
        key = os.getenv("SERPER_API_KEY")
        if not key:
            raise ValueError(
                "[ERROR] SERPER API key not found!\n"
                "   Set SERPER_API_KEY in .env file\n"
                "   Get key from: https://serper.dev/"
            )
        return key


class LLMConfig:
    """LLM configuration - Nemotron Ultra 550B with reasoning"""

    # NVIDIA NIM API settings
    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

    # Model - Nemotron Super 49B (fast + reliable, 550B was giving connection errors)
    NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

    # Model parameters (nemotron-super-49b). temperature 0.4 for focused,
    # deterministic output; top_p 0.95; max_tokens 4096.
    TEMPERATURE = 0.4
    TOP_P = 0.95
    MAX_TOKENS = 6000  # room for a comprehensive report with sub-sections
    FREQUENCY_PENALTY = 0.0
    PRESENCE_PENALTY = 0.0
    TIMEOUT = 150  # 2.5 minutes

    # Reasoning budget - enough for the agent to plan sequential tool calls.
    # Too low (e.g. 32) starves the ReAct loop and the agent dumps all tool calls
    # at once as one malformed batch (CrewAI rejects it → agent fabricates).
    REASONING_BUDGET = 256
    ENABLE_THINKING = True  # Keep thinking ON

    # Podcast agent settings — same sampling for consistency
    PODCAST_LLM_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
    PODCAST_TEMPERATURE = 0.4
    PODCAST_TOP_P = 0.95
    PODCAST_MAX_TOKENS = 4096


class FastModeConfig:
    """Fast mode config"""

    RESEARCH_ITERATIONS = 1
    QUESTIONS_COUNT = 10
    TEMPERATURE = 0.4   # Consistent with LLMConfig
    TOP_P = 0.9         # Consistent with LLMConfig
    MAX_TOKENS = 4096
    TIMEOUT = 180       # Consistent with LLMConfig


class AudioConfig:
    """Audio generation settings for natural voice output."""

    # TTS Settings for natural speech
    LANGUAGE = "en"

    # Different voices for Q&A variety
    QUESTION_VOICE = {
        "lang": "en",
        "tld": "com",  # US English
        "slow": False,  # Normal speed
    }

    ANSWER_VOICE = {
        "lang": "en",
        "tld": "co.uk",  # British English for variety
        "slow": False,  # Normal speed
    }

    # Pause between segments (milliseconds)
    PAUSE_SHORT = 400  # Between question and answer
    PAUSE_MEDIUM = 600  # Between Q&A pairs
    PAUSE_LONG = 800  # Between major sections

    # Text cleanup settings
    MAX_QUESTION_LENGTH = 250  # Characters
    MAX_ANSWER_LENGTH = 400  # Characters
    REMOVE_PUNCTUATION = True  # Remove excessive punctuation for natural speech
    NORMALIZE_WHITESPACE = True  # Clean up whitespace


# Ensure output directories exist
def ensure_directories():
    """Create required output directories."""
    directories = [
        Path("results"),
        Path("podcasts"),
        Path("uploads"),
        Path("reports"),
        Path("Report"),
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
