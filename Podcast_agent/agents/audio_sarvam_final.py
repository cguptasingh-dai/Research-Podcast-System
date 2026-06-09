"""
FINAL WORKING Audio Generator - Sarvam AI
Single API call - NO FFmpeg needed - NO pydub needed
Direct MP3 output from Sarvam API
"""

import re
import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def clean_text(text):
    """Clean text for speech"""
    if not text:
        return ""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Limit length per segment
    if len(text) > 400:
        text = text[:400] + "."
    return text


def call_sarvam(text: str, speaker: str = "shubh") -> bytes:
    """
    Single Sarvam API call
    Returns: MP3 audio bytes
    """
    api_key = os.getenv("SARVAM_API_KEY", "").strip().strip('"').strip("'")
    if not api_key:
        print("  ERROR: SARVAM_API_KEY not found!")
        return b""

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "target_language_code": "en-IN",
        "speaker": speaker,
        "model": "bulbul:v3",
        "pace": 1.0,
        "speech_sample_rate": 22050,
        "output_audio_codec": "mp3",
        "enable_preprocessing": True
    }

    try:
        response = requests.post(
            "https://api.sarvam.ai/text-to-speech/stream",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120
        )

        if response.status_code != 200:
            print(f"  ERROR: Sarvam API {response.status_code}: {response.text[:100]}")
            return b""

        # Collect all chunks
        audio_data = b""
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                audio_data += chunk

        return audio_data

    except requests.exceptions.Timeout:
        print("  ERROR: Sarvam API timeout!")
        return b""
    except Exception as e:
        print(f"  ERROR: {str(e)[:80]}")
        return b""


def create_audio_sarvam_final(questions, answers, topic):
    """
    Create COMPLETE audio podcast using Sarvam AI

    Strategy: Build ONE complete script text and call API ONCE
    - No FFmpeg needed!
    - No pydub needed!
    - Single MP3 file output
    - Works without any extra dependencies

    Returns: path to MP3 file
    """
    try:
        safe_topic = re.sub(r'[^a-z0-9_]', '_', topic.lower())

        # Use ABSOLUTE path - project root/podcasts/
        # This ensures streamlit can find the file
        project_root = Path(__file__).parent.parent.parent  # A2a/
        podcasts_dir = project_root / "podcasts"
        podcasts_dir.mkdir(parents=True, exist_ok=True)
        mp3_path = podcasts_dir / f"{safe_topic}_podcast.mp3"

        print(f"\n{'='*70}")
        print(f"  [SARVAM AI] Creating audio podcast...")
        print(f"  Topic: {topic}")
        print(f"  Q&A Pairs: {len(questions)}")
        print(f"  Method: Single API call (no FFmpeg needed)")
        print(f"{'='*70}\n")

        # Check API key
        api_key = os.getenv("SARVAM_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            print("  ERROR: SARVAM_API_KEY not found in environment!")
            return ""

        print(f"  API Key: {api_key[:15]}... OK")

        # Build REAL PODCAST script
        print("  [BUILD] Building podcast script...")

        podcast_lines = []

        # Professional intro
        podcast_lines.append(
            f"Welcome to today's special podcast. "
            f"I am your host, and today we are diving deep into the fascinating world of {topic}. "
            f"We have a lot to cover, so let us get started right away."
        )

        # Transitions - NO "question" word, natural radio style
        transitions = [
            "So let us begin.",
            "That is fascinating.",
            "Here is something really interesting.",
            "Let us explore this further.",
            "This is a very important point.",
            "Here is what experts say about this.",
            "Let us dig deeper.",
            "Now consider this.",
            "Something many people wonder about,",
            "And to wrap things up,",
        ]

        valid_pairs = 0
        for i, (q, a) in enumerate(zip(questions, answers), 1):
            if not q or not a:
                continue

            q_clean = clean_text(str(q))
            a_clean = clean_text(str(a))

            if not q_clean or not a_clean:
                continue

            # Add transition
            transition = transitions[min(i-1, len(transitions)-1)]
            podcast_lines.append(transition)

            # Add Q&A naturally
            podcast_lines.append(f"{q_clean}")
            podcast_lines.append(f"{a_clean}")

            valid_pairs += 1
            print(f"  [{i}] Added Q&A: {q_clean[:50]}...")

        # Professional outro
        podcast_lines.append(
            f"And that wraps up our podcast on {topic}. "
            f"We covered {valid_pairs} key topics today. "
            f"Thank you so much for tuning in. "
            f"Stay curious, stay informed, and we will see you next time."
        )

        # Join with natural pauses
        full_text = " ... ".join(podcast_lines)

        print(f"\n  Total text: {len(full_text)} characters")
        print(f"  Valid Q&A pairs: {valid_pairs}")

        # Check text length - Sarvam has limits
        MAX_CHARS = 4500
        if len(full_text) > MAX_CHARS:
            print(f"  Text too long ({len(full_text)} chars), trimming to {MAX_CHARS}...")
            full_text = full_text[:MAX_CHARS] + "... Thank you for listening. That is all for today."

        # Make SINGLE API call
        print(f"\n  [SARVAM API] Calling API (single call)...")
        print(f"  Speaker: shubh (Indian English)")
        print(f"  Model: bulbul:v3")

        audio_bytes = call_sarvam(full_text, speaker="shubh")

        if not audio_bytes:
            print("  ERROR: Sarvam API returned no audio!")
            return ""

        print(f"  OK: Received {len(audio_bytes)} bytes")

        if len(audio_bytes) < 1000:
            print(f"  ERROR: Audio too small ({len(audio_bytes)} bytes) - likely empty!")
            return ""

        # Save directly to MP3 file
        print(f"\n  [SAVE] Saving MP3 file...")
        with open(mp3_path, 'wb') as f:
            f.write(audio_bytes)

        # Verify file
        file_size = mp3_path.stat().st_size
        if file_size == 0:
            print(f"  ERROR: MP3 file is empty!")
            return ""

        print(f"\n{'='*70}")
        print(f"  AUDIO PODCAST READY!")
        print(f"{'='*70}")
        print(f"  File:     {mp3_path.name}")
        print(f"  Format:   MP3 (Sarvam AI)")
        print(f"  Size:     {file_size / 1024:.1f} KB")
        print(f"  Q&A:      {valid_pairs} pairs")
        print(f"  Voice:    shubh (Indian English)")
        print(f"  Model:    bulbul:v3")
        print(f"  Path:     {mp3_path.absolute()}")
        print(f"{'='*70}\n")

        return str(mp3_path)

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
