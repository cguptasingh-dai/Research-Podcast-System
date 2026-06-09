"""
DUAL VOICE Audio Generator - Sarvam AI
Host (Male - shubh) asks questions
Expert (Female - different voice) answers
Uses Python built-in wave module - NO FFmpeg needed!
"""

import re
import os
import io
import wave
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Cache API key and headers (read once, reuse for all calls)
_SARVAM_API_KEY = None
_SARVAM_HEADERS = None
_CACHED_SPEAKERS = None  # Cache (host, expert) speakers after first discovery

def _get_sarvam_headers():
    """Get cached Sarvam API headers - read key once."""
    global _SARVAM_API_KEY, _SARVAM_HEADERS
    if _SARVAM_HEADERS is None:
        _SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip().strip('"').strip("'")
        _SARVAM_HEADERS = {
            "api-subscription-key": _SARVAM_API_KEY,
            "Content-Type": "application/json"
        }
    return _SARVAM_HEADERS, _SARVAM_API_KEY


def clean_text(text):
    """Clean text for speech"""
    if not text:
        return ""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 450:
        text = text[:450] + "."
    return text


def get_valid_speakers():
    """
    Find valid speakers - cached after first discovery.
    Returns (host_speaker, expert_speaker)
    """
    global _CACHED_SPEAKERS
    if _CACHED_SPEAKERS is not None:
        print(f"  [CACHE] Using cached speakers: host={_CACHED_SPEAKERS[0]}, expert={_CACHED_SPEAKERS[1]}")
        return _CACHED_SPEAKERS

    headers, api_key = _get_sarvam_headers()
    # Valid bulbul:v3 speakers (confirmed working)
    male_candidates   = ["rahul", "rohan", "aditya", "ashutosh", "amit", "dev"]
    female_candidates = ["meera", "kavya", "priya", "neha", "pooja", "simran", "ritu", "isha"]

    host_speaker   = None
    expert_speaker = None

    def _test_speaker(speaker):
        payload = {
            "text": "Hello.",
            "target_language_code": "en-IN",
            "speaker": speaker,
            "model": "bulbul:v3",
            "pace": 1.0,
            "speech_sample_rate": 22050,
            "output_audio_codec": "wav",
            "enable_preprocessing": True
        }
        resp = requests.post(
            "https://api.sarvam.ai/text-to-speech/stream",
            headers=headers,
            json=payload,
            timeout=10
        )
        return resp.status_code == 200 and len(resp.content) > 100

    for speaker in male_candidates:
        try:
            if _test_speaker(speaker):
                host_speaker = speaker
                print(f"  OK: Found male host speaker: {speaker}")
                break
            else:
                print(f"  SKIP: {speaker} not valid")
        except Exception as e:
            print(f"  SKIP: {speaker} error: {str(e)[:30]}")

    for speaker in female_candidates:
        try:
            if _test_speaker(speaker):
                expert_speaker = speaker
                print(f"  OK: Found female expert speaker: {speaker}")
                break
            else:
                print(f"  SKIP: {speaker} not valid")
        except Exception as e:
            print(f"  SKIP: {speaker} error: {str(e)[:30]}")

    if not host_speaker:
        print("  WARN: No male speaker found, using rahul as fallback")
        host_speaker = "rahul"
    if not expert_speaker:
        print("  WARN: No female speaker found, using meera as fallback")
        expert_speaker = "meera"

    _CACHED_SPEAKERS = (host_speaker, expert_speaker)
    return _CACHED_SPEAKERS


def call_sarvam_wav(text: str, speaker: str, pace: float = 1.0) -> bytes:
    """
    Call Sarvam API - returns WAV audio bytes.
    Uses cached headers (no env read per call).
    """
    headers, api_key = _get_sarvam_headers()
    if not api_key:
        print(f"    ERROR: SARVAM_API_KEY not found!")
        return b""

    payload = {
        "text": text,
        "target_language_code": "en-IN",
        "speaker": speaker,
        "model": "bulbul:v3",
        "pace": pace,
        "speech_sample_rate": 22050,
        "output_audio_codec": "wav",
        "enable_preprocessing": True
    }

    try:
        response = requests.post(
            "https://api.sarvam.ai/text-to-speech/stream",
            headers=headers,
            json=payload,
            stream=True,
            timeout=60
        )

        if response.status_code != 200:
            print(f"    ERROR: Sarvam API {response.status_code}")
            if response.text:
                print(f"    Response: {response.text[:200]}")
            return b""

        audio_data = b""
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                audio_data += chunk

        return audio_data

    except Exception as e:
        print(f"    ERROR: {str(e)[:60]}")
        return b""


def create_silence_wav(duration_ms: int, sample_rate: int = 22050,
                       channels: int = 1, sampwidth: int = 2) -> bytes:
    """Create silent WAV using Python built-in wave module"""
    num_frames = int(sample_rate * duration_ms / 1000)
    silence_data = b'\x00' * (num_frames * channels * sampwidth)

    output = io.BytesIO()
    with wave.open(output, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(silence_data)

    return output.getvalue()


def combine_wav_segments(wav_segments: list) -> bytes:
    """
    Combine multiple WAV segments into ONE file
    Uses Python built-in wave module - NO FFmpeg needed!
    """
    all_frames = []
    wave_params = None

    for i, segment in enumerate(wav_segments):
        if not segment or len(segment) < 44:
            continue
        try:
            buf = io.BytesIO(segment)
            with wave.open(buf) as wf:
                if wave_params is None:
                    # Store parameters from first valid segment
                    wave_params = {
                        'nchannels': wf.getnchannels(),
                        'sampwidth': wf.getsampwidth(),
                        'framerate': wf.getframerate(),
                    }
                frames = wf.readframes(wf.getnframes())
                all_frames.append(frames)
        except Exception as e:
            print(f"    WARN: Skipping segment {i}: {str(e)[:40]}")
            continue

    if not wave_params or not all_frames:
        return b""

    # Combine all raw PCM frames
    combined_frames = b"".join(all_frames)

    # Write to output WAV
    output = io.BytesIO()
    with wave.open(output, 'wb') as out_wf:
        out_wf.setnchannels(wave_params['nchannels'])
        out_wf.setsampwidth(wave_params['sampwidth'])
        out_wf.setframerate(wave_params['framerate'])
        out_wf.writeframes(combined_frames)

    return output.getvalue()


def create_audio_sarvam_dual_voice(questions, answers, topic):
    """
    Create DUAL VOICE podcast
    Host asks questions, Expert answers
    Uses Python wave module - NO FFmpeg needed!
    """
    try:
        safe_topic = re.sub(r'[^a-z0-9_]', '_', topic.lower())
        project_root = Path(__file__).parent.parent.parent
        podcasts_dir = project_root / "podcasts"
        podcasts_dir.mkdir(parents=True, exist_ok=True)
        output_path = podcasts_dir / f"{safe_topic}_podcast.wav"

        print(f"\n{'='*70}")
        print(f"  [DUAL VOICE] Creating podcast...")
        print(f"  Topic: {topic}")
        print(f"  Q&A Pairs: {len(questions)}")
        print(f"{'='*70}\n")

        api_key = os.getenv("SARVAM_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            print("  ERROR: SARVAM_API_KEY not found!")
            return ""

        # Find valid speakers
        print("  [SPEAKERS] Finding available voices...")
        host_speaker, expert_speaker = get_valid_speakers()
        print(f"  Host (Questions):  {host_speaker} (pace: 1.0)")
        print(f"  Expert (Answers):  {expert_speaker} (pace: 0.95)")

        # Collect all WAV segments
        all_segments = []
        SHORT_PAUSE = 500   # ms between Q and A
        LONG_PAUSE  = 800   # ms between Q&A pairs

        # INTRO (Host voice)
        print("\n  [INTRO] Generating intro...")
        intro_text = (
            f"Welcome to today's special podcast. "
            f"I am your host, and joining me is our expert guest. "
            f"Today we are exploring the topic of {topic}. "
            f"Let us dive right in."
        )
        intro_bytes = call_sarvam_wav(intro_text, host_speaker, pace=1.0)
        if intro_bytes:
            all_segments.append(intro_bytes)
            all_segments.append(create_silence_wav(LONG_PAUSE))
            print(f"    OK: Intro ({len(intro_bytes):,} bytes)")

        # Q&A transitions - natural radio style
        transitions = [
            "So let us begin.",
            "That is fascinating.",
            "Interesting.",
            "Moving forward,",
            "Here is something important.",
            "I wanted to explore this further.",
            "Great point.",
            "Let us talk about something else.",
            "Many people wonder about this.",
            "And to wrap things up,",
        ]

        # Build list of valid pairs first
        valid_qa = []
        for i, (q, a) in enumerate(zip(questions, answers), 1):
            if not q or not a:
                continue
            q_clean = clean_text(str(q))
            a_clean = clean_text(str(a))
            if q_clean and a_clean:
                transition = transitions[min(i-1, len(transitions)-1)]
                valid_qa.append((i, f"{transition} {q_clean}", a_clean))

        if not valid_qa:
            print("  ERROR: No valid Q&A pairs!")
            return ""

        # PARALLEL TTS: submit all Q+A calls concurrently (5x faster than sequential)
        print(f"\n  [PARALLEL TTS] Submitting {len(valid_qa)*2} TTS calls concurrently...")
        MAX_TTS_WORKERS = 6  # Sarvam allows concurrent requests

        # Build all tasks: (index, text, speaker, pace, label)
        tts_tasks = []
        for i, q_text, a_text in valid_qa:
            tts_tasks.append((f"Q{i}", q_text, host_speaker, 1.0))
            tts_tasks.append((f"A{i}", a_text, expert_speaker, 0.95))

        # Execute in parallel, collect results keyed by label
        tts_results = {}
        with ThreadPoolExecutor(max_workers=MAX_TTS_WORKERS) as pool:
            future_map = {
                pool.submit(call_sarvam_wav, text, speaker, pace): label
                for label, text, speaker, pace in tts_tasks
            }
            for future in as_completed(future_map):
                label = future_map[future]
                try:
                    tts_results[label] = future.result()
                    size = len(tts_results[label])
                    print(f"    {label}: OK ({size:,} bytes)")
                except Exception as e:
                    print(f"    {label}: FAILED - {str(e)[:50]}")
                    tts_results[label] = b""

        # Reassemble segments IN ORDER
        valid_pairs = 0
        for i, q_text, a_text in valid_qa:
            q_bytes = tts_results.get(f"Q{i}", b"")
            a_bytes = tts_results.get(f"A{i}", b"")
            if q_bytes:
                all_segments.append(q_bytes)
                all_segments.append(create_silence_wav(SHORT_PAUSE))
            if a_bytes:
                all_segments.append(a_bytes)
                all_segments.append(create_silence_wav(LONG_PAUSE))
            if q_bytes or a_bytes:
                valid_pairs += 1

        if valid_pairs == 0:
            print("  ERROR: No valid Q&A pairs!")
            return ""

        # OUTRO (Host voice)
        print(f"\n  [OUTRO] Generating outro...")
        outro_text = (
            f"And that concludes our podcast on {topic}. "
            f"We covered {valid_pairs} key topics today. "
            f"Thank you for listening. See you next time!"
        )
        outro_bytes = call_sarvam_wav(outro_text, host_speaker, pace=1.0)
        if outro_bytes:
            all_segments.append(outro_bytes)
            print(f"    OK: Outro ({len(outro_bytes):,} bytes)")

        # COMBINE using Python wave module
        print(f"\n  [COMBINE] Merging {len(all_segments)} segments...")
        combined_wav = combine_wav_segments(all_segments)

        if not combined_wav:
            print("  ERROR: Failed to combine segments!")
            return ""

        # Save WAV file
        print(f"  [SAVE] Saving WAV file...")
        with open(output_path, 'wb') as f:
            f.write(combined_wav)

        file_size = output_path.stat().st_size
        if file_size == 0:
            print("  ERROR: Output file is empty!")
            return ""

        # Get duration
        try:
            with wave.open(str(output_path)) as wf:
                duration = wf.getnframes() / wf.getframerate()
        except:
            duration = 0

        print(f"\n{'='*70}")
        print(f"  DUAL VOICE PODCAST READY!")
        print(f"{'='*70}")
        print(f"  File:     {output_path.name}")
        print(f"  Format:   WAV")
        print(f"  Size:     {file_size/(1024*1024):.2f} MB")
        print(f"  Duration: {duration:.1f}s ({int(duration//60)}m {int(duration%60)}s)")
        print(f"  Q&A:      {valid_pairs} pairs")
        print(f"  Host:     {host_speaker} (Questions)")
        print(f"  Expert:   {expert_speaker} (Answers)")
        print(f"  Path:     {output_path.absolute()}")
        print(f"{'='*70}\n")

        return str(output_path)

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
