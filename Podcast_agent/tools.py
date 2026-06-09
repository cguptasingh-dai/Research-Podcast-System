"""
Podcast Agent Tools
===================
LangChain @tool decorated functions for the podcast pipeline.
Each tool is independently testable and validates its output.

Tools:
  - extract_facts_tool       : Extract structured facts from the research report
  - generate_qa_tool         : Generate Q&A pairs from facts
  - validate_qa_tool         : Validate Q&A quality (auto-retry if bad)
  - synthesize_audio_tool    : Convert Q&A to dual-voice audio
  - quality_check_tool       : Verify final audio file is valid
"""

import os
import re
import wave
from pathlib import Path
from typing import Dict, List, Tuple, Any

from langchain_core.tools import tool


# ============================================================================
# QUALITY THRESHOLDS
# ============================================================================

MIN_FACTS = 5           # Minimum facts to consider extraction successful
MIN_QA_PAIRS = 4        # Minimum Q&A pairs for valid podcast (target is 8)
MIN_QUESTION_LEN = 10   # Minimum chars per question
MIN_ANSWER_LEN = 30     # Minimum chars per answer
MAX_ANSWER_LEN = 400    # Maximum chars per answer for TTS
MIN_AUDIO_BYTES = 100_000  # Minimum audio file size (100KB)


# ============================================================================
# TOOL 1: FACT EXTRACTION
# ============================================================================

@tool
def extract_facts_tool(content: str, topic: str) -> Dict[str, Any]:
    """
    Extract ~10 key facts from research content covering all sections.

    Args:
        content: Research report content (markdown)
        topic: The topic being researched

    Returns:
        dict with 'facts' (str), 'fact_count' (int), 'success' (bool)
    """
    from config import get_llm
    from agents.question_generator import extract_key_facts

    if not content or len(content) < 200:
        return {
            "facts": "",
            "fact_count": 0,
            "success": False,
            "error": "Content too short to extract facts"
        }

    client = get_llm()
    facts = extract_key_facts(client, content, topic)

    # Count numbered facts
    fact_count = len(re.findall(r'^\s*\d+[\.\):]', facts, re.MULTILINE))
    success = fact_count >= MIN_FACTS

    return {
        "facts": facts,
        "fact_count": fact_count,
        "success": success,
        "char_length": len(facts),
    }


# ============================================================================
# TOOL 2: Q&A GENERATION
# ============================================================================

@tool
def generate_qa_tool(facts: str, topic: str) -> Dict[str, Any]:
    """
    Generate up to 10 Q&A pairs from extracted facts covering all topics.

    Args:
        facts: Extracted facts from research (numbered list)
        topic: The topic being discussed

    Returns:
        dict with 'questions' (list), 'answers' (list), 'pair_count' (int), 'success' (bool)
    """
    from config import get_llm
    from agents.question_generator import generate_questions_and_answers

    if not facts or len(facts) < 50:
        return {
            "questions": [],
            "answers": [],
            "pair_count": 0,
            "success": False,
            "error": "Facts too short to generate Q&A"
        }

    client = get_llm()
    questions, answers = generate_questions_and_answers(client, facts, topic)

    pair_count = min(len(questions), len(answers))
    success = pair_count >= MIN_QA_PAIRS

    return {
        "questions": questions,
        "answers": answers,
        "pair_count": pair_count,
        "success": success,
    }


# ============================================================================
# TOOL 3: Q&A QUALITY VALIDATION
# ============================================================================

@tool
def validate_qa_tool(questions: List[str], answers: List[str]) -> Dict[str, Any]:
    """
    Validate Q&A quality. Filters out bad pairs and reports issues.

    Args:
        questions: List of questions
        answers: List of answers

    Returns:
        dict with 'valid_questions', 'valid_answers', 'issues', 'success'
    """
    if not questions or not answers:
        return {
            "valid_questions": [],
            "valid_answers": [],
            "issues": ["No Q&A provided"],
            "success": False,
        }

    valid_q, valid_a = [], []
    issues = []

    for i, (q, a) in enumerate(zip(questions, answers), 1):
        q = (q or "").strip()
        a = (a or "").strip()

        # Validate question
        if len(q) < MIN_QUESTION_LEN:
            issues.append(f"Q{i} too short ({len(q)} chars)")
            continue

        # Validate answer
        if len(a) < MIN_ANSWER_LEN:
            issues.append(f"A{i} too short ({len(a)} chars)")
            continue
        if len(a) > MAX_ANSWER_LEN:
            # Truncate at sentence boundary
            a = a[:MAX_ANSWER_LEN].rsplit('.', 1)[0] + '.'

        # Check for generic/placeholder content
        generic_phrases = ['lorem ipsum', '[placeholder]', '[insert', '[your ']
        if any(p in q.lower() or p in a.lower() for p in generic_phrases):
            issues.append(f"Pair {i} has placeholder text")
            continue

        # Drop malformed meta/skip pairs, e.g. question "(Skipped for the same
        # reasons as...)" or an answer that begins with a stray ")"/":" bracket.
        skip_markers = ['skipped', 'same reason', 'same as above', '(omit', 'omitted',
                        'n/a', 'not applicable', '(skip']
        if q.startswith('(') or any(p in q.lower() for p in skip_markers):
            issues.append(f"Q{i} is a meta/skip note — skipped")
            continue
        if a[:1] in (')', ']', ':'):
            issues.append(f"A{i} is malformed (stray bracket) — skipped")
            continue

        # Drop "no data" non-answers — a topic without real facts should be
        # skipped entirely, not voiced as "the research does not specify...".
        no_data_phrases = [
            'no data', 'not specified', 'research does not', 'does not specify',
            'not available', 'no such insights', 'no information', "isn't specified",
            'is not specified', 'not mentioned', 'no specific data', 'does not mention',
            'not provided', 'unfortunately, the provided',
        ]
        if any(p in a.lower() for p in no_data_phrases):
            issues.append(f"A{i} is a 'no data' non-answer — skipped")
            continue

        valid_q.append(q)
        valid_a.append(a)

    success = len(valid_q) >= MIN_QA_PAIRS

    return {
        "valid_questions": valid_q,
        "valid_answers": valid_a,
        "valid_count": len(valid_q),
        "rejected": len(questions) - len(valid_q),
        "issues": issues[:10],  # First 10 issues only
        "success": success,
    }


# ============================================================================
# TOOL 4: AUDIO SYNTHESIS
# ============================================================================

@tool
def synthesize_audio_tool(questions: List[str], answers: List[str], topic: str) -> Dict[str, Any]:
    """
    Synthesize dual-voice audio podcast from Q&A pairs using Sarvam AI.

    Args:
        questions: Validated questions list
        answers: Validated answers list
        topic: Podcast topic

    Returns:
        dict with 'audio_path', 'success', 'size_bytes'
    """
    from agents.audio_sarvam_dual_voice import create_audio_sarvam_dual_voice

    if not questions or not answers:
        return {
            "audio_path": "",
            "success": False,
            "error": "No Q&A pairs to synthesize"
        }

    audio_path = create_audio_sarvam_dual_voice(questions, answers, topic)

    if not audio_path:
        return {
            "audio_path": "",
            "success": False,
            "error": "Sarvam TTS returned empty path"
        }

    # Verify file exists
    path_obj = Path(audio_path)
    if not path_obj.exists():
        return {
            "audio_path": audio_path,
            "success": False,
            "error": f"Audio file not created at {audio_path}"
        }

    size = path_obj.stat().st_size

    return {
        "audio_path": str(audio_path),
        "success": size >= MIN_AUDIO_BYTES,
        "size_bytes": size,
        "size_mb": round(size / (1024 * 1024), 2),
    }


# ============================================================================
# TOOL 5: AUDIO QUALITY CHECK
# ============================================================================

@tool
def quality_check_tool(audio_path: str) -> Dict[str, Any]:
    """
    Verify the generated audio file is valid and playable.

    Args:
        audio_path: Path to the generated WAV file

    Returns:
        dict with quality metrics: duration, sample_rate, channels, valid
    """
    if not audio_path or not Path(audio_path).exists():
        return {
            "valid": False,
            "error": "Audio file does not exist",
        }

    try:
        with wave.open(audio_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration_seconds = n_frames / sample_rate if sample_rate > 0 else 0

        size_bytes = Path(audio_path).stat().st_size

        # Quality criteria
        is_valid = (
            duration_seconds >= 30 and    # At least 30 seconds
            size_bytes >= MIN_AUDIO_BYTES  # At least 100KB
        )

        return {
            "valid": is_valid,
            "duration_seconds": round(duration_seconds, 2),
            "duration_minutes": round(duration_seconds / 60, 2),
            "sample_rate": sample_rate,
            "channels": channels,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
        }
    except wave.Error as e:
        return {"valid": False, "error": f"Invalid WAV file: {e}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ============================================================================
# TOOL REGISTRY (for LangChain agents that need tool lists)
# ============================================================================

PODCAST_TOOLS = [
    extract_facts_tool,
    generate_qa_tool,
    validate_qa_tool,
    synthesize_audio_tool,
    quality_check_tool,
]


def get_podcast_tools():
    """Return all podcast tools for use by LangChain agents."""
    return PODCAST_TOOLS
