"""
Answer Generator - uses ChatNVIDIA Nemotron Ultra 550B
Note: Q&A generation is now handled together in question_generator.py
This file kept for backward compatibility.
"""

from .question_generator import stream_llm


def generate_answers(client, content: str, questions: list) -> str:
    """
    Generate answers for given questions using Nemotron Ultra 550B.
    Used as fallback if combined Q&A generation fails.
    """
    q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    prompt = f"""You are an expert being interviewed on a radio show.
Answer each question naturally and conversationally based on this content.

Content:
{content[:6000]}

Questions:
{q_text}

Rules:
- Answer in 2-3 natural sentences per question
- Sound like you are speaking live on radio
- Only use information from the content above
- Number each answer (1. 2. 3. etc.)
- Be clear, engaging, and informative

Format:
1. [answer]
2. [answer]
..."""

    return stream_llm(client, prompt)
