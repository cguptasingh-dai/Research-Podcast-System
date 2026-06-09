from typing import TypedDict, Annotated
from operator import add

class ReportState(TypedDict, total=False):
    """State schema for LangGraph podcast pipeline"""
    # Required fields
    report_content: str
    topic: str

    # Generated fields
    facts: str  # Extracted facts from research (new - used by tool pipeline)
    questions: list[str]
    answers: list[str]
    audio_file_path: str

    # Optional fields with defaults
    qa_pairs: list
    conversation_text: str
    agent_logs: Annotated[list[str], add]
    audio_files: list
    transcript: str
