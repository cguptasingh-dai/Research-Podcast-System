"""
Tool-Based LangGraph Podcast Pipeline
======================================
Uses LangChain @tool decorated functions for validated, perfect output.

Pipeline:
  extract_facts -> generate_qa -> validate_qa (RETRY if bad) -> synthesize_audio -> quality_check
"""

import sys
from pathlib import Path

# Ensure Podcast_agent/ is always on sys.path regardless of how this file is imported
_THIS_DIR = str(Path(__file__).parent.resolve())
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from langgraph.graph import StateGraph, END
from state import ReportState
from tools import (
    extract_facts_tool,
    generate_qa_tool,
    validate_qa_tool,
    synthesize_audio_tool,
    quality_check_tool,
)


# Fallback Q&A if all retries fail (last resort - tells user something went wrong)
FALLBACK_QA = [
    ("What is the topic about?", "Unfortunately, the research did not return enough content to generate detailed Q&A. Please try again with a more specific topic."),
    ("Why is this topic important?", "Without sufficient research data, we cannot fully explain the importance. The research API may have returned limited results."),
    ("What are the next steps?", "We recommend trying again with a different topic or rerunning the research to get more comprehensive content."),
    ("Where can I learn more?", "Please consult additional sources or refine your research query for more detailed information on this topic."),
    ("What do you recommend?", "Try the research pipeline again with a more well-known topic to get the best results from the AI podcast agent."),
]


# Retry configuration
MAX_QA_RETRIES = 2  # Try Q&A generation up to 2 times if validation fails


class Pipeline:
    def __init__(self):
        print("\n[PIPELINE] Initializing Tool-Based LangGraph Pipeline...")
        self.graph = StateGraph(ReportState)
        self._build()
        self.compiled = self.graph.compile()  # Compile once, reuse
        print("[PIPELINE] OK: Podcast pipeline initialized with 5 nodes\n")

    def _build(self):
        """Build the tool-based pipeline graph."""
        g = self.graph
        g.add_node("extract_facts", self._extract_facts_node)
        g.add_node("generate_qa", self._generate_qa_node)
        g.add_node("validate_qa", self._validate_qa_node)
        g.add_node("synthesize_audio", self._synthesize_audio_node)
        g.add_node("quality_check", self._quality_check_node)

        # Linear pipeline: facts (from report) -> Q&A -> validate -> audio -> check
        g.add_edge("extract_facts", "generate_qa")
        g.add_edge("generate_qa", "validate_qa")
        g.add_edge("validate_qa", "synthesize_audio")
        g.add_edge("synthesize_audio", "quality_check")
        g.add_edge("quality_check", END)
        g.set_entry_point("extract_facts")

    # ========================================================================
    # NODE 1: EXTRACT FACTS (uses extract_facts_tool)
    # ========================================================================
    def _extract_facts_node(self, state):
        print("\n" + "="*70)
        print("  [TOOL 1/5] extract_facts_tool")
        print("="*70)

        topic = state.get('topic', 'Unknown')
        content = state.get('report_content', '')
        print(f"  Topic: {topic}")
        print(f"  Content: {len(content)} chars")

        try:
            result = extract_facts_tool.invoke({"content": content, "topic": topic})
            state['facts'] = result.get('facts', '')

            if result.get('success'):
                print(f"  OK: {result['fact_count']} facts extracted ({result['char_length']} chars)")
            else:
                print(f"  WARN: Only {result.get('fact_count', 0)} facts extracted - quality may be low")
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            state['facts'] = ""

        return state

    # ========================================================================
    # NODE 2: GENERATE Q&A (uses generate_qa_tool with retry)
    # ========================================================================
    def _generate_qa_node(self, state):
        print("\n" + "="*70)
        print("  [TOOL 2/5] generate_qa_tool")
        print("="*70)

        facts = state.get('facts', '')
        topic = state.get('topic', 'Unknown')

        if not facts:
            print("  ERROR: No facts to generate Q&A")
            state['questions'] = []
            state['answers'] = []
            return state

        # Try up to MAX_QA_RETRIES times
        best_questions = []
        best_answers = []

        for attempt in range(1, MAX_QA_RETRIES + 1):
            print(f"  Attempt {attempt}/{MAX_QA_RETRIES}...")
            try:
                result = generate_qa_tool.invoke({"facts": facts, "topic": topic})

                if result.get('success'):
                    print(f"  OK: {result['pair_count']} Q&A pairs generated")
                    best_questions = result['questions']
                    best_answers = result['answers']
                    break
                else:
                    pair_count = result.get('pair_count', 0)
                    print(f"  Attempt {attempt} returned only {pair_count} pairs - retrying...")
                    # Keep best attempt so far
                    if pair_count > len(best_questions):
                        best_questions = result.get('questions', [])
                        best_answers = result.get('answers', [])
            except Exception as e:
                print(f"  Attempt {attempt} ERROR: {str(e)[:80]}")

        state['questions'] = best_questions
        state['answers'] = best_answers
        return state

    # ========================================================================
    # NODE 3: VALIDATE Q&A (uses validate_qa_tool)
    # ========================================================================
    def _validate_qa_node(self, state):
        print("\n" + "="*70)
        print("  [TOOL 3/5] validate_qa_tool")
        print("="*70)

        questions = state.get('questions', [])
        answers = state.get('answers', [])

        if not questions or not answers:
            print("  WARN: No Q&A to validate, using fallback")
            state['questions'] = [q for q, _ in FALLBACK_QA]
            state['answers'] = [a for _, a in FALLBACK_QA]
            return state

        try:
            result = validate_qa_tool.invoke({"questions": questions, "answers": answers})

            print(f"  Validated: {result['valid_count']} pass, {result['rejected']} rejected")
            if result.get('issues'):
                print(f"  Issues found:")
                for issue in result['issues'][:5]:
                    print(f"    - {issue}")

            if result.get('success'):
                state['questions'] = result['valid_questions']
                state['answers'] = result['valid_answers']
                print(f"  OK: Final {len(state['questions'])} valid Q&A pairs")
                # Show samples
                for i, (q, a) in enumerate(zip(state['questions'][:3], state['answers'][:3]), 1):
                    print(f"    Q{i}: {q[:60].encode('ascii','replace').decode()}")
                    print(f"    A{i}: {a[:60].encode('ascii','replace').decode()}")
            else:
                print(f"  FAIL: Too few valid pairs, using fallback")
                state['questions'] = [q for q, _ in FALLBACK_QA]
                state['answers'] = [a for _, a in FALLBACK_QA]
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            state['questions'] = [q for q, _ in FALLBACK_QA]
            state['answers'] = [a for _, a in FALLBACK_QA]

        return state

    # ========================================================================
    # NODE 4: SYNTHESIZE AUDIO (uses synthesize_audio_tool)
    # ========================================================================
    def _synthesize_audio_node(self, state):
        print("\n" + "="*70)
        print("  [TOOL 4/5] synthesize_audio_tool")
        print("="*70)

        questions = state.get('questions', [])
        answers = state.get('answers', [])
        topic = state.get('topic', 'podcast')

        if not questions or not answers:
            print("  ERROR: No Q&A to synthesize")
            state['audio_file_path'] = ""
            return state

        try:
            result = synthesize_audio_tool.invoke({
                "questions": questions,
                "answers": answers,
                "topic": topic,
            })
            state['audio_file_path'] = result.get('audio_path', '')

            if result.get('success'):
                print(f"  OK: Audio created ({result.get('size_mb', 0)} MB)")
            else:
                print(f"  WARN: {result.get('error', 'Audio quality below threshold')}")
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")
            state['audio_file_path'] = ""

        return state

    # ========================================================================
    # NODE 5: QUALITY CHECK (uses quality_check_tool)
    # ========================================================================
    def _quality_check_node(self, state):
        print("\n" + "="*70)
        print("  [TOOL 5/5] quality_check_tool")
        print("="*70)

        audio_path = state.get('audio_file_path', '')

        if not audio_path:
            print("  SKIP: No audio file to check")
            return state

        try:
            result = quality_check_tool.invoke({"audio_path": audio_path})

            if result.get('valid'):
                print(f"  PASS Quality Check:")
                print(f"    Duration:    {result.get('duration_seconds', 0)}s ({result.get('duration_minutes', 0)} min)")
                print(f"    File size:   {result.get('size_mb', 0)} MB")
                print(f"    Sample rate: {result.get('sample_rate', 0)} Hz")
                print(f"    Channels:    {result.get('channels', 0)}")
            else:
                print(f"  FAIL Quality Check: {result.get('error', 'unknown')}")
                if 'duration_seconds' in result:
                    print(f"    Duration: {result['duration_seconds']}s (need >=30s)")
                if 'size_bytes' in result:
                    print(f"    Size: {result['size_bytes']} bytes (need >=100KB)")
        except Exception as e:
            print(f"  ERROR: {str(e)[:100]}")

        return state

    def run(self, state):
        """Execute the tool-based pipeline."""
        print("\n" + "="*80)
        print("  [MIC] PODCAST PIPELINE STARTING")
        print("="*80)

        result = self.compiled.invoke(state)

        print("\n" + "="*80)
        print("  [MIC] PODCAST PIPELINE COMPLETE")
        print("="*80)
        print(f"\n  SUMMARY:")
        print(f"  Q&A pairs:  {len(result.get('questions', []))}")
        print(f"  Audio file: {result.get('audio_file_path', 'NOT GENERATED')}\n")

        return result
