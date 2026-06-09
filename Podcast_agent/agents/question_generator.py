"""
Question & Answer Generator using Nemotron Ultra 550B
Extracts facts from research → generates Q&A in single call
"""

import re


def stream_llm(client, prompt: str) -> str:
    """
    Stream response from ChatNVIDIA.
    Skips reasoning_content (internal thinking), returns only final answer.
    """
    result = ""
    reasoning_tokens = 0

    try:
        for chunk in client.stream([{"role": "user", "content": prompt}]):
            # Count and skip internal reasoning tokens
            if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
                reasoning_tokens += len(chunk.additional_kwargs.get("reasoning_content", ""))
                continue  # Skip thinking, use only final answer
            if chunk.content:
                result += chunk.content
    except Exception as e:
        print(f"    ERROR: LLM stream failed: {str(e)[:80]}")
        return ""

    final = result.strip()
    if reasoning_tokens:
        print(f"    [THINK] Reasoning used: {reasoning_tokens} chars")
    if not final:
        print("    WARNING: LLM returned empty content (only reasoning?)")
    return final


def extract_key_facts(client, content: str, topic: str) -> str:
    """
    Step 1: Extract COMPREHENSIVE facts from research report.
    Uses full content (up to 10000 chars) to cover all sections.
    """
    # Use more content for comprehensive coverage
    content_chunk = content[:10000] if len(content) > 10000 else content

    prompt = f"""Extract the 14 most important facts about "{topic}" from this report.
Focus on facts with specific numbers, statistics, and real-world impact.

Report:
{content_chunk}

Pick the BEST facts from each section:
- Definition & overview (2 facts)
- Key findings/features (4 facts)
- Technology / how it works (2 facts)
- Market data & statistics (2 facts)
- Applications & use cases (2 facts)
- Challenges & future outlook (2 facts)

Rules:
- Each fact: 1-2 clear sentences max
- Prioritize specific numbers, percentages, dollar amounts
- Number each fact (1. 2. 3. etc.)
- Use ONLY facts from the report above

Output ONLY the numbered facts, one per line."""

    facts = stream_llm(client, prompt)

    # Fallback if LLM returns empty
    if not facts:
        print("    WARN: Facts extraction returned empty, using content summary")
        facts = f"Topic: {topic}\n" + content_chunk[:1000]

    print(f"    [FACTS] Extracted {facts.count(chr(10))} facts ({len(facts)} chars)")
    return facts


def generate_questions_and_answers(client, facts: str, topic: str):
    """
    Step 2: Generate Q&A pairs from extracted facts.
    Single LLM call for both questions AND answers (faster + coherent).
    Returns: (questions_list, answers_list)
    """
    prompt = f"""Write a focused podcast script about "{topic}".

Facts from research (use ONLY these - do not add your own):
{facts}

STRICT RULES: Use ONLY facts from the research above.
- NEVER invent statistics, percentages, or dollar figures not in the facts
- NEVER write an answer that says "no data", "not specified", "research does not
  mention", "not available", or anything similar. If the facts don't cover an
  aspect, SKIP that question entirely — just leave it out.
- Only ask a question if you can answer it fully with the real facts above.
- It is better to produce FEWER, fully-substantive pairs than to pad with
  non-answers.

Generate between 8 and 10 Q&A pairs between a host and expert — as many as the
facts genuinely support (aim for 10 when the facts are rich enough).

Preferred coverage (include an aspect ONLY if the facts support it; otherwise
skip it and move on — do NOT leave a placeholder):
- Introduction / what {topic} is
- Key features and how it works
- The underlying technology / architecture / how it works internally
- A key statistic or data point (only if a real number exists in the facts)
- Real-world application or use case
- Another use case or benefit
- A specific real-world example, product, or company (if mentioned)
- Comparison with alternatives (if the facts mention any)
- Challenges or limitations (ONLY if the facts mention them — else skip)
- Future outlook / recent developments (ONLY if the facts mention it — else skip)
- Conclusion / key takeaway

Rules:
- Host: SHORT engaging questions (1 sentence, no numbering in question text)
- Expert: 2-3 sentence answers using ONLY facts above
- Every answer must be substantive (a real fact, number, or concrete detail)
- Natural conversational tone (real radio interview)
- NEVER write meta-notes, placeholders, or skip-markers such as "(Skipped...)",
  "(same as above)", "(omitted)", or "N/A". If you won't ask a question, simply
  leave it out — do not write a line about skipping it.
- Number the pairs you DO produce CONSECUTIVELY with NO gaps (Q1, Q2, Q3, ...).
  Every Q must be a genuine question and every A a genuine, complete answer.

Format (number sequentially Q1/A1, Q2/A2, ... only for the pairs you actually
produce — it is fine to stop early, e.g. at Q8):
Q1: [question]
A1: [answer]
Q2: [question]
A2: [answer]
... (continue only while you have real facts to answer; end cleanly — no skip notes)"""

    raw = stream_llm(client, prompt)

    # Parse Q&A pairs from response
    questions = []
    answers = []

    # Strip markdown bold/italic markers so regex can match
    # Handles: **Q1:**, **Q1**:, *Q1:*, __Q1:__, etc.
    cleaned = re.sub(r'[\*_]{1,3}(Q\d+:?)[\*_]{0,3}', r'\1', raw)
    cleaned = re.sub(r'[\*_]{1,3}(A\d+:?)[\*_]{0,3}', r'\1', cleaned)
    # Ensure colon after Q/A markers
    cleaned = re.sub(r'(Q\d+)([^:\d])', r'\1:\2', cleaned)
    cleaned = re.sub(r'(A\d+)([^:\d])', r'\1:\2', cleaned)

    # Flexible regex: matches Q1: or Q1 followed by content until next Q/A
    q_matches = re.findall(r'Q\d+:\s*(.+?)(?=\s*[\*_]*A\d+|\s*[\*_]*Q\d+|$)', cleaned, re.DOTALL)
    a_matches = re.findall(r'A\d+:\s*(.+?)(?=\s*[\*_]*Q\d+|\s*[\*_]*A\d+|$)', cleaned, re.DOTALL)

    def clean_qa_text(text: str) -> str:
        """Remove markdown, brackets, stage directions, outros, and clean up Q&A text."""
        text = text.strip()
        # Remove markdown bold/italic
        text = re.sub(r'[\*_]{1,3}', '', text)
        # Remove stage directions / meta-notes like [Outro Music], [Music Fades],
        # [END OF PODCAST - 10 Q&A pairs achieved], [transcript], etc.
        text = re.sub(r'\[[^\]]*(?:music|outro|intro|fade|sound|sfx|pause|silence|'
                      r'laughs?|sighs?|end of podcast|podcast|broadcast|q&a|qa pairs?|'
                      r'pairs achieved|achieved|transcript)[^\]]*\]',
                      '', text, flags=re.IGNORECASE)
        # Catch-all: remove any short trailing bracketed note at the very end
        text = re.sub(r'\s*\[[^\]]{0,80}\]\s*$', '', text).strip()
        # Remove "Host:" / "Dr. Kim:" speaker tags inside answer
        text = re.sub(r'\b(?:Host|Dr\.?\s*\w+|Guest|Expert|Interviewer|Alex|Rachel)\s*:\s*', '', text, flags=re.IGNORECASE)
        # Remove leading brackets like [question] or (text)
        text = re.sub(r'^\[([^\]]+)\]$', r'\1', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Cut off trailing outro/dialog phrases (flexible: optional names between)
        outro_phrases = [
            r'\bthanks?\b[^.!?]*\bfor\b[^.!?]*\b(?:shedding|joining|listening|having|sharing|insights?|illuminating)\b.*$',
            r'\bthank you\b[^.!?]*\bfor\b.*$',
            r'\bmy pleasure\b.*$',
            r'\buntil next time\b.*$',
            r'\bstay (?:tuned|connected|informed)\b.*$',
            r'\bthat\'?s all for\b.*$',
            r'\bsee you next\b.*$',
            # Trailing speaker dialog: "Dr. X, ..." or "Alex, ..." at end
            r'\b(?:Dr\.?\s*\w+|Alex|Rachel|Sarah|Mike|John)\s*[,:]\s+[^.!?]*[.!?]?\s*$',
        ]
        for pattern in outro_phrases:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()

        # Strip standalone trailing courtesy words like "Thanks." "Thank you."
        text = re.sub(r'(?:^|\s)(?:thanks?\.?|thank\s+you\.?|cheers\.?|bye\.?)\s*$',
                      '', text, flags=re.IGNORECASE).strip()

        # Trim trailing punctuation/whitespace artifacts
        text = text.rstrip(' .,;:')

        # Cap length at 400 chars for clean TTS
        if len(text) > 400:
            text = text[:400].rsplit('.', 1)[0] + '.'
        elif text and not text.endswith(('.', '!', '?')):
            text = text + '.'

        return text.strip()

    for q in q_matches:
        q_clean = clean_qa_text(q)
        if q_clean and len(q_clean) > 5:
            questions.append(q_clean)

    for a in a_matches:
        a_clean = clean_qa_text(a)
        if a_clean and len(a_clean) > 5:
            answers.append(a_clean)

    print(f"  [PARSE] Extracted {len(questions)} Q, {len(answers)} A")

    # Debug: show first Q&A if parsing succeeded
    if questions and answers:
        print(f"  [SAMPLE] Q1: {questions[0][:80]}")
        print(f"  [SAMPLE] A1: {answers[0][:80]}")
    elif raw:
        # Parsing failed - show what we got
        print(f"  [DEBUG] Raw output (first 200 chars): {raw[:200]}")

    # Ensure equal counts
    min_count = min(len(questions), len(answers))
    questions = questions[:min_count]
    answers = answers[:min_count]

    return questions, answers


def generate_questions(client, content: str, topic: str) -> str:
    """Called by pipeline - extracts facts for use in Q&A generation"""
    facts = extract_key_facts(client, content, topic)
    print(f"  [FACTS] {len(facts)} chars extracted from report")
    return facts
