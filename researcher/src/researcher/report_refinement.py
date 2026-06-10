import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from crewai import Agent, Task

from .crew import researcher
from .pdf_generator_professional import convert_report_to_pdf_professional

# Absolute path to project root Report/ folder so PDFs always land there
# regardless of the process working directory (critical for Docker / EC2)
# __file__ = researcher/src/researcher/report_refinement.py
# 4 levels up  = A2a/ (project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_DEFAULT_REPORT_DIR = str(_PROJECT_ROOT / "Report")


class IterationResult(BaseModel):
    """Result of one iteration."""
    iteration: int
    report: str
    critique: str
    score: float
    suggestions: str


class ReportRefinementSession:
    """
    Direct iteration between Report Writer and Critic Agent.

    Workflow:
    1. Report Writer creates report
    2. Critic evaluates (scores & suggests improvements)
    3. Report Writer improves based on feedback
    4. Repeat 3 times
    5. Select best-scored version
    6. Provide final suggestions
    """

    def __init__(self, research_findings: Optional[str], topic: str = ""):
        self.research_findings = research_findings
        self.topic = topic
        self.iterations = []
        self.best_iteration = None
        self.crew = researcher()

    def _scrape_url(self, url: str) -> str:
        """Fetch a URL and return cleaned visible text (best-effort)."""
        try:
            import requests as _rq
            from bs4 import BeautifulSoup
            r = _rq.get(url, timeout=15,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot)"})
            if r.status_code != 200 or not r.text:
                return ""
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                tag.decompose()
            return " ".join(soup.get_text(separator=" ").split())
        except Exception as e:
            print(f"[DIRECT] scrape fail {url[:40]}: {str(e)[:40]}")
            return ""

    def _direct_web_research(self) -> str:
        """
        Deep web research executed DIRECTLY in Python — the reliable path.

        The CrewAI agent (nemotron) frequently SIMULATES tool calls and fabricates
        URLs instead of actually searching. To guarantee REAL sources we drive the
        search ourselves:
          1. Serper Google search across several angles (definition / features /
             use cases / challenges / news) — collect answer box, knowledge graph,
             organic results and "people also ask".
          2. Scrape the top real URLs (concurrently) for full-text content.
          3. Add real public GitHub repos via GithubRepoSearchTool.
        Returns assembled findings with ONLY real URLs, or "" if Serper unavailable.
        """
        import requests as _rq
        serper_key = os.getenv("SERPER_API_KEY", "").strip().strip('"').strip("'")
        if not serper_key:
            print("[DIRECT] No SERPER_API_KEY — skipping direct search")
            return ""

        topic = self.topic
        queries = [
            topic,
            f"{topic} what is definition overview",
            f"{topic} how it works features architecture",
            f"{topic} use cases applications examples",
            f"{topic} benefits advantages",
            f"{topic} challenges limitations problems",
            f"{topic} 2024 2025 latest news",
        ]

        facts = []        # bullet strings with [Source: url]
        sources = {}      # url -> title (dedup, preserves insertion order)
        definition = ""

        print(f"[DIRECT] Running {len(queries)} Serper searches for: {topic}")
        for q in queries:
            try:
                resp = _rq.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                    json={"q": q, "num": 6}, timeout=20,
                )
                if resp.status_code != 200:
                    print(f"[DIRECT] Serper {resp.status_code} for '{q[:40]}'")
                    continue
                data = resp.json()

                ab = data.get("answerBox") or {}
                ab_text = ab.get("answer") or ab.get("snippet") or ""
                if ab_text and len(ab_text) > 25:
                    link = ab.get("link", "")
                    facts.append(f"- {ab_text.strip()}" + (f" [Source: {link}]" if link else ""))
                    if link:
                        sources[link] = ab.get("title", "Answer")

                kg = data.get("knowledgeGraph") or {}
                kg_desc = kg.get("description") or ""
                if kg_desc and not definition:
                    definition = kg_desc.strip()
                    src = kg.get("descriptionLink") or kg.get("website") or ""
                    if src:
                        sources[src] = kg.get("title", topic)

                for o in data.get("organic", [])[:6]:
                    title = (o.get("title") or "").strip()
                    snippet = (o.get("snippet") or "").strip()
                    link = (o.get("link") or "").strip()
                    if snippet and link:
                        facts.append(f"- {snippet} [Source: {link}]")
                    if link:
                        sources.setdefault(link, title)
                    if not definition and snippet and len(snippet) > 60:
                        definition = snippet

                for paa in (data.get("peopleAlsoAsk") or [])[:3]:
                    snip = (paa.get("snippet") or "").strip()
                    link = (paa.get("link") or "").strip()
                    if snip:
                        facts.append(f"- {snip}" + (f" [Source: {link}]" if link else ""))
                        if link:
                            sources.setdefault(link, paa.get("title", ""))
            except Exception as e:
                print(f"[DIRECT] search error '{q[:30]}': {str(e)[:60]}")

        if not sources:
            print("[DIRECT] No real results returned from Serper")
            return ""

        # Scrape the top unique URLs concurrently for deeper content
        scraped = []
        top_urls = list(sources.keys())[:4]
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                fut = {pool.submit(self._scrape_url, u): u for u in top_urls}
                for f in as_completed(fut):
                    u = fut[f]
                    txt = f.result()
                    if txt and len(txt) > 200:
                        scraped.append(f"=== Full content from {u} ===\n{txt[:1500]}")
        except Exception as e:
            print(f"[DIRECT] scrape pool error: {str(e)[:50]}")

        # Real GitHub repos
        github_section = ""
        try:
            from researcher.tools.github_repo_tool import search_github_repos
            gh = search_github_repos(topic, max_repos=2)
            if gh and len(gh) > 50 and "no repositories" not in gh.lower():
                github_section = f"\nGITHUB REPOSITORIES (real):\n{gh}"
        except Exception as e:
            print(f"[DIRECT] github skip: {str(e)[:50]}")

        # Dedup facts, keep order
        seen, uniq_facts = set(), []
        for fct in facts:
            key = fct[:80].lower()
            if key not in seen:
                seen.add(key)
                uniq_facts.append(fct)

        src_lines = "\n".join(f"- {t or 'Source'}: {u}" for u, t in sources.items())
        findings = (
            f"TOPIC: {topic}\n"
            f"DEFINITION: {definition or 'See key facts below.'}\n\n"
            f"KEY FACTS (from real web search):\n"
            f"{chr(10).join(uniq_facts[:30])}\n\n"
            f"{chr(10).join(scraped)}\n"
            f"{github_section}\n\n"
            f"SOURCES (real URLs):\n{src_lines}\n"
        )
        print(f"[DIRECT] OK — {len(uniq_facts)} facts, {len(sources)} real URLs, "
              f"{len(scraped)} pages scraped")
        return findings

    def _download_image_as_datauri(self, url: str):
        """Download one image URL, convert to PNG, return (data_uri, w, h) or None."""
        if not url or url.lower().endswith(".svg"):
            return None
        try:
            import base64
            from io import BytesIO
            import requests as _rq
            from PIL import Image
            resp = _rq.get(url, timeout=12,
                           headers={"User-Agent": "Mozilla/5.0 (research-bot)"})
            if resp.status_code != 200 or len(resp.content) < 4000:
                return None
            img = Image.open(BytesIO(resp.content))
            img.load()
            w, h = img.size
            if w < 320 or h < 160:                 # skip icons/thumbnails
                return None
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            if w > 1000:                            # cap width to keep PDF light
                img = img.resize((1000, int(h * 1000 / w)))
            buf = BytesIO()
            img.save(buf, "PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}", w, h
        except Exception:
            return None

    def _fetch_topic_images(self, output_dir: str, max_images: int = 3):
        """
        Fetch SEVERAL relevant images for the topic via Serper Images (overview /
        how-it-works / use-cases), convert each to PNG, and return a list of
        {data_uri, source}. Best-effort: returns [] on failure so the PDF still builds.
        """
        serper_key = os.getenv("SERPER_API_KEY", "").strip().strip('"').strip("'")
        if not serper_key:
            return []
        try:
            import base64
            import requests as _rq
        except Exception:
            return []

        queries = [
            f"{self.topic} architecture diagram",
            f"{self.topic} how it works",
            f"{self.topic} use cases",
            f"{self.topic} explained",
            self.topic,
        ]
        images, seen = [], set()
        safe = re.sub(r'[^a-z0-9_]+', '_', self.topic.lower()).strip('_') or 'topic'
        for q in queries:
            if len(images) >= max_images:
                break
            try:
                r = _rq.post(
                    "https://google.serper.dev/images",
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                    json={"q": q, "num": 8}, timeout=20,
                )
                if r.status_code != 200:
                    continue
                for im in r.json().get("images", []):
                    if len(images) >= max_images:
                        break
                    iu = im.get("imageUrl", "")
                    if not iu or iu in seen:
                        continue
                    seen.add(iu)
                    got = self._download_image_as_datauri(iu)
                    if not got:
                        continue
                    data_uri, w, h = got
                    src = im.get("source") or re.sub(r'^https?://(www\.)?', '', iu).split('/')[0]
                    images.append({"data_uri": data_uri, "source": src})
                    try:
                        with open(os.path.join(output_dir, f"{safe}_visual_{len(images)}.png"), "wb") as fh:
                            fh.write(base64.b64decode(data_uri.split(',', 1)[1]))
                    except Exception:
                        pass
                    print(f"[VISUAL] Image {len(images)} ({w}x{h}) from {src}")
            except Exception as e:
                print(f"[VISUAL] image search error: {str(e)[:50]}")

        print(f"[VISUAL] Embedded {len(images)} topic image(s) in PDF")
        return images

    def _conduct_research(self) -> str:
        """
        Conduct web research via direct, Python-driven search (Serper + scrape +
        GitHub). This is the only research path: it does REAL tool calls reliably,
        unlike the LLM agent which simulates tool calls and fabricates URLs.
        Returns "" if no real data is found — the quality gate then produces an
        honest "no data" report instead of fabricated content.
        """
        if self.research_findings:
            return self.research_findings

        print(f"\n[RESEARCH] Starting direct web search for: {self.topic}")
        findings = self._direct_web_research()
        if findings and len(findings) > 300:
            print(f"[RESEARCH] Collected real findings ({len(findings)} chars)")
            return findings

        print("[RESEARCH] No real data found — gate will return an honest no-data report")
        return ""

    def _check_research_quality(self, findings: str) -> dict:
        """
        Hard gate before calling LLM report writer.
        If no real URLs found → block LLM entirely to prevent hallucination.
        No prompt can reliably stop an LLM from fabricating when it has nothing real.
        """
        if not findings or len(findings.strip()) < 50:
            print(f"[QUALITY] BLOCK — findings empty")
            return {'block': True, 'url_count': 0, 'fact_count': 0, 'reason': 'empty findings'}

        findings_lower = findings.lower()

        # Hard signal 1: hallucination keywords — LLM fabricated content.
        # Use ONLY unambiguous multi-word fabrication markers. Single words like
        # "hypothetical"/"speculative"/"non-existent" appear in legitimate articles
        # and caused false-positive blocks of good reports.
        hallucination_phrases = [
            'simulated research', 'if it were real', 'plausible narrative',
            'absence of concrete data', 'for the purposes of this speculative',
            'this hypothetical scenario', 'purely speculative narrative',
        ]
        is_hallucinated = any(p in findings_lower for p in hallucination_phrases)

        # Hard signal 2: no-data phrases from search agent
        no_data_phrases = [
            'no results found', 'no data found', 'no information found',
            'no sources found', 'no specific urls', 'search returned no',
            'could not find any', 'nothing found',
        ]
        has_no_data_signal = any(p in findings_lower for p in no_data_phrases)

        # Count source URLs — but EXCLUDE fabricated/placeholder URLs. When the agent
        # can't find real data it often invents URLs (arxiv.org/abs/XXXXXXX,
        # example.com, github.com/user/..., forum/.../12345). Those must NOT count
        # as real research, or the gate lets a 100%-hallucinated report through.
        fake_url_patterns = [
            r'x{4,}',                                   # /abs/XXXXXXX placeholder
            r'example\.(com|org|ai|net|io)',            # example.* domains
            r'/(12345|123456|1234567|00000|99999)\b',   # placeholder numeric IDs
            r'github\.com/(user|username|your[-_]?username|example|repo|owner|account)/',
            r'(yourdomain|your[-_]site|placeholder|sample\.|lorem|abc123|foo\.bar)',
            r'forum\.techq', r'blog\.example', r'test\.com', r'somesite',
        ]
        all_urls = re.findall(r'https?://[^\s\)\]\>"]+', findings)
        real_urls, fake_urls = [], []
        for u in all_urls:
            ul = u.lower()
            if any(re.search(p, ul) for p in fake_url_patterns):
                fake_urls.append(u)
            else:
                real_urls.append(u)
        url_count = len(real_urls)
        fake_count = len(fake_urls)

        # Count numbered/bulleted facts
        fact_count = len(re.findall(r'^\s*[\d\-\*]\d*[\.\)]\s+\S', findings, re.MULTILINE))
        if fact_count == 0:
            fact_count = findings.count('\n- ') + findings.count('\n* ')

        # BLOCK decision — REAL (non-fabricated) URLs are the ground truth for
        # "real research happened". Only block when truly empty or fabricated:
        #   - 0 real URLs              → no genuine sources → block
        #   - any fabricated URL found → agent invented data → block
        #   - fabrication marker       → LLM made things up → block
        #   - no-data phrase + <2 real URLs → genuinely empty → block
        # A report with >=2 real URLs and NO fake URLs is legitimate even if it
        # honestly says "no market data found" in one section.
        reason = []
        if url_count == 0:
            block = True
            reason.append('zero real URLs in research')
        elif fake_count > 0:
            block = True
            reason.append(f'{fake_count} fabricated/placeholder URL(s) detected')
        elif is_hallucinated:
            block = True
            reason.append('fabrication marker detected')
        elif has_no_data_signal and url_count < 2:
            block = True
            reason.append(f'no-data phrase with only {url_count} real URL(s)')
        else:
            block = False
            if has_no_data_signal:
                reason.append(f'no-data phrase present but {url_count} real URLs — PASS')
        reason_str = ' | '.join(reason) if reason else 'ok'

        status = 'BLOCK' if block else 'PASS'
        print(f"[QUALITY] {status} — real URLs:{url_count} fake:{fake_count} facts:{fact_count} | {reason_str}")

        return {
            'block': block,
            'url_count': url_count,
            'fact_count': fact_count,
            'is_hallucinated': is_hallucinated,
            'reason': reason_str,
        }

    def _make_no_data_report(self, quality_info: dict) -> str:
        """
        Build an honest 'no data found' report WITHOUT calling the LLM.
        Used when research returns nothing real — prevents hallucination entirely.
        """
        url_count  = quality_info.get('url_count', 0)
        fact_count = quality_info.get('fact_count', 0)
        reason     = quality_info.get('reason', 'no real data found')

        return f"""# {self.topic}: Research Report

## Research Status

**No sufficient data was found** for the topic **"{self.topic}"** after exhaustive web searches.

## What Was Searched

The research agent performed multi-phase web searches including:
- Direct searches: "{self.topic}", "{self.topic} definition", "{self.topic} technology"
- Deep searches: "{self.topic} features", "{self.topic} market", "{self.topic} use cases"
- Alternative searches: "{self.topic} product", "{self.topic} company", "{self.topic} review"
- GitHub repository search for "{self.topic}"

**Results:** {url_count} URLs found, {fact_count} facts extracted.

## Conclusion

The topic **"{self.topic}"** may be:
- A private/internal product not publicly indexed
- A very new or niche concept with limited web presence
- A misspelling or alternate name — try searching with different keywords
- A proprietary technology not yet publicly documented

## Recommendations

1. Verify the correct spelling or full name of "{self.topic}"
2. Try searching with related terms or the full product/company name
3. Check if it is an internal/private tool not publicly available
4. Try a more specific version of the topic name

## Sources

No source URLs were found in web searches for "{self.topic}".
"""

    def run_iterations(self, num_iterations: int = 1) -> dict:
        """
        Run the research + report refinement cycle.

        Returns:
            Dictionary with best report, all iterations, and final suggestions
        """
        # Conduct research if needed
        if not self.research_findings:
            self.research_findings = self._conduct_research()

        # CHECK RESEARCH QUALITY before calling LLM report writer
        quality_info = self._check_research_quality(self.research_findings)

        if quality_info.get('block'):
            print(f"[BLOCK] LLM report writer blocked — reason: {quality_info.get('reason')}")
            no_data_report = self._make_no_data_report(quality_info)
            # Store as a single iteration result
            result = IterationResult(
                iteration=1,
                report=no_data_report,
                critique="No data found — report writer skipped to prevent hallucination",
                score=0.0,
                suggestions="Try a different topic or more specific search terms"
            )
            self.iterations.append(result)
            self.best_iteration = result
            return self._generate_summary()

        for iteration_num in range(1, num_iterations + 1):
            print(f"\n[ITERATION {iteration_num}/{num_iterations}] Starting...")

            # Step 1: Generate/Improve Report
            print(f"[THINKING] Report Writer generating report...")
            report = self._generate_report(self.crew, iteration_num)
            print(f"[OK] Report generated ({len(report)} characters)")

            # Step 2: Critique - ONLY if more iterations follow (skip on last iteration)
            if iteration_num < num_iterations:
                print(f"[THINKING] Critic evaluating for iteration {iteration_num + 1}...")
                critique_result = self._critique_report(self.crew, report)
                score = critique_result['score']
                critique_text = critique_result['critique']
                suggestions = critique_result['suggestions']
                print(f"[OK] Critique complete - Score: {score:.1f}/100")
            else:
                # Last/only iteration - skip critique to save time!
                print(f"[SKIP] Critique skipped (final iteration - saving time)")
                score = 80.0  # Default score
                critique_text = "Final iteration - no critique needed"
                suggestions = ""

            # Display Critic feedback in CLI
            print(f"\n[FEEDBACK] Critic Agent Evaluation Details:")
            print("-" * 70)
            # Print first 1000 chars of critique to show dimension scores
            feedback_preview = critique_text[:800] if len(critique_text) > 800 else critique_text
            try:
                # Clean problematic Unicode
                feedback_preview = feedback_preview.replace(' ', ' ').replace('→', '->').replace('↑', '^')
                print(feedback_preview)
            except UnicodeEncodeError:
                # Fallback to ASCII
                feedback_preview_ascii = feedback_preview.encode('ascii', errors='replace').decode('ascii')
                print(feedback_preview_ascii)
            print("-" * 70)

            # Step 3: Store iteration result
            result = IterationResult(
                iteration=iteration_num,
                report=report,
                critique=critique_text,
                score=score,
                suggestions=suggestions
            )
            self.iterations.append(result)
            print(f"[STORED] Iteration {iteration_num} result saved")

            # Display improvement suggestions for next iteration
            if iteration_num < num_iterations and suggestions:
                print(f"\n[SUGGESTIONS] For next iteration:")
                print("-" * 70)
                try:
                    suggestions_clean = suggestions.replace(' ', ' ').replace('→', '->').replace('↑', '^')
                    print(suggestions_clean[:600])
                except UnicodeEncodeError:
                    print(suggestions[:600].encode('ascii', errors='replace').decode('ascii'))
                print("-" * 70)


        # Select best iteration
        print(f"\n[ANALYZING] Selecting best iteration from {num_iterations} versions...")
        self._select_best_iteration()
        if self.best_iteration:
            print(f"[SELECTED] Iteration {self.best_iteration.iteration} with score {self.best_iteration.score:.1f}/100")

        # Generate summary
        return self._generate_summary()

    def _generate_report(self, crew, iteration_num: int) -> str:
        """Generate or improve report."""
        if iteration_num == 1:
            # Use REAL research findings from web search
            findings_section = (
                f"REAL WEB RESEARCH FINDINGS (use ONLY these facts):\n"
                f"{self.research_findings}\n\n"
            ) if self.research_findings else ""

            # Use MOST of the findings so the writer sees all facts + every source URL
            # and can organize them into main points and sub-points.
            findings_text = self.research_findings[:12000] if self.research_findings else "No research data available."

            prompt = f"""TASK: Write a research report on "{self.topic}" using ONLY the facts below.

DATA TO USE (research notes from real web searches):
{findings_text}

STRICT RULES — MUST FOLLOW:
- Use ONLY facts that appear in the research notes above
- NEVER invent statistics, percentages, dollar figures, or CAGR values
- NEVER write phrases like "projected to", "estimated to", "hypothetically", "speculative"
- DATA-DRIVEN SECTIONS: If a section has NO real data in the research notes, OMIT
  that entire section (heading AND body). Do NOT write placeholder text like
  "No data found", "No market data found", or "not available". Simply skip it.
- Only include the OPTIONAL sections below that you can fill with REAL collected facts.
- If the topic is obscure/unknown, say so clearly — do NOT fabricate a plausible story
- Do not add meta-commentary like "this report constitutes..." or "see above"
- Start directly with the title line

WRITING STYLE — produce a COMPREHENSIVE, PROFESSIONAL report:
- Cover the topic IN DEPTH: every main point AND its sub-points.
- Use "## " for MAIN sections and "### " for SUB-sections/sub-points inside a section.
- Begin each main section with a 2-3 sentence explanatory PROSE paragraph, THEN
  break the details into ### sub-points and/or bullets.
- Explain the "why" and "how", connect related facts — do not just list one-liners.
- Keep EVERY claim grounded in the research notes (no invented data, no fabricated numbers).
- Keep each fact's source URL inline in [square brackets]; they are auto-converted
  to numbered citations later, so always attach the real URL to the fact.
- NEVER write a citation bracket without a real URL. Do NOT write "[Implicit...]",
  "[inferred from context]", or "[provided sources]" — if you lack a source for a
  claim, state it plainly without brackets, or leave the claim out.

REQUIRED MAIN SECTIONS (always include):
  Executive Summary, Introduction, How It Works, Key Findings, Conclusion, References

OPTIONAL MAIN SECTIONS (include ONLY if the notes contain real data — else OMIT the
heading entirely; never write a "no data" placeholder):
  Background & History, Key Features & Capabilities, Types / Variants,
  Market Analysis, Applications & Use Cases, Comparison with Alternatives,
  Challenges & Limitations, Future Outlook

STRUCTURE (use ### sub-points wherever the topic has distinct aspects):

# {self.topic}: Research Report

## Executive Summary
[4-6 sentence prose overview of the most important findings actually collected.]

## Introduction
[Full paragraph: clear definition of {self.topic}, context, and why it matters.]
### Background & Origin
[Who created it, when, and why — ONLY if the notes cover this; else omit this sub-heading.]

## How It Works
[1-2 prose paragraphs explaining the underlying mechanism/approach in plain language.]
### Core Components
[The main building blocks / parts as sub-points with sources — ONLY if notes cover them.]
### Key Concepts
[Important concepts or terms with sources — ONLY if notes cover them.]

## Key Features & Capabilities
[Prose intro, then bulleted sub-points of the distinct features, each with its [source URL]. OMIT the whole section if the notes list no features.]

## Key Findings
[2-3 sentence prose summary of the themes, THEN a bulleted list of every real fact
from the notes — as many as were found. Each bullet ends with its real [source URL].]

## Applications & Use Cases
[OPTIONAL. Prose intro, then group real-world uses — use ### per domain or bullets, each with [source]. Omit if no use cases in notes.]

## Market Analysis
[OPTIONAL. Real market figures/adoption only. Omit if none in notes.]

## Comparison with Alternatives
[OPTIONAL. How {self.topic} compares to alternatives mentioned in the notes, with [sources]. Omit if no comparison data.]

## Challenges & Limitations
[OPTIONAL. Real challenges as sub-points with [sources]. Omit if none in notes.]

## Future Outlook
[OPTIONAL. Recent developments / trajectory from the notes, with [sources]. Omit if none.]

## Conclusion
[Full paragraph: synthesis of findings, trade-offs, and outlook for {self.topic}.]

## References
[List the real source URLs from the notes (they are auto-numbered). If none: "Sources: Web searches via Serper API".]

WRITE THE FULL, COMPREHENSIVE REPORT NOW (start with "# {self.topic}: Research Report"; use ### sub-points; omit any optional section/sub-heading that has no real data):"""
        else:
            # Improve existing report based on feedback
            previous_iteration = self.iterations[iteration_num - 2]
            suggestions = previous_iteration.suggestions

            prompt = f"""Improve this report based on the following feedback. Make it significantly more detailed and comprehensive.

Feedback to address:
{suggestions}

Previous Report (Score: {previous_iteration.score:.1f}/100):
{previous_iteration.report}

Requirements:
- Address every feedback point specifically
- Add more statistics, data points and percentages
- Strengthen and expand recommendations
- Add more [Source: X] citations
- Improve depth and analysis in weak sections
- Maintain professional tone and all existing sections
- Minimum 2000 words"""

        agent = crew.report_writer()
        task = Task(
            description=prompt,
            agent=agent,
            expected_output="A complete professional report in markdown format"
        )
        for attempt in range(3):  # Reduced from 5 to 3 retries
            try:
                result = agent.execute_task(task)
                report = result if isinstance(result, str) else str(result)
                return self._clean_report(report)
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 15 if 'connection' in str(e).lower() else 8  # Reduced wait times
                print(f"[RETRY] Attempt {attempt + 1} failed: {str(e)[:60]}. Waiting {wait}s...")
                time.sleep(wait)

    def _clean_report(self, report: str) -> str:
        """Remove LLM artifacts and ensure report starts with proper heading."""
        if not report:
            return report

        # Remove common LLM preamble/postamble artifacts
        artifacts = [
            "Already Provided Above in the Required Format",
            "Already provided above",
            "Here is the report:",
            "Here is the professional report:",
            "Below is the report:",
            "OUTPUT THE REPORT BELOW",
            "START WITH THE # HEADING:",
            "(The report above constitutes the Final Answer as per the requirements)",
            "(The report above constitutes the Final Answer)",
            "The report above constitutes the Final Answer",
            "constitutes the comprehensive final answer",
            "constitutes the final answer",
            "Direct Link to Final Report",
            "Report Embedded Above",
            "END OF FINAL ANSWER",
            "the report is above",
            "see notes above",
            "see research notes above",
            "as per the requirements",
            "=== END RAW NOTES ===",
            "=== RAW RESEARCH NOTES",
            "WRITE THE FULL REPORT NOW",
            "(start with",
        ]
        for artifact in artifacts:
            # Case-insensitive replace
            report = re.sub(re.escape(artifact), "", report, flags=re.IGNORECASE).strip()

        # Remove entire meta-paragraphs that describe the report instead of being content
        # Pattern: "**Summary of Key Takeaways**" sections that follow the report
        report = re.sub(r'\n\*+(?:Summary of |Key Takeaways)[^\n]*\*+\n.*?(?=\n#|\Z)',
                       '\n', report, flags=re.DOTALL | re.IGNORECASE)
        report = re.sub(r'\n\*+(?:Action Required|Direct Link)[^\n]*\*+\n.*?(?=\n#|\Z)',
                       '\n', report, flags=re.DOTALL | re.IGNORECASE)

        # Remove standalone "**" markers
        report = re.sub(r'\n\s*\*\*\s*\n', '\n', report)
        report = re.sub(r'\n---+\s*$', '', report)

        # Remove leading blank lines and find the first # heading
        lines = report.splitlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                start_idx = i
                break

        report = "\n".join(lines[start_idx:]).strip()

        # If report doesn't start with heading, add one
        if not report.startswith('#'):
            report = f"# {self.topic}: Research Report\n\n{report}"

        # Detect hallucinated report (LLM invented facts instead of using real data)
        hallucination_signals = [
            'hypothetical', 'speculative', 'simulated research',
            'non-existent', 'if it were real', 'plausible narrative',
            'for the purposes of this speculative', 'absence of concrete data',
        ]
        report_lower = report.lower()
        if any(sig in report_lower for sig in hallucination_signals):
            print(f"  [WARN] Hallucination detected in report — LLM invented facts!")
            print(f"  [WARN] Report contains speculative/hypothetical content.")
            # Inject a clear warning at the top of the report
            warning_banner = (
                f"\n> **WARNING**: Real web search found no information about "
                f'"{self.topic}". The content below may be speculative. '
                f"Please verify independently.\n\n"
            )
            # Insert warning after the first heading
            lines = report.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('#'):
                    lines.insert(i + 1, warning_banner)
                    break
            report = '\n'.join(lines)

        # Detect meta-response (LLM said "see above" instead of writing report)
        if len(report) < 300:
            print(f"  [WARN] Report suspiciously short ({len(report)} chars) - may be meta-response")
            # Append research findings as fallback content
            if self.research_findings and len(self.research_findings) > 200:
                print(f"  [FALLBACK] Appending research findings ({len(self.research_findings)} chars) to report")
                report = f"""# {self.topic}: Research Report

## Executive Summary

This report presents the findings from research conducted on {self.topic}. The information below is compiled from web sources searched via Serper API.

## Research Findings

{self.research_findings}

## Conclusion

The above findings represent the current available information on {self.topic}. For deeper analysis, additional targeted research with industry-specific sources is recommended.

## References

Sources: Web searches conducted via Serper API. See findings above for specific data points and citations.
"""

        # Strip optional sections that contain only "no data found" placeholders
        report = self._strip_empty_sections(report)

        # Convert inline URL citations to numbered [n] markers and build a single
        # clean numbered reference list at the very end.
        report = self._renumber_citations(report)

        return report

    def _renumber_citations(self, report: str) -> str:
        """
        Replace inline URL citations in the body with numbered markers [1], [2], ...
        and rebuild ONE clean numbered '## References' list at the end of the report.

        - Inline citations like [https://x], [Source: https://x] or [https://a, https://b]
          become [1], [3] etc. (in order of first appearance).
        - The old References/Sources section is removed and rebuilt fresh, so any
          junk lines (e.g. "Omitted as per instructions") that carry no URL are dropped.
        """
        if not report:
            return report

        # Separate the body from any existing References/Sources/Bibliography section
        m = re.search(r'\n#{1,3}\s*(References|Sources|Bibliography)\b.*\Z',
                      report, re.IGNORECASE | re.DOTALL)
        body = report[:m.start()] if m else report
        existing_refs = report[m.start():] if m else ''

        url_order = []
        url_to_num = {}

        def _register(url: str) -> int:
            url = url.rstrip('.,);]>')
            if url not in url_to_num:
                url_order.append(url)
                url_to_num[url] = len(url_order)
            return url_to_num[url]

        def _repl(match):
            inner = match.group(1)
            urls = re.findall(r'https?://[^\s,\]]+', inner)
            if not urls:
                return match.group(0)  # not a URL citation — leave untouched
            nums = []
            for u in urls:
                n = _register(u)
                if str(n) not in nums:
                    nums.append(str(n))
            return '[' + ', '.join(nums) + ']'

        # Replace bracketed citations that contain at least one URL
        body = re.sub(r'\[([^\[\]]*https?://[^\[\]]*)\]', _repl, body)

        # Remove leftover non-citation bracket artifacts the LLM sometimes adds,
        # e.g. "[Implicit from the context...]" or "[inferred from provided sources]".
        body = re.sub(
            r'\s*\[[^\]\d][^\]]*?(?:implicit|inferred|from the context|from context|'
            r'provided sources?|no direct|not directly|mentioned in outline)[^\]]*\]',
            '', body, flags=re.IGNORECASE)

        # Preserve any URLs that only existed in the old references section
        for u in re.findall(r'https?://[^\s,\]\)>]+', existing_refs):
            _register(u)

        if not url_order:
            return body.rstrip() + ('\n\n' + existing_refs.strip() if existing_refs.strip() else '')

        ref_lines = ['## References', '']
        for i, u in enumerate(url_order, 1):
            domain = re.sub(r'^https?://(www\.)?', '', u).split('/')[0]
            ref_lines.append(f'{i}. {domain} — {u}')
        references = '\n'.join(ref_lines)

        return body.rstrip() + '\n\n' + references + '\n'

    def _strip_empty_sections(self, report: str) -> str:
        """
        Remove OPTIONAL report sections whose body is only a 'no data found'
        placeholder. User requirement: collect real data only — if a tool returned
        no data for a section, omit the section entirely instead of printing a
        placeholder. Required sections (Executive Summary, Introduction, Key
        Findings, Conclusion, References) are always preserved.
        """
        if not report:
            return report

        # Phrases that mark text as a "no real data" statement (not collected content)
        no_data_markers = [
            'no data found', 'no market data', 'no specific market data',
            'no specific use cases', 'no use cases found', 'no challenge',
            'no challenges', 'no information found', 'no specific data',
            'none found', 'no data was found', 'not found in web',
            'no relevant data', 'no concrete data', 'remains unquantified',
            'lack of market data', 'lack of publicly available',
            'lack of public', 'lack of specific market',
            'were found in the web search', 'was found in the web search',
            'no quantif', 'not publicly available', 'no academic papers',
            'absence of academic', 'absence of concrete', 'no detailed analys',
            'hinders a detailed', 'hinders a comprehensive', 'no in-depth',
        ]

        # Only these data-driven (optional) sections may be dropped when empty.
        # Required sections (Exec Summary, Introduction, How It Works, Key Findings,
        # Conclusion, References) are never in this list, so they're always kept.
        optional_headings = [
            'market analysis', 'market & industry data', 'market and industry data',
            'market data', 'applications & use cases', 'applications and use cases',
            'use cases', 'challenges & limitations', 'challenges and limitations',
            'limitations', 'challenges',
            'background & history', 'background and history', 'background',
            'key features & capabilities', 'key features and capabilities',
            'key features', 'features & capabilities', 'features', 'capabilities',
            'types / variants', 'types/variants', 'types', 'variants',
            'comparison with alternatives', 'comparison', 'comparisons',
            'future outlook', 'future', 'outlook',
        ]

        # Split on ## headings (capture the heading lines)
        parts = re.split(r'(?m)^(##\s+.+)$', report)
        if len(parts) < 3:
            return report  # no ## sections to process

        rebuilt = [parts[0]]  # preamble (title + anything before first ##)
        i = 1
        while i < len(parts):
            heading = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ''
            i += 2

            heading_text = heading.lstrip('#').strip().lower()
            is_optional = any(h == heading_text or h in heading_text
                              for h in optional_headings)

            if is_optional:
                # Keep only sentences that are NOT no-data statements
                sentences = re.split(r'(?<=[.!?])\s+', body.strip())
                real = [s for s in sentences
                        if s.strip() and not any(m in s.lower() for m in no_data_markers)]
                remaining_alnum = re.sub(r'[^a-z0-9]', '', ' '.join(real).lower())
                if len(remaining_alnum) < 40:
                    # Section is essentially empty / all placeholders → drop it
                    print(f"  [STRIP] Removed empty section:{heading.strip()}")
                    continue

            rebuilt.append(heading + body)

        cleaned = ''.join(rebuilt)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    def _critique_report(self, crew, report: str) -> dict:
        """
        Critique the report and extract:
        - Quality score
        - Critique details
        - Suggestions for next iteration
        """

        prompt = f"""Evaluate this report across 6 dimensions and provide specific improvement suggestions.

REPORT TO EVALUATE:
{report}

EVALUATION TASK:
Score the report on each dimension (0-100):

1. CORRECTNESS (0-100): Are facts accurate? Are sources cited?
2. CLARITY (0-100): Is writing clear and well-organized?
3. DEPTH (0-100): Is analysis thorough and comprehensive?
4. ACTIONABILITY (0-100): Are recommendations specific and practical?
5. ENGAGEMENT (0-100): Is tone professional and engaging?
6. SOURCE ATTRIBUTION (0-100): Are citations complete and proper?

PROVIDE:
- Score for each dimension with brief explanation
- 2-3 specific issues found
- Top 5 suggestions to improve the report for next iteration
- Calculate average score

FORMAT YOUR RESPONSE AS:
CORRECTNESS: [score]/100 - [reason]
CLARITY: [score]/100 - [reason]
DEPTH: [score]/100 - [reason]
ACTIONABILITY: [score]/100 - [reason]
ENGAGEMENT: [score]/100 - [reason]
SOURCE ATTRIBUTION: [score]/100 - [reason]

ISSUES FOUND:
- [Issue 1]
- [Issue 2]
- [Issue 3]

IMPROVEMENT SUGGESTIONS:
1. [Suggestion 1]
2. [Suggestion 2]
3. [Suggestion 3]
4. [Suggestion 4]
5. [Suggestion 5]

AVERAGE SCORE: [X.X]/100

End your response with XML tags:
<score>X.X</score>
<suggestions>Your top 5 improvement suggestions here</suggestions>"""

        agent = crew.critic()
        task = Task(
            description=prompt,
            agent=agent,
            expected_output="A detailed critique with scores (0-100) for each dimension and improvement suggestions"
        )
        for attempt in range(3):  # Reduced from 5 to 3 retries
            try:
                critique_text = agent.execute_task(task)
                critique_text = critique_text if isinstance(critique_text, str) else str(critique_text)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 15 if 'connection' in str(e).lower() else 8  # Reduced wait times
                print(f"[RETRY] Attempt {attempt + 1} failed: {str(e)[:60]}. Waiting {wait}s...")
                time.sleep(wait)

        # Extract score
        score = self._extract_score(critique_text)

        # Extract suggestions
        suggestions = self._extract_suggestions(critique_text)

        return {
            'critique': critique_text,
            'score': score,
            'suggestions': suggestions
        }

    def _extract_score(self, critique_text: str) -> float:
        """Extract average score from critique text using XML tags first."""

        # Try XML tag first (most reliable)
        xml_match = re.search(r'<score>([\d.]+)</score>', critique_text, re.IGNORECASE)
        if xml_match:
            try:
                return min(100.0, max(0.0, float(xml_match.group(1))))
            except ValueError:
                pass

        # Fallback: "AVERAGE SCORE: X.X/100"
        match = re.search(r'AVERAGE\s+SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*100', critique_text, re.IGNORECASE)
        if match:
            return min(100.0, max(0.0, float(match.group(1))))

        # Fallback: extract dimension scores and average
        dimension_pattern = r'^(?:CORRECTNESS|CLARITY|DEPTH|ACTIONABILITY|ENGAGEMENT|SOURCE ATTRIBUTION):\s*(\d+)\s*/\s*100'
        scores = re.findall(dimension_pattern, critique_text, re.IGNORECASE | re.MULTILINE)
        if scores:
            avg_score = sum(int(s) for s in scores) / len(scores)
            return min(100.0, max(0.0, avg_score))

        return 70.0

    def _extract_suggestions(self, critique_text: str) -> str:
        """Extract improvement suggestions using XML tags first."""

        # Try XML tag first
        xml_match = re.search(r'<suggestions>(.*?)</suggestions>', critique_text, re.IGNORECASE | re.DOTALL)
        if xml_match:
            return xml_match.group(1).strip()[:500]

        # Fallback: "IMPROVEMENT SUGGESTIONS:" section
        pattern = r'IMPROVEMENT\s+SUGGESTIONS:(.*?)(?:AVERAGE SCORE|<score>|$)'
        match = re.search(pattern, critique_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:500]

        return critique_text[:500]

    def _select_best_iteration(self):
        """Select the iteration with the highest score."""
        if not self.iterations:
            return

        best = max(self.iterations, key=lambda x: x.score)
        self.best_iteration = best

    def _generate_summary(self) -> dict:
        """Generate final summary with scores and suggestions."""
        summary = {
            'topic': self.topic,
            'timestamp': datetime.now().isoformat(),
            'num_iterations': len(self.iterations),
            'best_iteration': self.best_iteration.iteration if self.best_iteration else None,
            'best_score': self.best_iteration.score if self.best_iteration else None,
            'best_report': self.best_iteration.report if self.best_iteration else None,
            'all_iterations': [
                {
                    'iteration': it.iteration,
                    'score': it.score,
                    'suggestions': it.suggestions
                }
                for it in self.iterations
            ],
            'final_suggestions': self.best_iteration.suggestions if self.best_iteration else None
        }

        # Display summary
        self._display_summary(summary)

        return summary

    def _display_summary(self, summary: dict):
        """Display final summary to user."""
        pass  # Summary displayed in main.py

    def get_critique_history(self) -> list:
        """Get critique history for PDF generation."""
        return [{'iteration': it.iteration, 'score': it.score, 'status': '✓ Best' if it == self.best_iteration else ''} 
                for it in self.iterations]

    def save_results(self, output_dir: str = None) -> dict:
        """Save only PDF to Report folder (always uses absolute path)."""
        if output_dir is None:
            output_dir = _DEFAULT_REPORT_DIR
        print(f"\n[SAVING] Generating PDF to {output_dir}/ folder...")
        os.makedirs(output_dir, exist_ok=True)

        # Create safe filename from topic
        safe_topic = self.topic.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
        if not safe_topic:
            safe_topic = 'report'

        # Create temporary markdown file
        report_filename = f'{safe_topic}.md'
        report_path = os.path.join(output_dir, report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.best_iteration.report)

        # Fetch several topic visuals (diagrams/illustrations) to embed in the PDF
        images = self._fetch_topic_images(output_dir, max_images=4)

        # Generate Enhanced PDF
        pdf_path = None
        try:
            score = self.best_iteration.score if self.best_iteration else 0
            result = convert_report_to_pdf_professional(
                self.best_iteration.report,
                topic=self.topic,
                score=score,
                output_dir=output_dir,
                images=images,
            )
            if isinstance(result, dict):
                pdf_path = result.get("file_path")
            else:
                pdf_path = result
            pdf_filename = f'{safe_topic}.pdf'
            print(f"[OK] PDF generated: {pdf_filename}")
        except Exception as e:
            print(f"[ERROR] PDF generation failed: {e}")

        # Delete markdown file (keep only PDF)
        try:
            os.remove(report_path)
        except Exception:
            pass

        return {
            'pdf': pdf_path
        }


def run_report_refinement(research_findings: str, topic: str = "",
                         num_iterations: int = 1) -> dict:
    """
    Run the report refinement system.

    Args:
        research_findings: Research content/findings
        topic: Research topic for metadata
        num_iterations: Number of iterations (default: 1 for speed)

    Returns:
        Summary dictionary with best report and suggestions
    """
    session = ReportRefinementSession(research_findings, topic)
    summary = session.run_iterations(num_iterations)
    session.save_results()
    return summary
