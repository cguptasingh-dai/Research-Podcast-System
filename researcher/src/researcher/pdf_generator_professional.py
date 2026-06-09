"""
PROFESSIONAL PDF GENERATOR - Enhanced Design
Beautiful, modern reports with professional styling, icons, metrics, and layouts.
"""

import os
import re
from datetime import datetime
from xhtml2pdf import pisa


def _inline(text: str) -> str:
    """Convert inline markdown to HTML."""
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', text)
    text = re.sub(r'\[Source:\s*(.*?)\]', r'<span class="citation">[Source: \1]</span>', text)
    return text


def _markdown_to_html_body(markdown: str, body_figs: list = None) -> str:
    """Convert markdown to HTML body with professional formatting.

    body_figs: optional list of figure-HTML strings to distribute through the
    document — one is inserted right after each matching major section heading.
    """
    fig_queue = list(body_figs or [])
    target_sections = ('how it works', 'applications', 'use cases', 'comparison',
                        'challenges', 'key features', 'market', 'key findings')

    lines = markdown.split('\n')
    html = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        if line.startswith('# '):
            html.append(f'<h1>{_inline(line[2:].strip())}</h1>')
            i += 1
            continue

        if line.startswith('## '):
            title = line[3:].strip()
            html.append(f'<h2>📌 {_inline(title)}</h2>')
            i += 1
            # Insert a distributed figure right after a relevant section heading
            if fig_queue and any(t in title.lower() for t in target_sections):
                html.append(fig_queue.pop(0))
            continue

        if line.startswith('### '):
            html.append(f'<h3>➤ {_inline(line[4:].strip())}</h3>')
            i += 1
            continue

        if re.match(r'^[-_*]{3,}$', line.strip()):
            html.append('<div class="divider"></div>')
            i += 1
            continue

        if line.startswith('> '):
            html.append(f'<div class="highlight-box">{_inline(line[2:].strip())}</div>')
            i += 1
            continue

        if re.match(r'^[-*•+]\s+', line):
            html.append('<ul class="list-styled">')
            while i < len(lines) and re.match(r'^[-*•+]\s+', lines[i].rstrip()):
                item = re.sub(r'^[-*•+]\s+', '', lines[i].rstrip()).strip()
                html.append(f'<li>{_inline(item)}</li>')
                i += 1
            html.append('</ul>')
            continue

        if re.match(r'^\d+\.\s+', line):
            html.append('<ol class="list-styled">')
            while i < len(lines) and re.match(r'^\d+\.\s+', lines[i].rstrip()):
                item = re.sub(r'^\d+\.\s+', '', lines[i].rstrip()).strip()
                html.append(f'<li>{_inline(item)}</li>')
                i += 1
            html.append('</ol>')
            continue

        if line.lstrip().startswith('|'):
            i += 1
            continue

        if line.strip():
            html.append(f'<p>{_inline(line.strip())}</p>')
        i += 1

    # Any figures not placed after a section heading go at the end
    html.extend(fig_queue)

    return '\n'.join(html)


def _extract_stats(markdown: str) -> list:
    """
    Extract REAL statistics for visual stat cards.
    Skips hallucinated/speculative numbers — only shows stats
    that appear in sentences with real sourced language.
    Returns empty list if report looks speculative/hallucinated.
    """
    # Detect hallucinated report — these phrases mean LLM made up the data
    hallucination_signals = [
        'hypothetical', 'speculative', 'simulated', 'non-existent',
        'if it were real', 'plausible narrative', 'for the purposes of this',
        'no specific urls', 'no data found', 'could not be found',
        'no results found', 'absence of concrete data',
    ]
    markdown_lower = markdown.lower()
    if any(sig in markdown_lower for sig in hallucination_signals):
        # Report is hallucinated — show NO stat cards
        return []

    stats = []
    seen = set()

    # Only extract from lines that have citation signals (real sourced data)
    real_source_signals = ['according to', 'reported', 'source:', 'http', 'study', 'survey',
                           'research shows', 'data shows', 'analysts', 'market research',
                           'published', 'findings', 'based on']

    lines = markdown.split('\n')
    sourced_lines = []
    for line in lines:
        line_lower = line.lower()
        if any(sig in line_lower for sig in real_source_signals):
            sourced_lines.append(line)

    # Search in sourced lines first, then full markdown as fallback
    search_text = '\n'.join(sourced_lines) if sourced_lines else markdown

    # Pattern 1: Percentages with context
    pattern1 = r'(\d+(?:\.\d+)?%)[^\n]{0,60}?(growth|increase|share|adoption|rate|reduction|improvement|users|revenue)'
    for val, label in re.findall(pattern1, search_text, re.IGNORECASE):
        key = val + label
        if key not in seen and len(stats) < 4:
            stats.append({'value': val, 'label': label.title(), 'emoji': 'Up'})
            seen.add(key)

    # Pattern 2: Financial values
    pattern2 = r'\$(\d+(?:\.\d+)?)\s*(billion|million|trillion)'
    for val, unit in re.findall(pattern2, search_text, re.IGNORECASE):
        key = val + unit
        if key not in seen and len(stats) < 4:
            stats.append({'value': f'${val}', 'label': unit.title(), 'emoji': '$'})
            seen.add(key)

    return stats[:4]


def _build_stat_cards(stats: list) -> str:
    """Build professional stat cards with gradients and icons."""
    if not stats:
        return ''

    # Solid colors - xhtml2pdf doesn't render gradients
    colors = [
        '#667eea',  # Purple-blue
        '#f5576c',  # Pink-red
        '#4facfe',  # Sky blue
        '#43e97b',  # Green
        '#fa709a',  # Pink
        '#30cfd0',  # Cyan
    ]

    cards = ''
    for idx, stat in enumerate(stats):
        color = colors[idx % len(colors)]
        cards += f'''
        <div style="
            display: inline-block;
            width: 15.5%;
            background: {color};
            border-radius: 12px;
            padding: 16px 8px;
            text-align: center;
            color: white;
            margin: 6px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            vertical-align: top;
        ">
            <div style="font-size: 24pt; font-weight: 700; margin-bottom: 4px;">
                {stat['emoji']} {stat['value']}
            </div>
            <div style="font-size: 8.5pt; color: rgba(255,255,255,0.9); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">
                {stat['label']}
            </div>
        </div>
        '''

    return f'<div style="margin: 20px 0; text-align: center;">{cards}</div>'


def _build_executive_summary_box(markdown: str) -> str:
    """Extract and format executive summary."""
    summary_match = re.search(r'[Ee]xecutive\s+[Ss]ummary[:\s]+(.*?)(?=##|$)', markdown, re.DOTALL)

    if summary_match:
        summary_text = summary_match.group(1).strip()
        summary_text = '\n'.join(line for line in summary_text.split('\n') if line.strip())[:300]

        return f'''
        <div style="
            background-color: #667eea;
            color: white;
            padding: 20px;
            margin: 20px 0;
            border-left: 6px solid #f5576c;
        ">
            <h2 style="
                color: white;
                margin: 0 0 12px 0;
                padding: 0;
                border: none;
                font-size: 14pt;
            ">Executive Summary</h2>
            <p style="
                margin: 0;
                font-size: 10pt;
                line-height: 1.6;
                color: white;
            ">{_inline(summary_text)}</p>
        </div>
        '''

    return ''


def _build_key_findings_box(markdown: str) -> str:
    """Extract and format key findings."""
    findings_match = re.search(r'[Kk]ey\s+[Ff]indings?[:\s]+(.*?)(?=##|$)', markdown, re.DOTALL)

    if findings_match:
        findings_text = findings_match.group(1).strip()
        findings_lines = [line.strip() for line in findings_text.split('\n') if line.strip()][:5]

        findings_html = '<ul style="margin: 0; padding-left: 20px;">'
        for line in findings_lines:
            if line:
                findings_html += f'<li style="margin-bottom: 6px; color: #2c3e50;">{_inline(line)}</li>'
        findings_html += '</ul>'

        return f'''
        <div style="
            background: #f8f9ff;
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            border-left: 6px solid #667eea;
        ">
            <h3 style="
                color: #667eea;
                margin: 0 0 12px 0;
                padding: 0;
                border: none;
                font-size: 12pt;
            ">🎯 Key Findings</h3>
            {findings_html}
        </div>
        '''

    return ''


def _get_professional_css() -> str:
    """Professional CSS styling with modern design."""
    return """
    * {
        margin: 0;
        padding: 0;
    }

    body {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        font-size: 11pt;
        color: #2c3e50;
        line-height: 1.8;
        background-color: white;
    }

    /* Cover Page - Fixed: xhtml2pdf does NOT support gradients, use solid color */
    .cover {
        text-align: center;
        padding: 80px 60px;
        background-color: #ffffff;
        page-break-after: always;
        color: #000000;
    }

    .cover-badge {
        display: inline-block;
        background-color: #5468c5;
        border: 2px solid #ffffff;
        color: white;
        font-size: 10pt;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 8px 24px;
        margin-bottom: 30px;
    }

    .cover-title {
        font-size: 38pt;
        font-weight: 800;
        color: #000000;
        line-height: 1.3;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .cover-divider {
        width: 100px;
        height: 4px;
        background-color: #000000;
        margin: 24px auto;
    }

    .cover-subtitle {
        font-size: 14pt;
        color: #ffffff;
        margin-bottom: 60px;
        font-weight: 300;
    }

    .cover-meta {
        font-size: 11pt;
        color: #f0f0f0;
        margin-top: 40px;
    }

    .cover-meta strong {
        color: white;
        font-weight: 600;
    }

    /* Content */
    .content {
        padding: 0;
    }

    /* Headings */
    h1 {
        font-size: 20pt;
        font-weight: 800;
        color: #667eea;
        margin-top: 30px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 4px solid #667eea;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    h2 {
        font-size: 14pt;
        font-weight: 700;
        color: #764ba2;
        margin-top: 22px;
        margin-bottom: 12px;
        padding-left: 12px;
        border-left: 5px solid #667eea;
    }

    h3 {
        font-size: 12pt;
        font-weight: 700;
        color: #667eea;
        margin-top: 16px;
        margin-bottom: 10px;
    }

    /* Paragraphs */
    p {
        margin-bottom: 12px;
        text-align: justify;
        line-height: 1.8;
    }

    /* Lists */
    .list-styled {
        margin: 12px 0 16px 24px;
        line-height: 1.9;
    }

    .list-styled li {
        margin-bottom: 8px;
        color: #2c3e50;
    }

    /* Highlight Box - solid color for xhtml2pdf compatibility */
    .highlight-box {
        background-color: #f5576c;
        color: white;
        padding: 16px;
        margin: 16px 0;
        border-left: 5px solid #c0392b;
        font-style: italic;
        line-height: 1.7;
    }

    /* Code */
    code {
        background-color: #f4f6f9;
        color: #d63031;
        padding: 2px 6px;
        font-size: 10pt;
        font-family: 'Courier New', monospace;
        border-radius: 3px;
    }

    /* Citations */
    .citation {
        background-color: #e8f4fd;
        color: #667eea;
        font-size: 9pt;
        padding: 2px 6px;
        border-radius: 3px;
    }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-radius: 6px;
        overflow: hidden;
    }

    table th {
        background-color: #667eea;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
        font-size: 10.5pt;
    }

    table td {
        padding: 10px 12px;
        border-bottom: 1px solid #e0e6f2;
        color: #2c3e50;
    }

    table tr:last-child td {
        border-bottom: none;
    }

    /* Divider */
    .divider {
        height: 2px;
        background-color: #667eea;
        margin: 20px 0;
    }

    /* Strong text */
    strong {
        color: #667eea;
        font-weight: 700;
    }

    /* Page */
    @page {
        size: A4;
        margin: 20mm 18mm 20mm 18mm;
        @frame footer {
            -pdf-frame-content: footer_content;
            bottom: 10mm;
            margin-left: 18mm;
            margin-right: 18mm;
            height: 10mm;
        }
    }

    .footer {
        font-size: 9pt;
        color: #999;
        border-top: 1px solid #ddd;
        padding-top: 8px;
        text-align: center;
    }
    """


def _figure_html(topic: str, data_uri: str, source: str, caption: str = None) -> str:
    """Build a single centered figure (image + caption) for embedding in the PDF."""
    if not data_uri:
        return ''
    src_note = f"Source: {source}" if source else "Source: web image search"
    cap = caption or topic.title()
    return f'''
    <div style="text-align: center; margin: 16px 0 22px 0;">
        <img src="{data_uri}" style="max-width: 90%; height: auto; border: 1px solid #d7ddea;" />
        <div style="font-size: 8.5pt; color: #888; margin-top: 6px; font-style: italic;">
            Figure: {cap} — {src_note}
        </div>
    </div>
    '''


def _build_full_html_professional(topic: str, body_html: str, stats: list, score: float = 0) -> str:
    """Build professional HTML document."""
    stat_cards = _build_stat_cards(stats)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style type="text/css">
{_get_professional_css()}
</style>
</head>
<body>

<!-- COVER PAGE (headline only) -->
<div class="cover">
    <h1 class="cover-title">{topic.title()}</h1>
    <div class="cover-divider"></div>
</div>

<!-- KEY METRICS -->
{stat_cards}

<!-- MAIN CONTENT -->
<div class="content">
{body_html}
</div>

<!-- FOOTER (removed) -->
<div id="footer_content" class="footer"></div>

</body>
</html>"""


def convert_report_to_pdf_professional(
    markdown_report: str,
    topic: str,
    score: float = 0,
    output_dir: str = "Report",
    images: list = None
) -> dict:
    """
    Convert markdown report to professional PDF.

    Args:
        markdown_report: Markdown content
        topic: Report topic
        score: Quality score (0-100)
        output_dir: Output directory

    Returns:
        Dict with file_path and success status
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename
    safe_topic = re.sub(r'[^\w\s-]', '', topic.lower())
    safe_topic = re.sub(r'[-\s]+', '_', safe_topic.strip())
    filename = f"{safe_topic}.pdf"
    filepath = os.path.join(output_dir, filename)

    # Extract stats
    stats = _extract_stats(markdown_report)

    # Build figures from the fetched images. No image at the start — all figures
    # are distributed after relevant sections inside the body.
    images = images or []
    figs = [_figure_html(topic, im.get('data_uri'), im.get('source'))
            for im in images if im.get('data_uri')]

    # Convert markdown to HTML (with distributed figures)
    body_html = _markdown_to_html_body(markdown_report, figs)

    # Build full HTML document
    html_content = _build_full_html_professional(topic, body_html, stats, score)

    # Convert HTML to PDF
    try:
        with open(filepath, 'wb') as pdf_file:
            pisa.CreatePDF(html_content, pdf_file)

        return {
            "success": True,
            "file_path": filepath,
            "filename": filename,
            "topic": topic,
            "score": score
        }

    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": filepath
        }


# Backwards compatibility - use this function name
convert_report_to_pdf_enhanced = convert_report_to_pdf_professional
