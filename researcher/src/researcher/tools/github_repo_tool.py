"""
GitHub Public Repo Tool for CrewAI Researcher
==============================================
No GitHub token required — uses public GitHub API.

Capabilities:
  1. Search public repos by topic/keyword (sorted by stars)
  2. Read README.md from each repo for rich content
  3. Extract: description, stars, topics, language, README content
"""

import os
import re
import base64
import requests
from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


# GitHub public API rate limit: 60 req/hour (unauthenticated)
# With GITHUB_TOKEN:           5000 req/hour
GITHUB_API = "https://api.github.com"
TIMEOUT = 15  # seconds per request


def _get_headers() -> dict:
    """Build headers — adds token if available for higher rate limit."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN", "").strip().strip('"').strip("'")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_readme(full_name: str) -> str:
    """
    Fetch and decode README.md from a repo.
    Returns plain text content (max 3000 chars).
    """
    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{full_name}/readme",
            headers=_get_headers(),
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return ""

        data = resp.json()
        content_b64 = data.get("content", "")
        if not content_b64:
            return ""

        # GitHub returns base64-encoded content
        raw = base64.b64decode(content_b64).decode("utf-8", errors="ignore")

        # Strip markdown images, badges, HTML tags
        raw = re.sub(r'!\[.*?\]\(.*?\)', '', raw)            # images
        raw = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', raw) # badge links
        raw = re.sub(r'<[^>]+>', '', raw)                    # HTML tags
        raw = re.sub(r'\s{3,}', '\n\n', raw)                 # excess whitespace

        # Remove emoji / non-ASCII chars (safe for Windows console + LLM)
        raw = raw.encode('ascii', errors='ignore').decode('ascii')
        raw = re.sub(r'\s{3,}', '\n\n', raw)                 # clean again

        return raw[:1500].strip()

    except Exception:
        return ""


def search_github_repos(query: str, max_repos: int = 5) -> str:
    """
    Search GitHub public repos by query, fetch README for each.
    Returns rich structured text for the researcher agent.

    Args:
        query: Search keyword (e.g. "langgraph multi-agent")
        max_repos: Number of top repos to read (default 5)

    Returns:
        Formatted string with repo info + README content
    """
    try:
        # Search repos sorted by stars
        search_url = f"{GITHUB_API}/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(max_repos, 10),
        }
        resp = requests.get(
            search_url,
            headers=_get_headers(),
            params=params,
            timeout=TIMEOUT,
        )

        if resp.status_code == 403:
            return "[GitHub] Rate limit exceeded. Add GITHUB_TOKEN to .env for 5000 req/hour."
        if resp.status_code != 200:
            return f"[GitHub] Search failed: HTTP {resp.status_code}"

        items = resp.json().get("items", [])
        if not items:
            return f"[GitHub] No public repos found for: {query}"

        # Build output
        sections = []
        sections.append(f"=== GitHub Public Repos: '{query}' (top {len(items)} by stars) ===\n")

        for i, repo in enumerate(items[:max_repos], 1):
            full_name   = repo.get("full_name", "")
            description = repo.get("description", "No description") or "No description"
            stars       = repo.get("stargazers_count", 0)
            language    = repo.get("language", "Unknown")
            topics      = ", ".join(repo.get("topics", [])[:8]) or "none"
            html_url    = repo.get("html_url", "")
            updated_at  = repo.get("updated_at", "")[:10]  # just date

            header = (
                f"--- REPO {i}: {full_name} ---\n"
                f"URL:         {html_url}\n"
                f"Stars:       {stars:,}\n"
                f"Language:    {language}\n"
                f"Updated:     {updated_at}\n"
                f"Topics:      {topics}\n"
                f"Description: {description}\n"
            )

            # Fetch README
            print(f"    [GitHub] Reading README: {full_name}...")
            readme = _fetch_readme(full_name)
            if readme:
                readme_section = f"README:\n{readme}"
            else:
                readme_section = "README: Not available"

            sections.append(header + readme_section)

        result = "\n\n".join(sections)
        print(f"  [GitHub] Fetched {len(items)} repos + READMEs for '{query}'")
        return result

    except requests.exceptions.Timeout:
        return "[GitHub] Request timed out."
    except Exception as e:
        return f"[GitHub] Error: {str(e)[:100]}"


# ============================================================================
# CrewAI Tool (BaseTool subclass)
# ============================================================================

class GithubRepoInput(BaseModel):
    """Input schema for GithubRepoSearchTool."""
    query: str = Field(
        ...,
        description="Search query for GitHub repos. E.g. 'langgraph agent', 'RAG pipeline python'."
    )
    max_repos: int = Field(
        default=2,
        description="How many top repos to read (1-5). Default 2 for speed.",
        ge=1,
        le=5,
    )


class GithubRepoSearchTool(BaseTool):
    """
    Search GitHub public repos by topic and read their README files.
    No API key required — uses public GitHub API.
    Optionally add GITHUB_TOKEN to .env for higher rate limits.

    Use this tool when you need:
    - Popular open-source implementations of a technology
    - Real-world code examples and project descriptions
    - Community adoption signals (stars, topics, activity)
    - Technical details from README documentation
    """

    name: str = "GithubRepoSearchTool"
    description: str = (
        "Search GitHub public repos by keyword and read their README files. "
        "Returns top repos sorted by stars with full README content. "
        "No API key required. Use for: finding real implementations, "
        "code examples, project descriptions, community adoption. "
        "Input: {query: 'search term', max_repos: 4}"
    )
    args_schema: Type[BaseModel] = GithubRepoInput

    def _run(self, query: str, max_repos: int = 4) -> str:
        return search_github_repos(query, max_repos)
