import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
from pathlib import Path

# Import shared configuration
# __file__ = researcher/src/researcher/crew.py → 4 parents up = project root (A2a/)
_project_root = str(Path(__file__).parent.parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from config_shared import APIConfig, LLMConfig

load_dotenv()


# ============================================================================
# SMART TOOLKIT BUILDER
# Auto-loads tools based on available API keys and packages.
# Falls back gracefully if a tool can't be initialized.
# ============================================================================

def _build_researcher_toolkit() -> list:
    """
    Build the full researcher toolkit with graceful fallback.
    Tools are added only if their dependencies and API keys are present.

    Priority order (best content quality first):
      1. EXASearchTool        - Neural semantic search (best for AI topics)
      2. SerperDevTool        - Google search results (reliable fallback)
      3. ScrapeWebsiteTool    - Scrape raw HTML from any URL
      4. GithubRepoSearchTool - Public repos + README reader (no key needed)
      5. GithubSearchTool     - Code/issues/PRs search (needs GITHUB_TOKEN)
    """
    tools = []
    loaded = []
    skipped = []

    # ------------------------------------------------------------------
    # 1. EXASearchTool - Semantic neural search (best quality)
    # ------------------------------------------------------------------
    try:
        from crewai_tools import EXASearchTool
        exa_key = os.getenv("EXA_API_KEY", "").strip().strip('"').strip("'")
        if exa_key:
            tools.append(EXASearchTool())
            loaded.append("EXASearchTool (semantic neural search)")
        else:
            skipped.append("EXASearchTool (no EXA_API_KEY in .env)")
    except Exception as e:
        skipped.append(f"EXASearchTool ({str(e)[:50]})")

    # ------------------------------------------------------------------
    # 2. SerperDevTool - Google search via Serper API (always available)
    # ------------------------------------------------------------------
    try:
        from crewai_tools import SerperDevTool
        tools.append(SerperDevTool())
        loaded.append("SerperDevTool (Google search via Serper)")
    except Exception as e:
        skipped.append(f"SerperDevTool ({str(e)[:50]})")

    # ------------------------------------------------------------------
    # 3. ScrapeWebsiteTool - Scrape any URL (raw HTML → text)
    # ------------------------------------------------------------------
    try:
        from crewai_tools import ScrapeWebsiteTool
        tools.append(ScrapeWebsiteTool())
        loaded.append("ScrapeWebsiteTool (scrape any URL)")
    except Exception as e:
        skipped.append(f"ScrapeWebsiteTool ({str(e)[:50]})")

    # ------------------------------------------------------------------
    # 4. GithubRepoSearchTool - Find public repos + read READMEs (NO key needed)
    #    Also tries GithubSearchTool (crewai_tools) if GITHUB_TOKEN is set
    # ------------------------------------------------------------------
    try:
        from researcher.tools.github_repo_tool import GithubRepoSearchTool
        tools.append(GithubRepoSearchTool())
        loaded.append("GithubRepoSearchTool (public repos + README reader, no key needed)")
    except Exception as e:
        skipped.append(f"GithubRepoSearchTool ({str(e)[:50]})")

    # 5. crewai GithubSearchTool for code/issue search if token present
    try:
        from crewai_tools import GithubSearchTool
        gh_token = os.getenv("GITHUB_TOKEN", "").strip().strip('"').strip("'")
        if gh_token:
            tools.append(GithubSearchTool(gh_token=gh_token))
            loaded.append("GithubSearchTool (code/issues/PRs search, token provided)")
        else:
            skipped.append("GithubSearchTool/crewai (add GITHUB_TOKEN for code-level search)")
    except Exception as e:
        skipped.append(f"GithubSearchTool ({str(e)[:50]})")

    # Print summary
    print(f"\n[TOOLKIT] Researcher Tools Loaded: {len(tools)}")
    for t in loaded:
        print(f"  [OK]   {t}")
    for s in skipped:
        print(f"  [SKIP] {s}")
    print()

    return tools


def _build_writer_toolkit() -> list:
    """
    Toolkit for the report writer agent — INTENTIONALLY EMPTY.

    The writer must compose the report STRICTLY from the research findings passed
    to it (anti-hallucination requirement: no fake data). Giving it live search
    tools lets it introduce facts that were never collected/sourced and adds
    latency. A pure-LLM writer grounded only in the findings is safer and faster.
    """
    return []


def get_llm():
    """
    Get Nemotron Ultra 550B for CrewAI agents WITH REASONING ENABLED.
    Increased max_tokens to accommodate both thinking and tool calls.
    """
    import warnings
    warnings.filterwarnings("ignore")

    try:
        api_key = APIConfig.get_nvidia_api_key()

        # CrewAI with reasoning enabled
        print(f"[LLM] Researcher Model:  {LLMConfig.NVIDIA_MODEL}")
        print(f"[LLM] Temperature:       {LLMConfig.TEMPERATURE}")
        print(f"[LLM] Max Tokens:        {LLMConfig.MAX_TOKENS}")
        print(f"[LLM] Reasoning Budget:  {LLMConfig.REASONING_BUDGET}")
        return LLM(
            model=LLMConfig.NVIDIA_MODEL,
            api_key=api_key,
            base_url=LLMConfig.NVIDIA_BASE_URL,
            temperature=LLMConfig.TEMPERATURE,            # 0.6 - NVIDIA recommended
            top_p=LLMConfig.TOP_P,                        # 0.95 - NVIDIA recommended
            max_tokens=LLMConfig.MAX_TOKENS,              # 4096
            frequency_penalty=LLMConfig.FREQUENCY_PENALTY,  # 0.0
            presence_penalty=LLMConfig.PRESENCE_PENALTY,    # 0.0
            timeout=LLMConfig.TIMEOUT,                     # 150s
            extra_headers={
                "reasoning_budget": str(LLMConfig.REASONING_BUDGET),
            }
        )
    except Exception as e:
        print(f"[ERROR] LLM init failed: {str(e)[:100]}")
        raise


def get_chat_nvidia():
    """
    Get ChatNVIDIA with reasoning - for direct calls outside CrewAI.
    """
    import warnings
    warnings.filterwarnings("ignore")

    api_key = APIConfig.get_nvidia_api_key()

    return ChatNVIDIA(
        model=LLMConfig.NVIDIA_MODEL,
        api_key=api_key,
        temperature=LLMConfig.TEMPERATURE,
        top_p=LLMConfig.TOP_P,
        max_completion_tokens=LLMConfig.MAX_TOKENS,
        frequency_penalty=LLMConfig.FREQUENCY_PENALTY,
        presence_penalty=LLMConfig.PRESENCE_PENALTY,
        model_kwargs={
            "reasoning_budget": LLMConfig.REASONING_BUDGET,
            "chat_template_kwargs": {"enable_thinking": LLMConfig.ENABLE_THINKING},
        },
    )


@CrewBase
class researcher():
    """Researcher crew using Nemotron Ultra 550B"""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researchers(self) -> Agent:
        """Senior researcher with full web search toolkit"""
        # Build fresh toolkit per agent call (thread-safe, no shared state)
        local_toolkit = _build_researcher_toolkit()
        return Agent(
            config=self.agents_config['researchers'],
            llm=get_llm(),
            tools=local_toolkit,
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def report_writer(self) -> Agent:
        """Report writer - uses research findings + fact verification"""
        # Writer uses lighter toolkit for fact-checking only
        local_toolkit = _build_writer_toolkit()
        return Agent(
            config=self.agents_config['report_writer'],
            llm=get_llm(),
            tools=local_toolkit,
            allow_delegation=False,
        )

    @agent
    def critic(self) -> Agent:
        """Quality critic - evaluates report"""
        return Agent(
            config=self.agents_config['critic'],
            llm=get_llm(),
            tools=[],
            allow_delegation=False,
        )

    @task
    def deep_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['deep_research_task'],
        )

    @task
    def report_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['report_writing_task'],
        )

    @task
    def critique_task(self) -> Task:
        return Task(
            config=self.tasks_config['critique_task'],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def research_only_crew(self) -> Crew:
        """
        Build a crew with ONLY research + report tasks (no critique).
        Critique needs {report} variable which isn't in inputs - causes failure.
        Use this for the kickoff() call.
        """
        return Crew(
            agents=[self.researchers(), self.report_writer()],
            tasks=[self.deep_research_task(), self.report_writing_task()],
            process=Process.sequential,
            verbose=True,
        )
