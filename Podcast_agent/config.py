"""
Podcast Agent LLM Configuration
Model: nvidia/nemotron-3-ultra-550b-a55b
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_shared import APIConfig, LLMConfig

load_dotenv()


def get_llm():
    """
    Returns ChatNVIDIA client for Nemotron Ultra 550B.
    No connectivity test on init - just creates client object.
    """
    api_key = APIConfig.get_nvidia_api_key()

    client = ChatNVIDIA(
        model=LLMConfig.PODCAST_LLM_MODEL,
        api_key=api_key,
        temperature=LLMConfig.PODCAST_TEMPERATURE,
        top_p=LLMConfig.PODCAST_TOP_P,
        max_completion_tokens=LLMConfig.PODCAST_MAX_TOKENS,  # 4096
        frequency_penalty=LLMConfig.FREQUENCY_PENALTY,     # 0.0
        presence_penalty=LLMConfig.PRESENCE_PENALTY,       # 0.0
        model_kwargs={
            "reasoning_budget": LLMConfig.REASONING_BUDGET,
            "chat_template_kwargs": {"enable_thinking": LLMConfig.ENABLE_THINKING},
        },
    )

    print(f"[LLM] Podcast Model:      {LLMConfig.PODCAST_LLM_MODEL}")
    print(f"[LLM] Temperature:        {LLMConfig.PODCAST_TEMPERATURE}")
    print(f"[LLM] Top_p:              {LLMConfig.PODCAST_TOP_P}")
    print(f"[LLM] Max Tokens:         {LLMConfig.PODCAST_MAX_TOKENS}")
    print(f"[LLM] Reasoning Budget:   {LLMConfig.REASONING_BUDGET}")
    return client


# Ensure podcast output directory exists at project root
_project_root = Path(__file__).parent.parent
(_project_root / "podcasts").mkdir(exist_ok=True)
