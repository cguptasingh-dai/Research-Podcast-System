# A2A Research & Podcast System

A multi-agent AI pipeline that takes a topic and automatically produces:

- **PDF Research Report** — real web sources, numbered citations, embedded visuals
- **Audio Podcast** — dual-voice host/expert Q&A, synthesised with Sarvam AI TTS

Built on the **A2A (Agent-to-Agent) Protocol** — a lightweight in-memory task bus that coordinates agents without external message queues.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

```env
NVIDIA_API_KEY=your_nvidia_key
SERPER_API_KEY=your_serper_key
SARVAM_API_KEY=your_sarvam_key
TAVILY_API_KEY=your_tavily_key          # optional
EXA_API_KEY=your_exa_key               # optional
```

### 3. Start all services (4 terminals)

```bash
# Terminal 1 — Orchestrator (port 8000)
python fastapi_clean.py

# Terminal 2 — Researcher agent (port 8001)
cd researcher
python fastapi_researcher.py

# Terminal 3 — Podcast agent (port 8002)
cd Podcast_agent
python fastapi_podcast.py

# Terminal 4 — Streamlit UI (port 8501)
streamlit run streamlit_app_simple.py
```

### 4. Open the UI

```
http://localhost:8501
```

Enter a topic → click **Research Only** or **Both** → wait ~3–8 min → get PDF + audio.

---

## Project Structure

```
A2a/
├── fastapi_clean.py            # Orchestrator — FastAPI :8000
├── a2a_protocol.py             # A2A task bus (AgentCard, Task, TaskStatus)
├── config_shared.py            # Shared LLM config (model, temp, tokens)
├── streamlit_app_simple.py     # Streamlit UI — :8501
│
├── researcher/                 # Research Agent — FastAPI :8001
│   ├── fastapi_researcher.py
│   └── src/researcher/
│       ├── report_refinement.py    # Web research + LLM report writer
│       ├── pdf_generator_*.py      # PDF renderer (xhtml2pdf)
│       └── tools/
│           └── github_repo_tool.py
│
├── Podcast_agent/              # Podcast Agent — FastAPI :8002
│   ├── fastapi_podcast.py
│   ├── graph_fixed.py          # LangGraph 5-node pipeline
│   ├── state.py                # TypedDict state schema
│   ├── tools.py                # LangChain @tool functions
│   ├── config.py               # Podcast LLM client
│   └── agents/
│       ├── audio_sarvam_dual_voice.py  # Dual-voice TTS (rahul + meera)
│       ├── audio_sarvam_final.py       # Single-voice TTS fallback
│       ├── question_generator.py       # Fact extraction + Q&A generation
│       ├── answer_generator.py
│       ├── conversation_formatter.py
│       └── sarvam_config.py            # Sarvam AI voice settings
│
├── Report/                     # Generated PDF reports (git-ignored)
├── podcasts/                   # Generated WAV audio files (git-ignored)
├── .env                        # API keys — never commit (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How It Works

```
User enters topic
       │
       ▼
Streamlit UI  ──POST /pipeline/run──▶  Orchestrator (:8000)
                                              │
                          ┌───────────────────┤
                          │                   │
                          ▼                   ▼
               Researcher (:8001)     Podcast (:8002)
               7× Serper search       LangGraph pipeline
               Scrape + GitHub        Extract facts
               Quality gate           Generate Q&A
               LLM writes report      Validate pairs
               PDF rendered           Sarvam TTS
                          │                   │
                          ▼                   ▼
                   Report/<topic>.pdf   podcasts/<topic>.wav
```

---

## API Reference

All endpoints on the **Orchestrator** at `http://localhost:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + registered agents |
| `GET` | `/agents` | List registered agents |
| `POST` | `/tasks/research` | Start research-only task |
| `POST` | `/tasks/podcast` | Start podcast-only task |
| `POST` | `/pipeline/run` | Start full pipeline (research + podcast) |
| `GET` | `/tasks/{task_id}` | Poll task status / get result |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Models & APIs Used

| Service | Purpose |
|---------|---------|
| **NVIDIA Nemotron Llama-3.3 Super 49B** | Report writing, fact extraction, Q&A generation |
| **Serper Google Search** | 7 real web search queries per topic |
| **Sarvam AI bulbul:v3** | Text-to-speech — rahul (host) + meera (expert) |
| **BeautifulSoup** | Web page scraping |
| **GitHub API** | Repository search + README extraction |
| **LangGraph** | 5-node podcast pipeline state machine |
| **FastAPI + Uvicorn** | All 3 agent HTTP servers |
| **xhtml2pdf** | Markdown → PDF rendering |

---

## Configuration

Edit `config_shared.py` to change LLM settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `PODCAST_LLM_MODEL` | `meta/llama-3.3-70b-instruct` | NVIDIA model |
| `PODCAST_TEMPERATURE` | `0.4` | Creativity (lower = more factual) |
| `PODCAST_MAX_TOKENS` | `4096` | Max tokens per LLM call |

Edit `Podcast_agent/agents/sarvam_config.py` to change voices:

| Setting | Default | Options |
|---------|---------|---------|
| `SPEAKER_QUESTIONS` | `shubh` | `rahul`, `rohan`, `aditya` |
| `SPEAKER_ANSWERS` | `meera` | `meera`, `kavya`, `priya` |

---

## Requirements

- Python 3.10+
- NVIDIA API key (free tier available at [build.nvidia.com](https://build.nvidia.com))
- Serper API key ([serper.dev](https://serper.dev))
- Sarvam AI key ([sarvam.ai](https://sarvam.ai))
