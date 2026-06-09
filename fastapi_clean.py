"""
FastAPI - Research Agent + Podcast Agent via A2A Protocol
Two modes: Research Only | Research + Podcast (Pipeline)
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Windows: UTF-8 stdout so in-process podcast logs don't crash on Unicode
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load .env so the in-process podcast uses current keys (e.g. SARVAM_API_KEY)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")

sys.path.insert(0, str(Path(__file__).parent))
from a2a_protocol import AgentCard, TaskStatus, a2a

# ============================================================================
# MODELS
# ============================================================================


class ResearchRequest(BaseModel):
    topic: str
    num_iterations: int = 1  # 1 iteration = faster (was 2)


class PodcastRequest(BaseModel):
    report_content: str
    topic: str


class PipelineRequest(BaseModel):
    topic: str


class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="Research & Podcast Agent API",
    version="1.0.0",
    docs_url="/docs",
    description="A2A multi-agent pipeline: web research → PDF report → dual-voice podcast",
)
startup_time = datetime.utcnow()

# ============================================================================
# MIDDLEWARE
# ============================================================================

# 1. CORS — allow Streamlit UI (any origin) and EC2 public IP to call this API
#    On EC2: set ALLOWED_ORIGINS=https://yourdomain.com in .env for production
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Trusted Hosts — prevent HTTP Host header attacks on EC2
#    On EC2: set ALLOWED_HOSTS=your-ec2-public-ip,yourdomain.com in .env
_raw_hosts = os.getenv("ALLOWED_HOSTS", "*")
ALLOWED_HOSTS = [h.strip() for h in _raw_hosts.split(",")] if _raw_hosts != "*" else ["*"]

if ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# 3. Request Logging — logs every request with method, path, status, duration
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(f"Unhandled error: {exc}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    duration_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({duration_ms:.1f}ms) | client={request.client.host if request.client else 'unknown'}"
    )
    return response

# 4. Global Exception Handler — returns clean JSON instead of crash on EC2
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Check server logs."},
    )


@app.on_event("startup")
async def startup():
    a2a.register_agent(AgentCard(
        agent_id="researcher-001",
        name="Researcher",
        description="Research Agent — iterative report generation with critique loop",
        version="1.0.0",
        capabilities=["research"],
        api_endpoint="http://localhost:8001",
        created_at=datetime.utcnow().isoformat()
    ))
    a2a.register_agent(AgentCard(
        agent_id="podcast-001",
        name="Podcaster",
        description="Podcast Agent — converts report to Q&A audio podcast",
        version="1.0.0",
        capabilities=["podcast"],
        api_endpoint="http://localhost:8002",
        created_at=datetime.utcnow().isoformat()
    ))
    print("Agents registered: researcher-001, podcast-001")


# ============================================================================
# AGENT RUNNERS (async wrappers around blocking sync code)
# ============================================================================


async def run_researcher(task_id: str, topic: str, iterations: int) -> dict:
    """Run research agent — import + execution both in thread pool to never block event loop."""
    try:
        a2a.update_task_status(task_id, TaskStatus.IN_PROGRESS)

        researcher_src = str(Path(__file__).parent / "researcher" / "src")

        def _sync_research():
            # ALL heavy imports happen inside thread - never blocks event loop
            if researcher_src not in sys.path:
                sys.path.insert(0, researcher_src)
            from researcher.report_refinement import run_report_refinement
            return run_report_refinement(None, topic, iterations)

        # Run everything (import + LLM calls) in thread pool
        summary = await asyncio.to_thread(_sync_research)

        safe_topic = summary.get("topic", topic).replace(" ", "_").lower()
        result = {
            "topic": summary.get("topic", topic),
            "score": summary.get("best_score", 0),
            "file": f"Report/{safe_topic}.pdf",
            "report_content": summary.get("best_report", ""),
        }

        a2a.complete_task(task_id, result)
        return result

    except Exception as e:
        a2a.fail_task(task_id, str(e))
        raise


async def run_podcast(task_id: str, report_content: str, topic: str) -> dict:
    """Run podcast agent — uses thread pool so event loop is never blocked."""
    try:
        a2a.update_task_status(task_id, TaskStatus.IN_PROGRESS)

        podcast_dir = str(Path(__file__).parent / "Podcast_agent")

        def _sync_podcast():
            # Ensure Podcast_agent/ is on sys.path before import
            if podcast_dir not in sys.path:
                sys.path.insert(0, podcast_dir)

            # Clear stale cached modules so re-imports always use correct path
            for mod in ["graph_fixed", "state", "tools", "config"]:
                sys.modules.pop(mod, None)

            from graph_fixed import Pipeline

            state = {
                "report_content": report_content,
                "topic": topic,
                "questions": [],
                "answers": [],
                "qa_pairs": [],
                "conversation_text": "",
                "agent_logs": [],
                "audio_file_path": "",
                "transcript": "",
            }
            return Pipeline().run(state)

        final = await asyncio.to_thread(_sync_podcast)

        result = {
            "topic": final.get("topic", topic),
            "questions": final.get("questions", []),
            "answers": final.get("answers", []),
            "audio": final.get("audio_file_path", ""),
        }

        a2a.complete_task(task_id, result)
        return result

    except Exception as e:
        a2a.fail_task(task_id, str(e))
        raise





# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get("/health")
async def health():
    agent_count = len(a2a.list_agents())
    uptime = (datetime.utcnow() - startup_time).total_seconds()
    return {
        "status": "ok",
        "version": "1.0.0",
        "agents_registered": agent_count,
        "tasks_total": len(a2a.tasks),
        "uptime_seconds": round(uptime, 1),
        "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
        "environment": os.getenv("APP_ENV", "development"),
    }


@app.get("/agents")
async def get_agents():
    return {
        "agents": [
            {"id": ag.agent_id, "name": ag.name, "capabilities": ag.capabilities}
            for ag in a2a.list_agents()
        ]
    }


@app.post("/tasks/research", response_model=TaskResponse, status_code=202)
async def research(req: ResearchRequest, bg: BackgroundTasks):
    """Start research-only task."""
    task = a2a.create_task(
        source_agent="api",
        target_agent="researcher-001",
        task_type="research",
        payload={"topic": req.topic, "iterations": req.num_iterations},
    )
    bg.add_task(run_researcher, task.task_id, req.topic, req.num_iterations)
    return TaskResponse(task_id=task.task_id, status="pending")


@app.post("/tasks/podcast", response_model=TaskResponse, status_code=202)
async def podcast(req: PodcastRequest, bg: BackgroundTasks):
    """Start podcast-only task."""
    task = a2a.create_task(
        source_agent="api",
        target_agent="podcast-001",
        task_type="podcast",
        payload={"report": req.report_content, "topic": req.topic},
    )
    bg.add_task(run_podcast, task.task_id, req.report_content, req.topic)
    return TaskResponse(task_id=task.task_id, status="pending")


@app.post("/pipeline/run", response_model=TaskResponse, status_code=202)
async def pipeline(req: PipelineRequest, bg: BackgroundTasks):
    """
    Run full pipeline: Research → Podcast (sequential).
    Returns a pipeline_task_id that completes only when both agents finish.
    The podcast agent receives the actual research report content.
    Iterations (2) are applied only to the research phase.
    """
    pipeline_task = a2a.create_task(
        source_agent="api",
        target_agent="pipeline",
        task_type="pipeline",
        payload={"topic": req.topic},
    )

    async def _pipeline_flow():
        try:
            # Step 1: Research (with 1 iteration for speed - saves 5 minutes!)
            iterations = 1  # Reduced from 2 to 1 for FAST MODE
            research_task = a2a.create_task(
                source_agent="pipeline",
                target_agent="researcher-001",
                task_type="research",
                payload={"topic": req.topic, "iterations": iterations},
            )
            research_result = await run_researcher(
                research_task.task_id, req.topic, iterations
            )

            # Step 2: Podcast using actual research report
            report_content = research_result.get("report_content", "").strip()
            if not report_content:
                a2a.fail_task(pipeline_task.task_id, "Research returned empty report")
                return
            podcast_task = a2a.create_task(
                source_agent="pipeline",
                target_agent="podcast-001",
                task_type="podcast",
                payload={"report": report_content, "topic": req.topic},
            )
            podcast_result = await run_podcast(
                podcast_task.task_id, report_content, req.topic
            )

            # Combined result for the pipeline task
            combined = {
                "topic": research_result.get("topic"),
                "score": research_result.get("score"),
                "file": research_result.get("file"),
                "questions": podcast_result.get("questions", []),
                "answers": podcast_result.get("answers", []),
                "audio": podcast_result.get("audio", ""),
            }
            a2a.complete_task(pipeline_task.task_id, combined)

        except Exception as e:
            a2a.fail_task(pipeline_task.task_id, str(e))

    bg.add_task(_pipeline_flow)
    return TaskResponse(task_id=pipeline_task.task_id, status="pending")


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    task = a2a.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    progress = {
        TaskStatus.PENDING: 0,
        TaskStatus.IN_PROGRESS: 50,
        TaskStatus.COMPLETED: 100,
        TaskStatus.FAILED: 0,
    }.get(task.status, 0)

    return TaskResponse(
        task_id=task.task_id,
        status=task.status.value,
        result=task.result,
        error=task.error,
        progress=progress,
    )


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "id": t.task_id,
                "type": t.task_type,
                "status": t.status.value,
                "created": t.created_at,
            }
            for t in a2a.tasks.values()
        ]
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn, socket, subprocess

    # ── Config (override via .env or environment variables on EC2) ──────────
    HOST  = os.getenv("HOST", "0.0.0.0")           # 0.0.0.0 = accept all IPs
    PORT  = int(os.getenv("PORT", "8000"))
    WORKERS = int(os.getenv("WORKERS", "1"))        # increase on EC2 if needed
    ENV   = os.getenv("APP_ENV", "development")

    # ── Auto-kill stale process on same port (Windows + Linux) ──────────────
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        if _s.connect_ex(("127.0.0.1", PORT)) == 0:
            logger.warning(f"[STARTUP] Port {PORT} in use — killing stale process...")
            try:
                # Windows
                if sys.platform == "win32":
                    out = subprocess.check_output(
                        f"netstat -ano | findstr :{PORT}", shell=True
                    ).decode()
                    for line in out.strip().splitlines():
                        parts = line.split()
                        if len(parts) >= 5 and f":{PORT}" in parts[1]:
                            pid = int(parts[-1])
                            if pid != os.getpid():
                                subprocess.call(f"taskkill /PID {pid} /F", shell=True)
                                logger.info(f"[STARTUP] Killed PID {pid}")
                                break
                # Linux / EC2
                else:
                    out = subprocess.check_output(
                        f"lsof -ti :{PORT}", shell=True
                    ).decode().strip()
                    for pid_str in out.splitlines():
                        pid = int(pid_str)
                        if pid != os.getpid():
                            subprocess.call(f"kill -9 {pid}", shell=True)
                            logger.info(f"[STARTUP] Killed PID {pid}")
            except Exception as _e:
                logger.warning(f"[STARTUP] Could not auto-kill: {_e} — free port {PORT} manually")

    # ── Banner ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESEARCH & PODCAST ORCHESTRATOR")
    print("=" * 65)
    print(f"  Environment : {ENV}")
    print(f"  Host        : {HOST}")
    print(f"  Port        : {PORT}")
    print(f"  Workers     : {WORKERS}")
    print(f"  API Docs    : http://{HOST}:{PORT}/docs")
    print(f"  Health      : http://{HOST}:{PORT}/health")
    print(f"  CORS Origins: {ALLOWED_ORIGINS}")
    print("=" * 65 + "\n")

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        workers=WORKERS,
        log_level="info" if ENV == "production" else "debug",
        access_log=True,
    )
