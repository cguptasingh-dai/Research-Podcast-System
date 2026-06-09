"""
FastAPI - Research Agent + Podcast Agent via A2A Protocol
Two modes: Research Only | Research + Podcast (Pipeline)
"""

import asyncio
import sys
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

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

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

app = FastAPI(title="Research & Podcast Agent API", version="1.0.0", docs_url="/docs")
startup_time = datetime.utcnow()


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
            # Import must happen inside thread fn because graph.py uses relative imports
            if podcast_dir not in sys.path:
                sys.path.insert(0, podcast_dir)
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
    return {
        "status": "ok",
        "agents": agent_count,
        "agents_registered": agent_count,
        "tasks": len(a2a.tasks),
        "uptime_seconds": (datetime.utcnow() - startup_time).total_seconds(),
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
    import uvicorn, socket, subprocess, os as _os
    _PORT = 8000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        if _s.connect_ex(("127.0.0.1", _PORT)) == 0:
            print(f"[STARTUP] Port {_PORT} in use — killing stale process...")
            try:
                out = subprocess.check_output(
                    f"netstat -ano | findstr :{_PORT}", shell=True
                ).decode()
                for line in out.strip().splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and f":{_PORT}" in parts[1]:
                        pid = int(parts[-1])
                        if pid != _os.getpid():
                            subprocess.call(f"taskkill /PID {pid} /F", shell=True)
                            print(f"[STARTUP] Killed PID {pid}")
                            break
            except Exception as _e:
                print(f"[STARTUP] Could not auto-kill: {_e} — free port {_PORT} manually")
    uvicorn.run(app, host="0.0.0.0", port=8000)
