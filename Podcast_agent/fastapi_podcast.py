"""
FastAPI Server for Podcast Agent
Proper integration with A2A Protocol
Port: 8002
"""

import sys
import asyncio

# Windows: force UTF-8 stdout/stderr so log handlers don't crash with
# 'charmap codec can't encode' on Unicode content.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv  # LOAD .ENV FILE

# Load environment variables BEFORE anything else
load_dotenv()

# Add paths: root (for a2a_protocol, config_shared) + Podcast_agent/ (for graph_fixed, state, tools, config)
_root_dir   = str(Path(__file__).parent.parent)   # A2a/
_agent_dir  = str(Path(__file__).parent)           # A2a/Podcast_agent/
if _root_dir  not in sys.path: sys.path.insert(0, _root_dir)
if _agent_dir not in sys.path: sys.path.insert(0, _agent_dir)

from a2a_protocol import a2a, AgentCard, TaskStatus
from graph_fixed import Pipeline  # graph_fixed.py lives in Podcast_agent/

# Create Pipeline ONCE at module level (not per-request)
# This avoids re-initializing LLM client and re-compiling graph every request
_pipeline: Pipeline = None

def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        print("[PODCAST] Initializing Pipeline (first request)...")
        _pipeline = Pipeline()
    return _pipeline

# ============================================================================
# MODELS
# ============================================================================

class PodcastTaskPayload(BaseModel):
    """Payload for podcast task"""
    report_content: str
    topic: str


class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Podcast Agent API",
    description="LangGraph-based Podcast Agent with A2A Protocol",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

startup_time = datetime.utcnow()

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup():
    """Register podcast agent on startup"""
    podcast_card = AgentCard(
        agent_id="podcast-001",
        name="Podcast Agent",
        description="LangGraph-based podcast agent for Q&A extraction and audio synthesis",
        version="1.0.0",
        capabilities=["podcast", "audio-synthesis", "qa-extraction", "tts"],
        api_endpoint="http://localhost:8002",
        created_at=datetime.utcnow().isoformat()
    )

    a2a.register_agent(podcast_card)
    print("[OK] Podcast Agent Registered")
    print(f"   Endpoint: http://localhost:8002")
    print(f"   Capabilities: {podcast_card.capabilities}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def execute_podcast_task(task_id: str, report_content: str, topic: str):
    """Execute podcast task in background"""
    try:
        # Update task status
        a2a.update_task_status(task_id, TaskStatus.IN_PROGRESS)
        print(f"[PODCAST] Starting podcast: {topic}")

        # Prepare state for LangGraph pipeline
        state = {
            "report_content": report_content,
            "topic": topic,
            "questions": [],
            "answers": [],
            "qa_pairs": [],
            "conversation_text": "",
            "agent_logs": [],
            "audio_file_path": "",
            "transcript": ""
        }

        # CRITICAL FIX: Run blocking pipeline in thread pool
        # Pipeline uses LLM + Sarvam TTS (blocking I/O) - must NOT run in async context
        def _run_pipeline():
            return get_pipeline().run(state)

        final_state = await asyncio.to_thread(_run_pipeline)

        questions = final_state.get("questions", [])
        answers = final_state.get("answers", [])
        audio_path = final_state.get("audio_file_path") or ""

        # Build a readable transcript from the Q&A (pipeline doesn't store one).
        # This is what the UI displays alongside the audio.
        transcript = final_state.get("transcript", "")
        if not transcript and questions:
            lines = [f"# {final_state.get('topic', topic)} — Podcast Transcript", ""]
            for idx, (q, a) in enumerate(zip(questions, answers), 1):
                lines.append(f"Host: {q}")
                lines.append(f"Expert: {a}")
                lines.append("")
            transcript = "\n".join(lines).strip()

        # Prepare result
        result = {
            "topic": final_state.get("topic"),
            "questions": questions,
            "answers": answers,
            "audio": audio_path,
            "audio_files": [audio_path] if audio_path else [],
            "transcript": transcript,
            "qa_count": len(questions),
        }

        # Complete task
        a2a.complete_task(task_id, result)
        print(f"[PODCAST] OK Podcast complete: {topic} ({result['qa_count']} Q&A pairs)")

    except Exception as e:
        error_msg = f"Podcast failed: {str(e)}"
        print(f"[PODCAST] ERROR {error_msg}")
        a2a.fail_task(task_id, error_msg)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "podcast-001",
        "uptime_seconds": (datetime.utcnow() - startup_time).total_seconds(),
        "registered_agents": len(a2a.list_agents())
    }


@app.get("/info")
async def info():
    """Agent information endpoint"""
    agents = a2a.list_agents()
    podcast_agent = next((a for a in agents if a.agent_id == "podcast-001"), None)

    return {
        "agent_id": "podcast-001",
        "name": "Podcast Agent",
        "description": "LangGraph-based podcast agent",
        "version": "1.0.0",
        "capabilities": ["podcast", "audio-synthesis", "qa-extraction", "tts"],
        "endpoint": "http://localhost:8002",
        "framework": "LangGraph",
        "llm": "NVIDIA Llama-3.3 Nemotron Super 49B",
        "tts": "Sarvam AI (bulbul:v3)",
        "voices": ["shubh (male host - questions)", "kavya/meera (female expert - answers)"],
        "temperature": 0.4,
        "max_tokens": 4000,
        "status": "active"
    }


@app.post("/podcast")
async def podcast(payload: PodcastTaskPayload, bg_tasks: BackgroundTasks):
    """
    Execute podcast task

    Args:
        payload: Podcast task payload with report content and topic
        bg_tasks: Background tasks queue

    Returns:
        Task response with task_id
    """
    # Create A2A task
    task = a2a.create_task(
        source_agent="external",
        target_agent="podcast-001",
        task_type="podcast",
        payload={
            "report_content": payload.report_content,
            "topic": payload.topic
        }
    )

    # Queue background task
    bg_tasks.add_task(
        execute_podcast_task,
        task.task_id,
        payload.report_content,
        payload.topic
    )

    print(f"[API] Podcast task created: {task.task_id}")

    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "message": f"Podcast for '{payload.topic}' queued"
    }


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task status and result"""
    task = a2a.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "result": task.result,
        "error": task.error,
        "created_at": task.created_at,
        "updated_at": task.updated_at
    }


@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    tasks = list(a2a.tasks.values())

    return {
        "total_tasks": len(tasks),
        "tasks": [
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "type": t.task_type,
                "created_at": t.created_at
            }
            for t in tasks
        ]
    }


@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    agents = a2a.list_agents()

    return {
        "total_agents": len(agents),
        "agents": [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "capabilities": a.capabilities,
                "api_endpoint": a.api_endpoint
            }
            for a in agents
        ]
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Free port 8002 if a stale process is still holding it
    import socket, subprocess, os as _os
    _PORT = 8002
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        if _s.connect_ex(("127.0.0.1", _PORT)) == 0:
            print(f"[STARTUP] Port {_PORT} in use — killing stale process...")
            try:
                import sys as _sys
                if _sys.platform == "win32":
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
                else:  # Linux / EC2
                    out = subprocess.check_output(
                        f"lsof -ti :{_PORT}", shell=True
                    ).decode().strip()
                    for pid_str in out.splitlines():
                        pid = int(pid_str)
                        if pid != _os.getpid():
                            subprocess.call(f"kill -9 {pid}", shell=True)
                            print(f"[STARTUP] Killed PID {pid}")
            except Exception as _e:
                print(f"[STARTUP] Could not auto-kill: {_e} — free port {_PORT} manually")

    print("\n" + "="*70)
    print("PODCAST AGENT - FastAPI Server")
    print("="*70)
    print(f"Starting on http://0.0.0.0:8002")
    print(f"Docs: http://localhost:8002/docs")
    print("="*70 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
