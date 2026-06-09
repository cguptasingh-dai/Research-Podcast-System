"""
FastAPI Server for Researcher Agent
Proper integration with A2A Protocol
Port: 8001
"""

import sys
import asyncio

# Windows: force UTF-8 stdout/stderr so CrewAI's event-bus log handlers don't
# crash with 'charmap codec can't encode' on tool output containing Unicode.
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

# Add paths for imports
root_path = str(Path(__file__).parent.parent)  # A2a root
src_path = str(Path(__file__).parent / "src")   # src folder

sys.path.insert(0, root_path)
sys.path.insert(0, src_path)

from a2a_protocol import a2a, AgentCard, TaskStatus as A2ATaskStatus
from researcher.report_refinement import run_report_refinement
from config_shared import LLMConfig

# ============================================================================
# MODELS
# ============================================================================

class ResearchTaskPayload(BaseModel):
    """Payload for research task"""
    topic: str
    iterations: int = 1  # 1 iteration = faster (was 2)
    research_findings: Optional[str] = None


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
    title="Researcher Agent API",
    description="CrewAI-based Research Agent with A2A Protocol",
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
    """Register researcher agent on startup"""
    researcher_card = AgentCard(
        agent_id="researcher-001",
        name="Researcher Agent",
        description="CrewAI-based research agent for comprehensive web research",
        version="1.0.0",
        capabilities=["research", "report-generation", "web-search"],
        api_endpoint="http://localhost:8001",
        created_at=datetime.utcnow().isoformat()
    )

    a2a.register_agent(researcher_card)
    print("[OK] Researcher Agent Registered")
    print(f"   Endpoint: http://localhost:8001")
    print(f"   Capabilities: {researcher_card.capabilities}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def execute_research_task(task_id: str, topic: str, iterations: int):
    """Execute research task in background"""
    try:
        # Update task status
        a2a.update_task_status(task_id, A2ATaskStatus.IN_PROGRESS)
        print(f"[RESEARCHER] Starting research: {topic} ({iterations} iterations)")

        # Execute research — run_report_refinement is blocking (CrewAI),
        # must run in a thread to avoid freezing the FastAPI event loop
        def _run_research():
            return run_report_refinement(
                research_findings=None,
                topic=topic,
                num_iterations=iterations
            )

        summary = await asyncio.to_thread(_run_research)

        # Prepare result - keys match what orchestrator expects
        result = {
            "topic": summary.get("topic", topic),
            "score": summary.get("best_score", 0),
            "iteration": summary.get("best_iteration", 1),
            "report_content": summary.get("best_report", ""),  # KEY for podcast agent
            "file": f"Report/{topic.replace(' ','_').lower()}.pdf",
        }

        # Complete task
        a2a.complete_task(task_id, result)
        print(f"[RESEARCHER] OK Research complete: {topic} (Score: {result['score']}/100)")

    except Exception as e:
        error_msg = f"Research failed: {str(e)}"
        print(f"[RESEARCHER] ERROR {error_msg}")
        a2a.fail_task(task_id, error_msg)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "researcher-001",
        "uptime_seconds": (datetime.utcnow() - startup_time).total_seconds(),
        "registered_agents": len(a2a.list_agents())
    }


@app.get("/info")
async def info():
    """Agent information endpoint"""
    agents = a2a.list_agents()
    researcher = next((a for a in agents if a.agent_id == "researcher-001"), None)

    return {
        "agent_id": "researcher-001",
        "name": "Researcher Agent",
        "description": "CrewAI-based research agent",
        "version": "1.0.0",
        "capabilities": ["research", "report-generation", "web-search"],
        "endpoint": "http://localhost:8001",
        "framework": "CrewAI",
        "llm": LLMConfig.NVIDIA_MODEL,
        "temperature": LLMConfig.TEMPERATURE,
        "max_tokens": LLMConfig.MAX_TOKENS,
        "status": "active"
    }


@app.post("/research")
async def research(payload: ResearchTaskPayload, bg_tasks: BackgroundTasks):
    """
    Execute research task

    Args:
        payload: Research task payload with topic and iterations
        bg_tasks: Background tasks queue

    Returns:
        Task response with task_id
    """
    # Create A2A task
    task = a2a.create_task(
        source_agent="external",
        target_agent="researcher-001",
        task_type="research",
        payload={
            "topic": payload.topic,
            "iterations": payload.iterations,
            "research_findings": payload.research_findings
        }
    )

    # Queue background task
    bg_tasks.add_task(
        execute_research_task,
        task.task_id,
        payload.topic,
        payload.iterations
    )

    print(f"[API] Research task created: {task.task_id}")

    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "message": f"Research on '{payload.topic}' queued"
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
    # Free port 8001 if a stale process is still holding it
    import socket, subprocess, os as _os
    _PORT = 8001
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

    print("\n" + "="*70)
    print("RESEARCHER AGENT - FastAPI Server")
    print("="*70)
    print(f"Starting on http://0.0.0.0:8001")
    print(f"Docs: http://localhost:8001/docs")
    print("="*70 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
