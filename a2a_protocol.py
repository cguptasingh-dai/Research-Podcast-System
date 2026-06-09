"""
A2A Protocol (Agent-to-Agent Communication)
Enables agents to communicate and pass tasks to each other.
Based on Google's A2A protocol concepts, simplified for this use case.
"""

import json
import uuid
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentCard:
    """Represents an agent in the system."""
    agent_id: str
    name: str
    description: str
    version: str
    capabilities: List[str]  # ["research", "podcast", "orchestrate"]
    api_endpoint: str
    created_at: str

    def to_dict(self):
        return asdict(self)


@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    task_id: str
    source_agent: str  # Who created this task
    target_agent: str  # Who should execute this task
    task_type: str  # "research", "podcast", "pipeline"
    status: TaskStatus
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self):
        data = asdict(self)
        data["status"] = self.status.value
        return data


class A2AProtocol:
    """
    A2A Protocol implementation for agent communication.
    Manages task creation, passing, and result retrieval.
    """

    def __init__(self):
        self.agents: Dict[str, AgentCard] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_history: List[str] = []

    def register_agent(self, agent_card: AgentCard) -> None:
        """Register an agent in the system."""
        self.agents[agent_card.agent_id] = agent_card
        print(f"[A2A] Registered agent: {agent_card.name} ({agent_card.agent_id})")

    def get_agent_card(self, agent_id: str) -> Optional[AgentCard]:
        """Get agent card by ID."""
        return self.agents.get(agent_id)

    def list_agents(self) -> List[AgentCard]:
        """List all registered agents."""
        return list(self.agents.values())

    def create_task(
        self,
        source_agent: str,
        target_agent: str,
        task_type: str,
        payload: Dict[str, Any]
    ) -> Task:
        """Create a new task for inter-agent communication."""
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            task_id=task_id,
            source_agent=source_agent,
            target_agent=target_agent,
            task_type=task_type,
            status=TaskStatus.PENDING,
            payload=payload
        )
        self.tasks[task_id] = task
        self.task_history.append(task_id)
        print(f"[A2A] Task created: {task_id} ({task_type}) - {source_agent} -> {target_agent}")
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: TaskStatus) -> Task:
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].updated_at = datetime.utcnow().isoformat()
            print(f"[A2A] Task {task_id} status: {status.value}")
            return self.tasks[task_id]
        raise ValueError(f"Task {task_id} not found")

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> Task:
        """Mark task as completed with result."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        self.tasks[task_id].result = result
        self.tasks[task_id].status = TaskStatus.COMPLETED
        self.tasks[task_id].updated_at = datetime.utcnow().isoformat()
        print(f"[A2A] Task {task_id} completed")
        return self.tasks[task_id]

    def fail_task(self, task_id: str, error: str) -> Task:
        """Mark task as failed with error message."""
        task = self.update_task_status(task_id, TaskStatus.FAILED)
        self.tasks[task_id].error = error
        self.tasks[task_id].updated_at = datetime.utcnow().isoformat()
        print(f"[A2A] Task {task_id} failed: {error}")
        return self.tasks[task_id]

    def get_tasks_for_agent(self, agent_id: str) -> List[Task]:
        """Get all pending tasks for a specific agent."""
        return [
            task for task in self.tasks.values()
            if task.target_agent == agent_id and task.status == TaskStatus.PENDING
        ]

    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get full task execution history."""
        return [self.tasks[task_id].to_dict() for task_id in self.task_history]


# Global A2A Protocol instance
a2a = A2AProtocol()
