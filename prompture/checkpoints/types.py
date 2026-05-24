"""Types for checkpointed / resumable runs."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class RunStatus(str, enum.Enum):
    """Lifecycle states for a checkpointed run.

    Values are stored as plain strings so backends can persist them
    portably.  Custom states (e.g. ``"awaiting_approval"``) are
    permitted — the type is only used for display and filtering.
    """

    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


@dataclass
class Checkpoint:
    """A JSON-serialisable snapshot of an in-flight run.

    Attributes:
        run_id: Stable identifier for the run.  A single run may
            produce many checkpoints over its lifetime; each save
            overwrites the previous record for the same ``run_id``
            unless the backend supports versioned history.
        id: Per-snapshot id (auto-generated UUID4 hex).
        agent_name: Optional name of the producing agent.
        model: Model string in ``"provider/model"`` form.
        messages: Conversation message history.
        usage: Aggregate usage summary (typically the output of
            :meth:`UsageSession.summary`).
        iteration: Current iteration/tool-round/tick index.
        status: Lifecycle state.  See :class:`RunStatus`.
        prompt: The original user prompt that started the run.
        partial_output: Any partially-generated output worth persisting
            (a streaming text buffer, an incomplete JSON parse, etc.).
        system_prompt: Captured system prompt so the restored
            conversation has the same priming.
        ts: Wall-clock timestamp of the snapshot.
        version: Snapshot schema version.  Bumped if the on-disk shape
            changes incompatibly.
        metadata: Free-form caller-supplied payload (todos, sub-agent
            specs, retrieval context, etc.).
    """

    run_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    status: str = RunStatus.running.value
    prompt: str = ""
    partial_output: str = ""
    system_prompt: str | None = None
    agent_name: str | None = None
    model: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "version": self.version,
            "id": self.id,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "usage": self.usage,
            "iteration": self.iteration,
            "status": self.status,
            "prompt": self.prompt,
            "partial_output": self.partial_output,
            "ts": self.ts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Inverse of :meth:`to_dict`.  Unknown keys are ignored."""
        return cls(
            run_id=data["run_id"],
            id=data.get("id", uuid.uuid4().hex),
            agent_name=data.get("agent_name"),
            model=data.get("model"),
            system_prompt=data.get("system_prompt"),
            messages=list(data.get("messages") or []),
            usage=dict(data.get("usage") or {}),
            iteration=int(data.get("iteration", 0)),
            status=data.get("status", RunStatus.running.value),
            prompt=data.get("prompt", ""),
            partial_output=data.get("partial_output", ""),
            ts=float(data.get("ts", time.time())),
            version=int(data.get("version", 1)),
            metadata=dict(data.get("metadata") or {}),
        )
