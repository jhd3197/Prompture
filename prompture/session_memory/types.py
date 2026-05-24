"""Types for cross-session memory."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class MemoryKind(str, enum.Enum):
    """Categorisation tag used by :class:`MemoryFact`.

    Values are stored as plain strings so callers can introduce custom
    kinds without modifying the enum (memory backends compare kinds by
    string equality).
    """

    fact = "fact"
    """An atomic stable fact about the user or world."""

    preference = "preference"
    """A user-stated preference (likes/dislikes/conventions)."""

    summary = "summary"
    """A rolling summary of a chunk of conversation."""

    note = "note"
    """A free-form note the caller asked to persist."""

    instruction = "instruction"
    """A persistent system-instruction the user asked Claude to follow."""


@dataclass
class MemoryFact:
    """A single record persisted by :class:`SessionMemory`.

    Attributes:
        id: Unique identifier (auto-generated UUID4 hex if omitted).
        user_id: Owner of the memory.  Required.
        session_id: Optional sub-key for session-scoped memories.  Use
            ``None`` for user-wide memories.
        content: The remembered text.
        kind: Classification (see :class:`MemoryKind`).
        ts: Wall-clock timestamp of authorship.
        importance: Float on ``[0.0, 1.0]``.  Higher values are more
            likely to surface in :meth:`SessionMemory.recall`.
        metadata: Free-form structured payload.
    """

    user_id: str
    content: str
    kind: str = MemoryKind.fact.value
    session_id: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "content": self.content,
            "kind": self.kind,
            "ts": self.ts,
            "importance": self.importance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryFact:
        """Inverse of :meth:`to_dict`."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            session_id=data.get("session_id"),
            content=data["content"],
            kind=data.get("kind", MemoryKind.fact.value),
            ts=float(data.get("ts", time.time())),
            importance=float(data.get("importance", 0.5)),
            metadata=dict(data.get("metadata") or {}),
        )
