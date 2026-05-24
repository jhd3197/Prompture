"""Resumable workflows for Prompture.

Take a snapshot of an in-flight :class:`~prompture.agents.Conversation`
or :class:`~prompture.agents.Agent` so it can resume after a process
restart, crash, or human-in-the-loop pause.

Three layers:

- :class:`Checkpoint` — JSON-serialisable snapshot of a run.
- :class:`CheckpointStore` — pluggable backend protocol.  Ships with
  :class:`InMemoryCheckpointStore`, :class:`FileCheckpointStore`, and
  :class:`SQLiteCheckpointStore`.
- :class:`CheckpointManager` — high-level API tying a ``run_id`` to a
  store, with ``save`` / ``load`` / ``resume`` helpers.

Designed to be non-invasive: no modifications to :class:`Agent` or
:class:`Conversation` are required.  Hook
:meth:`CheckpointManager.save_conversation` into
:attr:`Conversation.before_turn` or
:attr:`AgentCallbacks.on_step` to persist progress automatically.
"""

from .core import (
    CheckpointManager,
    restore_conversation,
    snapshot_conversation,
)
from .stores import (
    CheckpointStore,
    FileCheckpointStore,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from .types import Checkpoint, RunStatus

__all__ = [
    "Checkpoint",
    "CheckpointManager",
    "CheckpointStore",
    "FileCheckpointStore",
    "InMemoryCheckpointStore",
    "RunStatus",
    "SQLiteCheckpointStore",
    "restore_conversation",
    "snapshot_conversation",
]
