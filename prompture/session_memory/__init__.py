"""Cross-session memory for Prompture.

Distinct from :class:`~prompture.swarm.MemoryStore` — which is a
per-agent, per-run scratchpad — :class:`SessionMemory` persists across
*conversations*, keyed by ``(user_id, session_id)``.  Designed to plug
into :class:`~prompture.agents.Conversation` and
:class:`~prompture.agents.Agent` (especially with
``persistent_conversation=True``) so user-specific facts survive a
process restart.

Three layers:

- :class:`MemoryFact` — an atomic remembered item (a recalled snippet,
  a summarised round, a user preference).
- :class:`SessionMemoryStore` — pluggable backend protocol; ships with
  :class:`InMemorySessionStore` and :class:`SQLiteSessionStore`.
- :class:`SessionMemory` — high-level API: ``remember`` /
  ``recall`` / ``summarize`` / ``render_for_prompt``.
"""

from .core import (
    SessionMemory,
    Summarizer,
    default_summarizer,
)
from .stores import (
    InMemorySessionStore,
    SessionMemoryStore,
    SQLiteSessionStore,
)
from .types import MemoryFact, MemoryKind

__all__ = [
    "InMemorySessionStore",
    "MemoryFact",
    "MemoryKind",
    "SQLiteSessionStore",
    "SessionMemory",
    "SessionMemoryStore",
    "Summarizer",
    "default_summarizer",
]
