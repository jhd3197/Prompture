"""High-level :class:`SessionMemory` API.

Sits in front of a :class:`SessionMemoryStore` and provides the
ergonomics callers actually want:

- ``remember(text)`` — write a single fact.
- ``recall(query)`` — fetch the most relevant facts as a list, or as a
  ready-to-paste prompt fragment via :meth:`SessionMemory.render_for_prompt`.
- ``summarize(messages)`` — collapse a chunk of conversation into a
  rolling summary (uses a user-supplied :class:`Summarizer` callable).

Hooks into :class:`~prompture.agents.Conversation` /
:class:`~prompture.agents.Agent` via the ``before_turn`` mechanism: the
caller builds a system-prompt fragment from :meth:`render_for_prompt`
and concatenates it onto the agent's system prompt.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

from .stores import InMemorySessionStore, SessionMemoryStore
from .types import MemoryFact, MemoryKind


@runtime_checkable
class Summarizer(Protocol):
    """Callable that compresses a list of messages into a summary string.

    Used by :meth:`SessionMemory.summarize` to roll up long
    conversations.  The contract is intentionally simple: receive
    role/content message dicts, return a single short paragraph.
    """

    def __call__(self, messages: list[dict[str, Any]]) -> str: ...


def default_summarizer(messages: list[dict[str, Any]]) -> str:
    """Pure-Python fallback summarizer used when none is configured.

    Picks the first and last user messages and joins them.  This is a
    *placeholder* — for real summarisation pass an LLM-backed summarizer
    (e.g. wrap a :class:`~prompture.agents.Conversation`).
    """
    if not messages:
        return ""
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return ""
    first = user_msgs[0].get("content", "")
    last = user_msgs[-1].get("content", "")
    if first == last:
        return f"User said: {first}"
    return f"Conversation started with: {first} … and recently: {last}"


class SessionMemory:
    """User- and session-scoped persistent memory.

    Args:
        user_id: Owning user identifier.  Required.
        session_id: Optional session identifier.  ``None`` means
            "user-wide" — these facts persist across every session.
        store: Backing :class:`SessionMemoryStore`.  Defaults to a fresh
            :class:`InMemorySessionStore`.
        recall_k: Default number of facts surfaced by
            :meth:`render_for_prompt`.
        summarizer: Optional callable used by :meth:`summarize`.  When
            ``None``, :func:`default_summarizer` is used.

    Example::

        from prompture import Conversation, SessionMemory

        memory = SessionMemory(user_id="alice", session_id="trip-2026")
        memory.remember("User prefers concise answers.", kind="preference")

        conv = Conversation(model_name="openai/gpt-4o-mini",
                            system_prompt=memory.render_for_prompt())
        conv.ask("What's the weather?")
    """

    def __init__(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        store: SessionMemoryStore | None = None,
        recall_k: int = 5,
        summarizer: Summarizer | None = None,
    ) -> None:
        if not user_id:
            raise ValueError("SessionMemory requires a non-empty user_id")
        self._user_id = user_id
        self._session_id = session_id
        self._store: SessionMemoryStore = store if store is not None else InMemorySessionStore()
        self._recall_k = recall_k
        self._summarizer = summarizer

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def store(self) -> SessionMemoryStore:
        return self._store

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        kind: str = MemoryKind.fact.value,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        session_scoped: bool = False,
    ) -> MemoryFact:
        """Persist a memory and return the typed :class:`MemoryFact`.

        Args:
            content: Free-form text to remember.
            kind: Classification (see :class:`MemoryKind`).  Custom
                strings are allowed.
            importance: ``[0.0, 1.0]`` — higher values rank higher in
                :meth:`recall`.
            metadata: Optional structured payload.
            session_scoped: When ``True`` the fact is recorded against
                this :class:`SessionMemory`'s ``session_id``; when
                ``False`` (default) the fact is user-wide (no session
                key).  User-wide facts surface across every session.
        """
        sid: str | None = self._session_id if session_scoped else None
        fact = MemoryFact(
            user_id=self._user_id,
            session_id=sid,
            content=content,
            kind=kind,
            importance=float(importance),
            metadata=dict(metadata) if metadata else {},
        )
        self._store.add(fact)
        return fact

    def forget(self, fact_id: str) -> bool:
        """Drop a specific fact by id.  Returns ``True`` when removed."""
        removed = self._store.delete(self._user_id, fact_id=fact_id)
        return removed > 0

    def clear_session(self) -> int:
        """Drop every fact tied to this session.  User-wide facts remain.

        Returns the number of removed records.
        """
        if self._session_id is None:
            # Caller asked to clear "this session" but it has no id;
            # interpret as the user-wide bucket.
            return self._store.delete(self._user_id, session_id="__user__")
        return self._store.delete(self._user_id, session_id=self._session_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def all(
        self,
        *,
        kinds: tuple[str, ...] | None = None,
        include_user_wide: bool = True,
        limit: int | None = None,
    ) -> list[MemoryFact]:
        """List facts visible to this session.

        Returns the session's own facts plus user-wide facts when
        ``include_user_wide`` is ``True`` (the default).
        """
        out: list[MemoryFact] = []
        if include_user_wide:
            out.extend(self._store.list(self._user_id, session_id="__user__", kinds=kinds))
        if self._session_id is not None:
            out.extend(self._store.list(self._user_id, session_id=self._session_id, kinds=kinds))
        out.sort(key=lambda f: f.ts)
        if limit is not None and limit >= 0:
            out = out[-limit:]
        return out

    def recall(
        self,
        query: str,
        *,
        k: int | None = None,
        kinds: tuple[str, ...] | None = None,
        include_user_wide: bool = True,
    ) -> list[MemoryFact]:
        """Return up to *k* facts most relevant to *query*.

        Searches both this session's facts and user-wide facts when
        ``include_user_wide`` is ``True``, merges the candidate lists,
        and re-scores them so the top-*k* spans both buckets.
        """
        effective_k = k if k is not None else self._recall_k
        # The store scorer combines overlap + importance + recency.
        # Querying both buckets and merging by score is fine because the
        # ranking is over (overlap, importance, recency), not bucket.
        merged: list[MemoryFact] = []
        if include_user_wide:
            merged.extend(
                self._store.search(
                    self._user_id,
                    query,
                    session_id="__user__",
                    kinds=kinds,
                    k=effective_k,
                )
            )
        if self._session_id is not None:
            merged.extend(
                self._store.search(
                    self._user_id,
                    query,
                    session_id=self._session_id,
                    kinds=kinds,
                    k=effective_k,
                )
            )
        # De-duplicate by id while preserving order, then truncate.
        seen: set[str] = set()
        out: list[MemoryFact] = []
        for f in merged:
            if f.id in seen:
                continue
            seen.add(f.id)
            out.append(f)
            if effective_k >= 0 and len(out) >= effective_k:
                break
        return out

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def render_for_prompt(
        self,
        query: str | None = None,
        *,
        k: int | None = None,
        header: str = "## Long-term memory",
        include_user_wide: bool = True,
    ) -> str:
        """Render relevant memories as a prompt-ready string.

        When *query* is ``None`` the most recent ``k`` facts are
        included; otherwise :meth:`recall` is used to score by
        relevance.  Returns an empty string when no facts apply.
        """
        if query is None:
            facts = self.all(include_user_wide=include_user_wide, limit=k or self._recall_k)
        else:
            facts = self.recall(query, k=k, include_user_wide=include_user_wide)
        if not facts:
            return ""

        lines = [header]
        for f in facts:
            lines.append(f"- ({f.kind}) {f.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarize(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        importance: float = 0.6,
        summarizer: Summarizer | Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> MemoryFact | None:
        """Collapse a slice of conversation into a stored summary memory.

        Args:
            messages: Iterable of role/content message dicts (typically
                :attr:`Conversation.messages`).
            importance: Importance score for the resulting fact.
            summarizer: Override the configured summarizer for this
                call only.  Useful when wiring up an LLM-backed
                summarizer dynamically.

        Returns:
            The persisted :class:`MemoryFact`, or ``None`` when the
            summarizer produced an empty string.
        """
        msg_list = list(messages)
        if not msg_list:
            return None
        fn = summarizer or self._summarizer or default_summarizer
        summary_text = fn(msg_list)
        if not summary_text:
            return None
        return self.remember(
            summary_text,
            kind=MemoryKind.summary.value,
            importance=importance,
            metadata={"message_count": len(msg_list)},
            session_scoped=True,
        )
