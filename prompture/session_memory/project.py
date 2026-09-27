"""Project memory: what coding agents learned about a project, shared between them.

:class:`ProjectMemory` keeps a project's decisions, conventions, useful
commands and verified fixes as :class:`~.types.MemoryFact` records in a
:class:`~.stores.SessionMemoryStore`, one owner per project
(``project:<name>``), so the facts one agent verified reach the next agent
that works there instead of being rediscovered. Each fact can carry where it
came from (``source``: a file and line, a commit, a session), whether it was
verified, and which agent (or person) wrote it.

:meth:`ProjectMemory.select` picks what a new task gets: only facts relevant
to its prompt (plus ones pinned for every task), verified ones by default,
best first, within a token budget. :meth:`ProjectMemory.render` turns them
into the block an agent receives.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .stores import SessionMemoryStore, SQLiteSessionStore, _tokens
from .types import MemoryFact, MemoryKind

DEFAULT_DB_PATH = Path.home() / ".prompture" / "memory" / "projects.db"
PREFIX = "project:"
#: Kinds a project fact can have, in the order they're shown.
PROJECT_KINDS = (
    MemoryKind.decision.value,
    MemoryKind.convention.value,
    MemoryKind.command.value,
    MemoryKind.fix.value,
    MemoryKind.fact.value,
)
#: Default ceiling for the memory a task receives.
DEFAULT_BUDGET_TOKENS = 600

_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "it",
    "this",
    "that",
    "be",
    "are",
    "was",
    "at",
    "by",
    "as",
    "from",
    "we",
    "you",
    "i",
    "me",
    "my",
    "our",
    "please",
    "can",
    "do",
    "does",
    "how",
    "what",
    "why",
    "when",
    "should",
    "would",
    "could",
    "not",
    "no",
    "so",
    "if",
}


def estimate_tokens(text: str) -> int:
    """About four characters per token, the usual rule of thumb for English and code."""
    return max(1, (len(text) + 3) // 4)


def _terms(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOP and len(t) > 1}


def project_name(cwd: str | None) -> str | None:
    """A project's name from its folder: the last part of the working directory."""
    if not cwd or not cwd.strip():
        return None
    name = re.split(r"[\\/]", cwd.strip().rstrip("\\/"))[-1]
    return name or None


class ProjectMemory:
    """Facts about projects, one owner per project. See the module docstring."""

    def __init__(self, store: SessionMemoryStore | None = None, *, db_path: str | Path | None = None) -> None:
        self.store: SessionMemoryStore = store if store is not None else SQLiteSessionStore(db_path or DEFAULT_DB_PATH)

    @staticmethod
    def owner(project: str) -> str:
        return f"{PREFIX}{project}"

    # -- write ----------------------------------------------------------------

    def add(
        self,
        project: str,
        content: str,
        *,
        kind: str = MemoryKind.fact.value,
        source: str | None = None,
        verified: bool = False,
        agent: str | None = None,
        pinned: bool = False,
        importance: float = 0.5,
    ) -> MemoryFact:
        content = content.strip()
        if not project or not content:
            raise ValueError("A project fact needs a project and some text.")
        fact = MemoryFact(
            user_id=self.owner(project),
            content=content,
            kind=kind if kind in PROJECT_KINDS else MemoryKind.fact.value,
            importance=float(importance),
            metadata={
                "source": (source or "").strip() or None,
                "verified": bool(verified),
                "agent": agent,
                "pinned": bool(pinned),
                "updated": time.time(),
            },
        )
        self.store.add(fact)
        return fact

    def get(self, project: str, fact_id: str) -> MemoryFact | None:
        return next((f for f in self.notes(project) if f.id == fact_id), None)

    def update(self, project: str, fact_id: str, **changes: Any) -> MemoryFact | None:
        """Change a fact's ``content``, ``kind``, ``source``, ``verified``, ``pinned`` or ``importance``."""
        fact = self.get(project, fact_id)
        if fact is None:
            return None
        if isinstance(changes.get("content"), str) and changes["content"].strip():
            fact.content = changes["content"].strip()
        if changes.get("kind") in PROJECT_KINDS:
            fact.kind = changes["kind"]
        if isinstance(changes.get("importance"), (int, float)):
            fact.importance = max(0.0, min(1.0, float(changes["importance"])))
        meta = dict(fact.metadata)
        if "source" in changes:
            meta["source"] = (str(changes["source"] or "")).strip() or None
        for flag in ("verified", "pinned"):
            if isinstance(changes.get(flag), bool):
                meta[flag] = changes[flag]
        meta["updated"] = time.time()
        fact.metadata = meta
        self.store.add(fact)  # upsert by id
        return fact

    def delete(self, project: str, fact_id: str) -> bool:
        return self.store.delete(self.owner(project), fact_id=fact_id) > 0

    # -- read -----------------------------------------------------------------

    def notes(self, project: str, *, kinds: tuple[str, ...] | None = None) -> list[MemoryFact]:
        return self.store.list(self.owner(project), kinds=kinds)

    def projects(self) -> list[dict[str, Any]]:
        """Projects with facts: name, count and how many are verified."""
        owners = getattr(self.store, "user_ids", None)
        names = [o[len(PREFIX) :] for o in owners(PREFIX)] if callable(owners) else []
        out = []
        for name in names:
            facts = self.notes(name)
            if facts:
                verified = sum(1 for f in facts if f.metadata.get("verified"))
                out.append({"project": name, "facts": len(facts), "verified": verified})
        return out

    def select(
        self,
        project: str,
        query: str,
        *,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        verified_only: bool = True,
    ) -> list[MemoryFact]:
        """The facts a task about *query* should get, best first, within *budget_tokens*.

        A fact qualifies when it shares a meaningful word with the query (its
        text or its source), or when it's pinned. Verified facts rank above the
        rest; then relevance, importance and recency.
        """
        q = _terms(query)
        now = time.time()
        scored: list[tuple[float, MemoryFact]] = []
        for fact in self.notes(project):
            meta = fact.metadata
            if verified_only and not meta.get("verified"):
                continue
            overlap = len(q & _terms(f"{fact.content} {meta.get('source') or ''}"))
            if overlap == 0 and not meta.get("pinned"):
                continue
            age_days = max(0.0, (now - float(meta.get("updated") or fact.ts)) / 86400.0)
            score = (
                overlap
                + (3.0 if meta.get("pinned") else 0.0)
                + (1.0 if meta.get("verified") else 0.0)
                + fact.importance * 0.5
                + 0.5 ** (age_days / 30.0) * 0.25
            )
            scored.append((score, fact))
        scored.sort(key=lambda t: t[0], reverse=True)
        chosen: list[MemoryFact] = []
        used = 0
        for _, fact in scored:
            cost = estimate_tokens(self.line(fact))
            if used + cost > budget_tokens:
                continue
            chosen.append(fact)
            used += cost
        return chosen

    @staticmethod
    def line(fact: MemoryFact) -> str:
        source = fact.metadata.get("source")
        return f"- [{fact.kind}] {fact.content}" + (f" (source: {source})" if source else "")

    def render(self, project: str, facts: list[MemoryFact], *, save_hint: str | None = None) -> str:
        """The block a task receives: its project's facts, and how to add one.

        With no facts and a ``save_hint``, only the hint: that's how the first
        agent learns it can leave something for the next.
        """
        if not facts and not save_hint:
            return ""
        lines = [f'<project-memory project="{project}">']
        if facts:
            lines += [
                "Notes about this project from earlier sessions of your coding agents. "
                "Prefer them to rediscovering, but the code wins if it disagrees.",
                *(self.line(f) for f in facts),
            ]
        if save_hint:
            lines.append(save_hint)
        lines.append("</project-memory>")
        return "\n".join(lines)
