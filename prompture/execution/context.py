"""Shared context allocation, selective loading, and safe compaction.

A long-running agent's context window is a shared, contended resource: the
system prompt, the task's hard constraints, recent turns, retrieved evidence,
skill instructions, tool schemas, and the model's own response allowance all want
space in it.  Filling it greedily in whatever order things arrive is how a run
loses the one constraint it needed, or truncates a tool result mid-JSON.

This module makes the allocation explicit.

Four pieces
-----------

:class:`ContextPolicy`
    A budget divided into named :class:`ContextSection`\\ s with weights,
    minimums, and a *protected* flag.  Protected sections are never trimmed;
    everything else competes for what is left after the response allowance is
    reserved.
:class:`SkillCatalog`
    Skill *summaries* are cheap and always loaded; full instructions are fetched
    only for the skills a task actually selects.  Eager loading stays available
    and supported for callers who want every skill in the prompt.
:class:`ToolCatalog`
    Search over tool names and one-line descriptions, with full JSON schemas
    loaded on demand.  The caller's allow-list is applied **at construction**,
    so an unauthorised tool is not merely deprioritised — it is not in the
    catalogue to be found, described, or called.
:class:`ArtifactStore`
    Large tool outputs are stored once and referenced by handle.  Excerpts are
    loaded by relevance instead of pasting a 200 KB payload into the window, and
    a handle is readable only from the scope that created it.

Compaction
----------

:func:`compact_messages` drops the *oldest droppable* turns first and guarantees
four things about what survives:

1. Protected content (system prompt, declared constraints) is kept verbatim.
2. Source attributions on kept content are kept with it.
3. Tool calls and their results stay paired — a result is never left orphaned,
   and a call is never left dangling, because a provider will reject either.
4. Unresolved questions are carried forward in the summary rather than silently
   dropped.

Token counting reuses :func:`prompture.infra.budget.estimate_tokens` (tiktoken
when installed, a chars/4 heuristic otherwise), and every report says which
counter produced its numbers — a heuristic count is not a measurement.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("prompture.execution.context")

__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "CompactionResult",
    "ContextAssembly",
    "ContextPolicy",
    "ContextSection",
    "SkillCatalog",
    "SkillSummary",
    "ToolCatalog",
    "ToolSummary",
    "compact_messages",
    "count_tokens",
    "supports_native_tool_discovery",
]


def count_tokens(text: str) -> int:
    """Token estimate for *text*, via tiktoken when available."""
    from ..infra.budget import estimate_tokens

    return estimate_tokens(text)


def _counter_name() -> str:
    try:
        import tiktoken  # noqa: F401

        return "tiktoken"
    except Exception:
        return "heuristic"


_WORD = re.compile(r"[a-z0-9_]+")


def _terms(text: str) -> set[str]:
    return set(_WORD.findall((text or "").casefold()))


def _relevance(query: str, *fields: str) -> float:
    """Fraction of the query's terms that appear in *fields*.

    Deliberately a keyword overlap, not an embedding: it is free, deterministic,
    and explainable.  A caller with an embedding store should pass its own
    ``rank`` function to :meth:`ToolCatalog.search` / :meth:`SkillCatalog.select`.
    """
    wanted = _terms(query)
    if not wanted:
        return 0.0
    haystack = _terms(" ".join(f for f in fields if f))
    return len(wanted & haystack) / len(wanted)


# ---------------------------------------------------------------------------
# Context allocation
# ---------------------------------------------------------------------------


@dataclass
class ContextSection:
    """One competing claim on the context window.

    Attributes:
        name: Section identifier (``"instructions"``, ``"evidence"``, …).
        items: Ordered candidate strings.  Earlier items are preferred.
        weight: Share of the discretionary budget this section may claim.
        min_tokens: Floor this section keeps even when the budget is tight.
        protected: When ``True`` the section is never trimmed and its cost is
            deducted from the budget before anything else competes.  Use it for
            the system prompt and for constraints the task must not lose.
        max_tokens: Optional ceiling, regardless of weight.
    """

    name: str
    items: list[str] = field(default_factory=list)
    weight: float = 1.0
    min_tokens: int = 0
    protected: bool = False
    max_tokens: int | None = None

    def token_cost(self) -> int:
        return sum(count_tokens(item) for item in self.items)


@dataclass
class ContextAssembly:
    """The result of allocating a window across sections.

    Attributes:
        sections: ``{name: [kept items]}`` in section order.
        dropped: ``{name: [dropped items]}`` — what did not fit, so a caller can
            store it as an artifact or summarise it rather than lose it.
        allocated: ``{name: token budget}`` the section was granted.
        used: ``{name: tokens actually used}``.
        total_tokens: Sum of ``used``.
        window_tokens / response_allowance: The inputs to the allocation.
        token_counter: ``"tiktoken"`` or ``"heuristic"`` — never present a
            heuristic count as an exact one.
        overflow: ``True`` when protected content alone exceeded the window.
    """

    sections: dict[str, list[str]] = field(default_factory=dict)
    dropped: dict[str, list[str]] = field(default_factory=dict)
    allocated: dict[str, int] = field(default_factory=dict)
    used: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0
    window_tokens: int = 0
    response_allowance: int = 0
    token_counter: str = "heuristic"
    overflow: bool = False

    def render(self, *, separator: str = "\n\n") -> str:
        """Join every kept item, section by section, in policy order."""
        blocks: list[str] = []
        for items in self.sections.values():
            blocks.extend(items)
        return separator.join(b for b in blocks if b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocated": self.allocated,
            "used": self.used,
            "total_tokens": self.total_tokens,
            "window_tokens": self.window_tokens,
            "response_allowance": self.response_allowance,
            "token_counter": self.token_counter,
            "overflow": self.overflow,
            "dropped_counts": {k: len(v) for k, v in self.dropped.items()},
        }


@dataclass
class ContextPolicy:
    """How a context window is divided between competing sections.

    Args:
        window_tokens: Total usable context for the model.
        response_allowance: Tokens reserved for the model's answer.  Taken off
            the top: a policy that budgets the whole window and then wonders why
            the response was truncated has simply not budgeted the response.
        sections: The competing claims, in the order they should be rendered.

    Example::

        policy = ContextPolicy(
            window_tokens=8000,
            response_allowance=1000,
            sections=[
                ContextSection("instructions", protected=True),
                ContextSection("constraints", protected=True),
                ContextSection("evidence", weight=3.0, min_tokens=500),
                ContextSection("history", weight=2.0),
                ContextSection("tools", weight=1.0),
            ],
        )
        assembly = policy.allocate({"instructions": [system_prompt], "evidence": passages})
    """

    window_tokens: int = 8000
    response_allowance: int = 1000
    sections: list[ContextSection] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.window_tokens <= 0:
            raise ValueError(f"window_tokens must be positive (got {self.window_tokens})")
        if self.response_allowance < 0:
            raise ValueError(f"response_allowance must be >= 0 (got {self.response_allowance})")
        if self.response_allowance >= self.window_tokens:
            raise ValueError(
                f"response_allowance ({self.response_allowance}) must leave room in the window ({self.window_tokens})"
            )
        names = [s.name for s in self.sections]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Duplicate context section names: {duplicates}")

    @classmethod
    def default(cls, *, window_tokens: int = 8000, response_allowance: int = 1000) -> ContextPolicy:
        """A sensible starting allocation for an evidence-grounded tool task."""
        return cls(
            window_tokens=window_tokens,
            response_allowance=response_allowance,
            sections=[
                ContextSection("instructions", protected=True),
                ContextSection("constraints", protected=True),
                ContextSection("skills", weight=1.0),
                ContextSection("tools", weight=1.0),
                ContextSection("evidence", weight=3.0, min_tokens=200),
                ContextSection("history", weight=2.0),
            ],
        )

    def section(self, name: str) -> ContextSection:
        for section in self.sections:
            if section.name == name:
                return section
        raise KeyError(name)

    def allocate(self, content: dict[str, Sequence[str]] | None = None) -> ContextAssembly:
        """Fit *content* into the window and report what was kept and dropped.

        Args:
            content: ``{section_name: items}``.  Items for a section not in the
                policy are ignored (and logged), so a typo cannot silently
                consume the budget.
        """
        content = {k: list(v) for k, v in (content or {}).items()}
        unknown = sorted(set(content) - {s.name for s in self.sections})
        if unknown:
            logger.warning("ignoring content for unknown context section(s): %s", unknown)

        assembly = ContextAssembly(
            window_tokens=self.window_tokens,
            response_allowance=self.response_allowance,
            token_counter=_counter_name(),
        )

        budget = self.window_tokens - self.response_allowance

        # 1. Protected sections are taken off the top, whole.
        protected_cost = 0
        for section in self.sections:
            items = content.get(section.name, list(section.items))
            if not section.protected:
                continue
            cost = sum(count_tokens(i) for i in items)
            assembly.sections[section.name] = list(items)
            assembly.allocated[section.name] = cost
            assembly.used[section.name] = cost
            assembly.dropped.setdefault(section.name, [])
            protected_cost += cost

        remaining = budget - protected_cost
        if remaining < 0:
            assembly.overflow = True
            logger.warning(
                "protected context (%d tokens) exceeds the window budget (%d); nothing discretionary will fit",
                protected_cost,
                budget,
            )
            remaining = 0

        # 2. Discretionary sections split what is left, by weight, honouring
        #    minimums first so a small-but-required section is not starved.
        discretionary = [s for s in self.sections if not s.protected]
        total_weight = sum(max(0.0, s.weight) for s in discretionary) or 1.0
        floor_total = sum(s.min_tokens for s in discretionary)
        after_floors = max(0, remaining - floor_total)

        for section in discretionary:
            share = int(after_floors * (max(0.0, section.weight) / total_weight))
            grant = section.min_tokens + share
            if section.max_tokens is not None:
                grant = min(grant, section.max_tokens)
            grant = min(grant, remaining)
            assembly.allocated[section.name] = grant

        # 3. Fill each discretionary section in item order; unused budget from an
        #    under-filled section rolls forward to the next one.
        carry = 0
        for section in discretionary:
            grant = assembly.allocated[section.name] + carry
            items = content.get(section.name, list(section.items))
            kept: list[str] = []
            dropped: list[str] = []
            used = 0
            for item in items:
                cost = count_tokens(item)
                if used + cost <= grant:
                    kept.append(item)
                    used += cost
                else:
                    dropped.append(item)
            carry = max(0, grant - used)
            assembly.sections[section.name] = kept
            assembly.dropped[section.name] = dropped
            assembly.used[section.name] = used

        # Render in policy order.
        assembly.sections = {s.name: assembly.sections.get(s.name, []) for s in self.sections}
        assembly.total_tokens = sum(assembly.used.values())
        return assembly


# ---------------------------------------------------------------------------
# Selective skill loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillSummary:
    """The cheap half of a skill: what it is, without its instructions."""

    name: str
    description: str
    source: str = ""

    def render(self) -> str:
        return f"- {self.name}: {self.description}"


class SkillCatalog:
    """Skill summaries now, full instructions on demand.

    Wraps any collection of :class:`~prompture.agents.skills.SkillInfo` (or the
    global skill registry).  A prompt gets the one-line summaries — a few tokens
    each — and only the skills a task actually selects contribute their full
    instruction bodies.

    Eager loading stays supported and is not deprecated: pass
    ``eager=True`` to :meth:`instructions_for` semantics by simply selecting
    every name, or call :meth:`all_instructions`.

    Example::

        catalog = SkillCatalog.from_registry()
        print(catalog.summary_block())               # cheap, always included
        chosen = catalog.select("cite the source passages", k=2)
        body = catalog.instructions_for([s.name for s in chosen])   # loaded now
    """

    def __init__(self, skills: Iterable[Any] = ()) -> None:
        self._skills: dict[str, Any] = {}
        for skill in skills:
            name = getattr(skill, "name", None)
            if name:
                self._skills[str(name)] = skill
        self._loaded: set[str] = set()

    @classmethod
    def from_registry(cls) -> SkillCatalog:
        """Build from the process-wide skill registry."""
        from ..agents.skills import get_skill_registry_snapshot

        return cls(get_skill_registry_snapshot().values())

    # ---- cheap half ---------------------------------------------------

    @property
    def names(self) -> list[str]:
        return sorted(self._skills)

    def summaries(self) -> list[SkillSummary]:
        return [
            SkillSummary(
                name=name,
                description=str(getattr(skill, "description", "") or ""),
                source=str(getattr(skill, "source", "") or ""),
            )
            for name, skill in sorted(self._skills.items())
        ]

    def summary_block(self, *, header: str = "## Available skills") -> str:
        """The summaries as a single prompt block."""
        summaries = self.summaries()
        if not summaries:
            return ""
        return "\n".join([header, *(s.render() for s in summaries)])

    def select(
        self,
        query: str,
        *,
        k: int = 3,
        min_relevance: float = 0.1,
        rank: Callable[[str, SkillSummary], float] | None = None,
    ) -> list[SkillSummary]:
        """The ``k`` most relevant skill summaries for *query*.

        Args:
            query: The task text.
            k: How many to return.
            min_relevance: Skills below this score are not returned at all —
                loading an irrelevant skill costs tokens and misdirects the
                model, so "none matched" is a valid answer.
            rank: Optional custom scorer, e.g. an embedding similarity.
        """
        scorer = rank or (lambda q, s: _relevance(q, s.name.replace("-", " "), s.description))
        scored = [(scorer(query, s), s) for s in self.summaries()]
        hits = [(score, s) for score, s in scored if score >= min_relevance]
        hits.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [s for _score, s in hits[:k]]

    # ---- expensive half -----------------------------------------------

    def instructions_for(self, names: Iterable[str]) -> str:
        """Full instruction bodies for *names*, loaded now.

        Unknown names are skipped with a warning rather than raising: a model
        that hallucinates a skill name should not crash the turn.
        """
        blocks: list[str] = []
        for name in names:
            skill = self._skills.get(str(name))
            if skill is None:
                logger.warning("skill %r was requested but is not in the catalogue", name)
                continue
            self._loaded.add(str(name))
            description = getattr(skill, "description", "") or ""
            instructions = getattr(skill, "instructions", "") or ""
            blocks.append(f"## Skill: {name}\n{description}\n\n{instructions}".rstrip())
        return "\n\n".join(blocks)

    def all_instructions(self) -> str:
        """Every skill's full body — the supported eager-loading path."""
        return self.instructions_for(self.names)

    @property
    def loaded(self) -> set[str]:
        """Names whose full instructions have been materialised."""
        return set(self._loaded)


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSummary:
    """A tool's searchable surface: name and one line, without its schema."""

    name: str
    description: str

    def render(self) -> str:
        return f"- {self.name}: {self.description}"


class ToolCatalog:
    """Searchable tool catalogue with lazy schema loading and a hard allow-list.

    Large tool registries are expensive to send in full: a hundred tools with
    nested parameter schemas can cost more than the task.  This catalogue sends
    names and one-line descriptions, and materialises full JSON schemas only for
    the tools a turn actually needs.

    **Authorisation is applied at construction.**  A tool outside
    ``allowed_tools`` is dropped from the catalogue entirely — it cannot be
    searched, described, scheduled, or executed, regardless of what the model
    asks for.  That is deliberately not a ranking signal.

    Example::

        catalog = ToolCatalog(registry, allowed_tools={"get_stock", "list_inventory"})
        hits = catalog.search("how many units are in warehouse A")
        schemas = catalog.schemas([h.name for h in hits])
    """

    def __init__(self, registry: Any, *, allowed_tools: Iterable[str] | None = None) -> None:
        self._registry = registry
        definitions = list(getattr(registry, "definitions", []) or [])
        allowed = None if allowed_tools is None else {str(t) for t in allowed_tools}
        self._allowed = allowed
        self._definitions = {d.name: d for d in definitions if allowed is None or d.name in allowed}
        self._excluded = sorted(d.name for d in definitions if allowed is not None and d.name not in allowed)
        self._schema_loads: list[str] = []

    @property
    def names(self) -> list[str]:
        """Every *authorised* tool name."""
        return sorted(self._definitions)

    @property
    def excluded(self) -> list[str]:
        """Tools the allow-list removed.  Reported, never reachable."""
        return list(self._excluded)

    @property
    def schema_loads(self) -> list[str]:
        """Tool names whose full schema has been materialised, in order."""
        return list(self._schema_loads)

    def summaries(self) -> list[ToolSummary]:
        return [
            ToolSummary(name=name, description=_first_line(getattr(d, "description", "")))
            for name, d in sorted(self._definitions.items())
        ]

    def summary_block(self, *, header: str = "## Available tools") -> str:
        summaries = self.summaries()
        if not summaries:
            return ""
        return "\n".join([header, *(s.render() for s in summaries)])

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_relevance: float = 0.0,
        rank: Callable[[str, ToolSummary], float] | None = None,
    ) -> list[ToolSummary]:
        """Rank authorised tools against *query* — summaries only, no schemas."""
        scorer = rank or (lambda q, s: _relevance(q, s.name.replace("_", " "), s.description))
        scored = [(scorer(query, s), s) for s in self.summaries()]
        hits = [(score, s) for score, s in scored if score >= min_relevance]
        hits.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [s for _score, s in hits[:limit]]

    def schemas(self, names: Iterable[str], *, strict: bool = False) -> list[dict[str, Any]]:
        """Full OpenAI-format tool schemas for *names*, loaded now.

        A name outside the allow-list is skipped and logged; it is never loaded,
        even if a model asked for it by name.
        """
        out: list[dict[str, Any]] = []
        for name in names:
            definition = self._definitions.get(str(name))
            if definition is None:
                logger.warning("tool %r is not available to this caller; refusing to load its schema", name)
                continue
            self._schema_loads.append(str(name))
            out.append(definition.to_openai_format(strict=strict))
        return out

    def subset_registry(self, names: Iterable[str]) -> Any:
        """A :class:`ToolRegistry` containing only *names*, intersected with the allow-list.

        This is what should be handed to an agent for a turn: the executable
        surface matches the schemas the model was shown, so a model cannot call
        something it was never offered.
        """
        wanted = {str(n) for n in names} & set(self._definitions)
        subset = getattr(self._registry, "subset", None)
        if subset is None:
            raise TypeError("the wrapped registry does not support subset()")
        return subset(sorted(wanted))


def _first_line(text: str) -> str:
    line = (text or "").strip().splitlines()
    return line[0].strip() if line else ""


def supports_native_tool_discovery(model: str, *, driver: Any = None) -> bool:
    """Whether *model*'s provider can be given tools natively.

    Used to choose between handing a provider its own tool definitions and
    falling back to the portable path (tool descriptions rendered into the
    prompt).  Resolution goes through the existing
    :mod:`prompture.infra.capabilities` registry, so a user override there
    applies here too.  Returns ``False`` on any lookup failure — a portable
    fallback that works everywhere is the safe default.
    """
    try:
        from ..infra.capabilities import get_capabilities

        return bool(get_capabilities(model, driver=driver).tool_use)
    except Exception:  # pragma: no cover - capability lookup is best-effort
        return False


# ---------------------------------------------------------------------------
# Scoped artifacts for large results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRef:
    """A handle to a stored value, plus enough metadata to decide about it.

    Attributes:
        handle: Opaque reference (``"artifact://<scope>/<id>"``).
        scope: Who may read it.  A read from another scope is refused.
        kind: Free-form type label (``"tool_result"``, ``"document"``).
        summary: One line describing what is inside.
        size_chars / approx_tokens: How expensive inlining it would be.
        source: Optional attribution (a tool name, a URI) preserved with it.
    """

    handle: str
    scope: str
    kind: str
    summary: str
    size_chars: int
    approx_tokens: int
    source: str = ""

    def render(self) -> str:
        """The line that goes in the prompt instead of the payload."""
        origin = f" from {self.source}" if self.source else ""
        return (
            f"[{self.kind}{origin}] {self.summary} "
            f"({self.size_chars} chars, ~{self.approx_tokens} tokens) -> {self.handle}"
        )


class ArtifactStore:
    """Stores large values once and hands out references and excerpts.

    A 200 KB tool result does not belong in a context window, but neither does
    it belong in the bin: a later turn may need three lines of it.  Store it,
    put :meth:`ArtifactRef.render` in the prompt, and let the run pull
    :meth:`excerpt` when it needs the detail.

    Scoping is enforced, not advisory: :meth:`get` and :meth:`excerpt` refuse a
    handle created under a different scope, so one task's tool output cannot be
    read by another's just because the handle leaked into a shared transcript.
    """

    def __init__(self) -> None:
        self._values: dict[str, tuple[str, Any, str]] = {}  # handle -> (scope, value, text)
        self._refs: dict[str, ArtifactRef] = {}
        self._lock = threading.RLock()

    # ---- writing ------------------------------------------------------

    def put(
        self,
        value: Any,
        *,
        scope: str,
        kind: str = "tool_result",
        summary: str = "",
        source: str = "",
    ) -> ArtifactRef:
        """Store *value* under *scope* and return its reference."""
        text = value if isinstance(value, str) else _to_text(value)
        handle = f"artifact://{scope}/{uuid.uuid4().hex[:12]}"
        ref = ArtifactRef(
            handle=handle,
            scope=scope,
            kind=kind,
            summary=summary or _auto_summary(text),
            size_chars=len(text),
            approx_tokens=count_tokens(text),
            source=source,
        )
        with self._lock:
            self._values[handle] = (scope, value, text)
            self._refs[handle] = ref
        return ref

    def maybe_put(
        self,
        value: Any,
        *,
        scope: str,
        inline_char_limit: int = 2000,
        **kwargs: Any,
    ) -> tuple[str, ArtifactRef | None]:
        """Inline small values; store large ones and return their reference line.

        Returns ``(text_for_the_prompt, ref_or_None)``.
        """
        text = value if isinstance(value, str) else _to_text(value)
        if len(text) <= inline_char_limit:
            return text, None
        ref = self.put(value, scope=scope, **kwargs)
        return ref.render(), ref

    # ---- reading ------------------------------------------------------

    def ref(self, handle: str) -> ArtifactRef | None:
        with self._lock:
            return self._refs.get(handle)

    def refs_for(self, scope: str) -> list[ArtifactRef]:
        with self._lock:
            return [r for r in self._refs.values() if r.scope == scope]

    def get(self, handle: str, *, scope: str) -> Any:
        """The stored value.

        Raises:
            KeyError: Unknown handle.
            PermissionError: The handle belongs to a different scope.
        """
        with self._lock:
            entry = self._values.get(handle)
        if entry is None:
            raise KeyError(f"Unknown artifact handle {handle!r}")
        owner, value, _text = entry
        if owner != scope:
            raise PermissionError(f"Artifact {handle!r} belongs to scope {owner!r} and is not readable from {scope!r}.")
        return value

    def excerpt(
        self,
        handle: str,
        *,
        scope: str,
        query: str = "",
        max_chars: int = 1200,
        context_chars: int = 200,
    ) -> str:
        """Load only the relevant part of a stored value.

        With a *query*, returns the windows around its best keyword matches;
        without one, returns the head of the value.  Either way the result is
        capped at ``max_chars``, and the return says when it was truncated so a
        model is not left thinking it saw everything.
        """
        with self._lock:
            entry = self._values.get(handle)
        if entry is None:
            raise KeyError(f"Unknown artifact handle {handle!r}")
        owner, _value, text = entry
        if owner != scope:
            raise PermissionError(f"Artifact {handle!r} belongs to scope {owner!r} and is not readable from {scope!r}.")

        if len(text) <= max_chars:
            return text

        if not query:
            return text[:max_chars] + f"\n… (truncated; {len(text)} chars total, handle {handle})"

        lowered = text.casefold()
        windows: list[tuple[int, int]] = []
        for term in sorted(_terms(query), key=len, reverse=True):
            if len(term) < 3:
                continue
            position = lowered.find(term)
            if position >= 0:
                windows.append((max(0, position - context_chars), min(len(text), position + context_chars)))
            if len(windows) >= 5:
                break

        if not windows:
            return text[:max_chars] + f"\n… (no match for the query; {len(text)} chars total, handle {handle})"

        windows.sort()
        merged: list[list[int]] = []
        for start, end in windows:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        pieces = [text[start:end] for start, end in merged]
        excerpt = "\n…\n".join(pieces)
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars]
        return excerpt + f"\n… (excerpt of {len(text)} chars, handle {handle})"


def _to_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(value)


def _auto_summary(text: str, *, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


@dataclass
class CompactionResult:
    """What compaction kept, dropped, and had to say about it."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    token_counter: str = "heuristic"
    preserved_constraints: list[str] = field(default_factory=list)
    preserved_sources: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": len(self.messages),
            "dropped": len(self.dropped),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "token_counter": self.token_counter,
            "preserved_constraints": list(self.preserved_constraints),
            "preserved_sources": list(self.preserved_sources),
            "unresolved_questions": list(self.unresolved_questions),
        }


def _message_tokens(message: dict[str, Any]) -> int:
    content = message.get("content")
    if isinstance(content, str):
        base = count_tokens(content)
    else:
        base = count_tokens(_to_text(content))
    for call in message.get("tool_calls") or ():
        base += count_tokens(_to_text(call))
    return base


def _tool_call_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for call in message.get("tool_calls") or ():
        identifier = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        if identifier:
            ids.add(str(identifier))
    return ids


def compact_messages(
    messages: Sequence[dict[str, Any]],
    *,
    max_tokens: int,
    keep_last: int = 4,
    constraints: Iterable[str] = (),
    summarize: Callable[[list[dict[str, Any]]], str] | None = None,
) -> CompactionResult:
    """Trim a message history to fit ``max_tokens`` without breaking it.

    Guarantees:

    * Every ``system`` message is kept, in place.
    * The last ``keep_last`` messages are kept.
    * A dropped assistant message takes its tool results with it, and a kept
      assistant message keeps its tool results — providers reject an orphaned
      ``tool`` message and a tool call with no result, so the pairing is
      structural, not cosmetic.
    * Declared ``constraints`` are re-stated in the summary, so a requirement
      stated once, early, survives an arbitrary number of compactions.
    * Source attributions (``metadata["source"]`` or a ``[source: …]`` marker)
      and unanswered questions found in dropped content are carried into the
      summary rather than lost.

    Args:
        messages: Chat-format messages.
        max_tokens: Target size for the kept messages.
        keep_last: How many trailing messages are always kept.
        constraints: Task constraints that must survive.
        summarize: Optional custom summariser for the dropped block.  Defaults
            to a deterministic, model-free digest — compaction should not itself
            require a model call.

    Returns:
        A :class:`CompactionResult`.  ``messages`` is the new history; when
        anything was dropped, a ``system`` message carrying the summary is
        inserted after the leading system messages.
    """
    messages = [dict(m) for m in messages]
    tokens_before = sum(_message_tokens(m) for m in messages)
    result = CompactionResult(
        tokens_before=tokens_before,
        token_counter=_counter_name(),
        preserved_constraints=[str(c) for c in constraints],
    )

    if tokens_before <= max_tokens:
        result.messages = messages
        result.tokens_after = tokens_before
        return result

    n = len(messages)
    protected = set()
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            protected.add(index)
    for index in range(max(0, n - keep_last), n):
        protected.add(index)

    # Map each tool result back to the assistant turn that requested it, so the
    # two are always dropped or kept together.
    call_owner: dict[str, int] = {}
    for index, message in enumerate(messages):
        for call_id in _tool_call_ids(message):
            call_owner[call_id] = index
    partners: dict[int, set[int]] = {}
    for index, message in enumerate(messages):
        if message.get("role") == "tool":
            owner = call_owner.get(str(message.get("tool_call_id") or ""))
            if owner is not None:
                partners.setdefault(owner, set()).add(index)
                partners.setdefault(index, set()).add(owner)

    # Expand protection across pairs before deciding anything.
    frontier = list(protected)
    while frontier:
        index = frontier.pop()
        for partner in partners.get(index, ()):
            if partner not in protected:
                protected.add(partner)
                frontier.append(partner)

    keep = set(protected)
    used = sum(_message_tokens(messages[i]) for i in keep)

    # Walk newest-first through the droppable middle, keeping what still fits.
    for index in range(n - 1, -1, -1):
        if index in keep:
            continue
        group = {index} | partners.get(index, set())
        if group & keep:
            continue
        cost = sum(_message_tokens(messages[i]) for i in group)
        if used + cost <= max_tokens:
            keep |= group
            used += cost

    kept_indices = sorted(keep)
    dropped_indices = [i for i in range(n) if i not in keep]
    result.dropped = [messages[i] for i in dropped_indices]

    # Salvage attribution and open questions from what is going away.
    sources: list[str] = []
    questions: list[str] = []
    for message in result.dropped:
        source = (message.get("metadata") or {}).get("source")
        if source:
            sources.append(str(source))
        content = message.get("content")
        if isinstance(content, str):
            sources.extend(re.findall(r"\[source:\s*([^\]]+)\]", content))
            if message.get("role") == "user":
                questions.extend(s.strip() for s in re.findall(r"([^.?!\n]*\?)", content) if s.strip())
    result.preserved_sources = list(dict.fromkeys(sources))
    result.unresolved_questions = list(dict.fromkeys(questions))[:10]

    kept_messages = [messages[i] for i in kept_indices]

    if result.dropped:
        result.summary = summarize(result.dropped) if summarize is not None else _default_summary(result)
        insert_at = 0
        while insert_at < len(kept_messages) and kept_messages[insert_at].get("role") == "system":
            insert_at += 1
        kept_messages.insert(
            insert_at,
            {"role": "system", "content": result.summary, "metadata": {"compaction": True}},
        )

    result.messages = kept_messages
    result.tokens_after = sum(_message_tokens(m) for m in kept_messages)
    return result


def _default_summary(result: CompactionResult) -> str:
    """A deterministic digest — no model call, so compaction never costs money."""
    lines = [f"[Context compacted: {len(result.dropped)} earlier message(s) removed.]"]
    if result.preserved_constraints:
        lines.append("Constraints that still apply:")
        lines.extend(f"- {c}" for c in result.preserved_constraints)
    if result.preserved_sources:
        lines.append("Sources referenced in the removed messages:")
        lines.extend(f"- {s}" for s in result.preserved_sources)
    if result.unresolved_questions:
        lines.append("Questions raised earlier and not yet answered:")
        lines.extend(f"- {q}" for q in result.unresolved_questions)
    return "\n".join(lines)
