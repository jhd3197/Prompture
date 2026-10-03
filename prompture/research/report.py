"""Result types for :class:`~prompture.research.ResearchAgent`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReportSource:
    """A URL the run found, and whether it was actually read.

    Attributes:
        n: Citation number used in the answer, ``None`` for pages that were
            found but not opened (those can never be cited).
        url: The URL as found.
        title: Page title (from the reader when opened, else the search hit).
        opened: ``True`` when the page content was read and given to synthesis.
        reader: Reader or backend that served the content (``youtube``,
            ``jina_reader``, ``pack:finance``, ...).
        kind: Content kind reported by the reader (``html``, ``video``, ...).
        cited: ``True`` when the answer cites this source.
        sub_questions: Indexes of the sub-questions this source was gathered for.
        origins: Where the URL came from (``web``, ``github``, ``pack``, ...).
        transcribed: ``True`` when the content is a media transcript.
        chars: Characters of content read.
        snippet: Search snippet (for unopened sources) or opening excerpt.
        error: Why opening failed, when it did.
        route: Backend route reported by the reader/fetcher.
    """

    n: int | None
    url: str
    title: str = ""
    opened: bool = False
    reader: str | None = None
    kind: str | None = None
    cited: bool = False
    sub_questions: list[int] = field(default_factory=list)
    origins: list[str] = field(default_factory=list)
    transcribed: bool = False
    chars: int = 0
    snippet: str = ""
    error: str | None = None
    route: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "url": self.url,
            "title": self.title,
            "opened": self.opened,
            "reader": self.reader,
            "kind": self.kind,
            "cited": self.cited,
            "sub_questions": list(self.sub_questions),
            "origins": list(self.origins),
            "transcribed": self.transcribed,
            "chars": self.chars,
            "snippet": self.snippet,
            "error": self.error,
            "route": self.route,
        }


@dataclass
class SubQuestionResult:
    """A planned sub-question and what was found for it."""

    question: str
    queries: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    candidates: int = 0

    @property
    def answered(self) -> bool:
        return bool(self.sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "queries": list(self.queries),
            "platforms": list(self.platforms),
            "sources": list(self.sources),
            "candidates": self.candidates,
            "answered": self.answered,
        }


@dataclass
class ResearchReport:
    """The outcome of a research run.

    Attributes:
        question: The question asked.
        answer: Markdown answer with inline ``[n]`` citations; every number
            refers to an opened source in :attr:`sources`.
        sources: Opened sources (numbered) followed by unopened hits.
        sub_questions: The plan and per-sub-question coverage.
        gaps: What the gathered sources don't answer, plus early-stop notes.
        conflicts: Claims on which opened sources disagree, with citations.
        budget_used: Spend against each limit (see :meth:`BudgetTracker.snapshot`).
        cost: Total USD cost of LLM calls.
        usage: Aggregated LLM token usage.
        routes: Backend route per search/fetch (which backend served it, fallbacks).
        model: Model used for planning and synthesis.
        depth: Depth preset used.
        plan_source: ``"llm"`` or ``"heuristic"``.
        synthesis: ``"llm"``, ``"extractive"`` (limit hit or synthesis failed)
            or ``"none"`` (nothing could be opened).
        warnings: Degradations worth surfacing (failed backends, dropped citations).
        elapsed_s: Wall-clock seconds for the run.
    """

    question: str
    answer: str
    sources: list[ReportSource] = field(default_factory=list)
    sub_questions: list[SubQuestionResult] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    budget_used: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    routes: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    depth: str = "standard"
    plan_source: str = "llm"
    synthesis: str = "llm"
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def opened_sources(self) -> list[ReportSource]:
        return [s for s in self.sources if s.opened]

    @property
    def cited_sources(self) -> list[ReportSource]:
        return [s for s in self.sources if s.cited]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (stable keys; used by ``prompture research --json``)."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "sub_questions": [q.to_dict() for q in self.sub_questions],
            "gaps": list(self.gaps),
            "conflicts": list(self.conflicts),
            "budget_used": dict(self.budget_used),
            "cost": self.cost,
            "usage": dict(self.usage),
            "routes": list(self.routes),
            "model": self.model,
            "depth": self.depth,
            "plan_source": self.plan_source,
            "synthesis": self.synthesis,
            "warnings": list(self.warnings),
            "elapsed_s": self.elapsed_s,
        }

    def to_markdown(self, *, include_unopened: int = 10) -> str:
        """Render the report: answer, conflicts, gaps, opened sources and a budget line."""
        lines: list[str] = [f"# {self.question}", "", self.answer.strip() or "_No answer._", ""]
        if self.conflicts:
            lines += ["## Conflicting claims", ""]
            lines += [f"- {c}" for c in self.conflicts]
            lines.append("")
        if self.gaps:
            lines += ["## Coverage gaps", ""]
            lines += [f"- {g}" for g in self.gaps]
            lines.append("")
        opened = self.opened_sources
        lines += ["## Sources", ""]
        if opened:
            for s in opened:
                via = f" — via {s.reader}" if s.reader else ""
                extra = " (transcript)" if s.transcribed else ""
                title = s.title or s.url
                lines.append(f"{s.n}. [{_md_escape(title)}]({s.url}){extra}{via}")
        else:
            lines.append("_No sources could be opened._")
        unopened = [s for s in self.sources if not s.opened]
        if unopened and include_unopened > 0:
            lines += ["", "<details><summary>Found but not opened</summary>", ""]
            for s in unopened[:include_unopened]:
                reason = f" — {s.error}" if s.error else ""
                lines.append(f"- [{_md_escape(s.title or s.url)}]({s.url}){reason}")
            if len(unopened) > include_unopened:
                lines.append(f"- ... and {len(unopened) - include_unopened} more")
            lines += ["", "</details>"]
        lines += ["", _budget_line(self)]
        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        return self.to_markdown()


def _md_escape(text: str) -> str:
    return text.replace("[", "(").replace("]", ")").replace("\n", " ").strip()


def _budget_line(report: ResearchReport) -> str:
    b = report.budget_used
    parts = [
        f"{b.get('fetches', 0)}/{b.get('max_fetches') or '∞'} fetches",
        f"{b.get('searches', 0)} searches",
        f"{b.get('tokens', 0):,} tokens",
        f"${report.cost:.4f}",
        f"{b.get('elapsed_s', report.elapsed_s)}s",
    ]
    line = f"_{report.depth} research with {report.model or 'no model'}: " + ", ".join(parts)
    if b.get("limits_hit"):
        line += f"; stopped early on {', '.join(b['limits_hit'])}"
    return line + "_"


__all__ = ["ReportSource", "ResearchReport", "SubQuestionResult"]
