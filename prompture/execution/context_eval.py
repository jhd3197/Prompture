"""Measuring whether dynamic context assembly actually helps.

Fewer tokens is not an improvement.  A catalogue that sends nothing at all is
maximally cheap and completely useless, so a token saving only means something
alongside **selection accuracy** — did the right tools and skills survive the
filter? — and alongside the quality and latency the run actually achieved.

This module measures the parts that can be measured deterministically and
offline:

* :func:`measure_tool_selection` — precision, recall and token cost of a
  catalogue's search against cases that declare which tools are genuinely
  required, plus how often an unauthorised tool leaked through (which must be
  zero, not merely low).
* :func:`measure_compaction` — token reduction *and* the structural invariants:
  protected content intact, tool-call pairing intact, constraints and
  attributions carried forward.

Both return reports that refuse to summarise themselves into a single "better"
number.  Quality is not in here, because quality needs a model; wire the
assembled context into a :mod:`prompture.execution.strategies` run and score it
with :mod:`prompture.execution.harness` for that.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .context import CompactionResult, ToolCatalog, count_tokens

__all__ = [
    "CompactionReport",
    "SelectionCase",
    "ToolSelectionReport",
    "measure_compaction",
    "measure_tool_selection",
]


@dataclass(frozen=True)
class SelectionCase:
    """One "given this task, these tools are needed" example.

    Attributes:
        query: The task text a catalogue search is given.
        required: Tool names a correct selection must include.
        forbidden: Tool names that must never appear — typically the ones the
            caller is not authorised for.  A single appearance is a failure, not
            a lowered score.
    """

    query: str
    required: frozenset[str] = frozenset()
    forbidden: frozenset[str] = frozenset()


@dataclass
class ToolSelectionReport:
    """What a catalogue's filtering achieved, and what it cost.

    Attributes:
        cases: How many selection cases ran.
        catalog_size: Authorised tools in the catalogue.
        mean_precision / mean_recall: Over the cases that declared required
            tools.  Recall is the one that matters most — a missing tool makes
            the task impossible, while a spare one only costs tokens.
        perfect_recall_rate: Fraction of cases where every required tool was
            selected.
        unauthorised_leaks: Times a forbidden tool appeared.  Must be 0.
        full_catalog_tokens: Cost of sending every schema.
        mean_selected_tokens: Mean cost of sending only the selected schemas.
        token_reduction: ``1 - mean_selected/full``.  Meaningless on its own —
            read it next to ``mean_recall``.
        mean_latency_ms: Search time.
        notes: Caveats a reader must not skip.
    """

    cases: int = 0
    catalog_size: int = 0
    mean_precision: float | None = None
    mean_recall: float | None = None
    perfect_recall_rate: float | None = None
    unauthorised_leaks: int = 0
    full_catalog_tokens: int = 0
    mean_selected_tokens: float | None = None
    token_reduction: float | None = None
    mean_latency_ms: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        """No unauthorised tool was ever selected."""
        return self.unauthorised_leaks == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "catalog_size": self.catalog_size,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "perfect_recall_rate": self.perfect_recall_rate,
            "unauthorised_leaks": self.unauthorised_leaks,
            "safe": self.safe,
            "full_catalog_tokens": self.full_catalog_tokens,
            "mean_selected_tokens": self.mean_selected_tokens,
            "token_reduction": self.token_reduction,
            "mean_latency_ms": self.mean_latency_ms,
            "notes": list(self.notes),
        }

    def format(self) -> str:
        def num(value: float | None, fmt: str = "{:.3f}") -> str:
            return "n/a" if value is None else fmt.format(value)

        lines = [
            f"tool selection over {self.cases} case(s), catalogue of {self.catalog_size} authorised tool(s)",
            f"  recall        : {num(self.mean_recall)} (perfect on {num(self.perfect_recall_rate)} of cases)",
            f"  precision     : {num(self.mean_precision)}",
            f"  authorisation : {'SAFE' if self.safe else f'{self.unauthorised_leaks} LEAK(S)'}",
            f"  tokens        : {num(self.mean_selected_tokens, '{:.0f}')} selected "
            f"vs {self.full_catalog_tokens} for the full catalogue "
            f"({num(self.token_reduction, '{:.1%}')} reduction)",
            f"  latency       : {num(self.mean_latency_ms, '{:.2f}')}ms per search",
        ]
        lines.extend(f"  NOTE          : {n}" for n in self.notes)
        return "\n".join(lines)


def measure_tool_selection(
    catalog: ToolCatalog,
    cases: Sequence[SelectionCase],
    *,
    limit: int = 5,
    min_relevance: float = 0.0,
) -> ToolSelectionReport:
    """Score a :class:`ToolCatalog`'s search against labelled cases.

    Args:
        catalog: The catalogue under test, already constructed with whatever
            allow-list the caller has.
        cases: Labelled selection examples.
        limit: How many tools each search may return.
        min_relevance: Relevance floor for the search.
    """
    report = ToolSelectionReport(cases=len(cases), catalog_size=len(catalog.names))
    if not cases:
        report.notes.append("no cases supplied; nothing was measured")
        return report

    report.full_catalog_tokens = count_tokens(str(catalog.schemas(catalog.names)))

    precisions: list[float] = []
    recalls: list[float] = []
    perfect = 0
    selected_tokens: list[int] = []
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        hits = catalog.search(case.query, limit=limit, min_relevance=min_relevance)
        latencies.append((time.perf_counter() - started) * 1000)

        names = {h.name for h in hits}
        selected_tokens.append(count_tokens(str(catalog.schemas(sorted(names)))))

        leaked = names & set(case.forbidden)
        report.unauthorised_leaks += len(leaked)

        if case.required:
            hit_count = len(names & set(case.required))
            recalls.append(hit_count / len(case.required))
            precisions.append(hit_count / len(names) if names else 0.0)
            if hit_count == len(case.required):
                perfect += 1

    if recalls:
        report.mean_recall = statistics.fmean(recalls)
        report.mean_precision = statistics.fmean(precisions)
        report.perfect_recall_rate = perfect / len(recalls)
    else:
        report.notes.append("no case declared required tools; precision and recall are unmeasured")

    if selected_tokens:
        report.mean_selected_tokens = statistics.fmean(selected_tokens)
        if report.full_catalog_tokens:
            report.token_reduction = 1 - (report.mean_selected_tokens / report.full_catalog_tokens)
    report.mean_latency_ms = statistics.fmean(latencies) if latencies else None

    report.notes.append(
        "token reduction alone is not an improvement; read it together with recall, "
        "and confirm end-task quality separately with a scored benchmark run"
    )
    return report


@dataclass
class CompactionReport:
    """Token reduction plus the structural invariants that must hold.

    Attributes:
        tokens_before / tokens_after / reduction: The saving.
        protected_intact: Every ``system`` message survived verbatim.
        pairing_intact: No orphaned tool result and no dangling tool call.
        constraints_preserved: Every declared constraint appears in the result.
        sources_preserved: How many attributions were carried forward.
        questions_preserved: How many unanswered questions were carried forward.
        messages_dropped: Count.
        notes: Caveats.
    """

    tokens_before: int = 0
    tokens_after: int = 0
    reduction: float | None = None
    protected_intact: bool = True
    pairing_intact: bool = True
    constraints_preserved: bool = True
    sources_preserved: int = 0
    questions_preserved: int = 0
    messages_dropped: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def sound(self) -> bool:
        """Whether every structural invariant held.  A saving without this is a bug."""
        return self.protected_intact and self.pairing_intact and self.constraints_preserved

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "reduction": self.reduction,
            "protected_intact": self.protected_intact,
            "pairing_intact": self.pairing_intact,
            "constraints_preserved": self.constraints_preserved,
            "sources_preserved": self.sources_preserved,
            "questions_preserved": self.questions_preserved,
            "messages_dropped": self.messages_dropped,
            "sound": self.sound,
            "notes": list(self.notes),
        }


def measure_compaction(
    original: Sequence[dict[str, Any]],
    result: CompactionResult,
    *,
    constraints: Sequence[str] = (),
) -> CompactionReport:
    """Check a :func:`~prompture.execution.context.compact_messages` result.

    Verifies the invariants rather than trusting them: a compaction that halves
    the token count but leaves a dangling tool call has not saved anything, it
    has produced a request the provider will reject.
    """
    report = CompactionReport(
        tokens_before=result.tokens_before,
        tokens_after=result.tokens_after,
        messages_dropped=len(result.dropped),
        sources_preserved=len(result.preserved_sources),
        questions_preserved=len(result.unresolved_questions),
    )
    if result.tokens_before:
        report.reduction = 1 - (result.tokens_after / result.tokens_before)

    original_systems = [m for m in original if m.get("role") == "system"]
    kept_systems = [
        m for m in result.messages if m.get("role") == "system" and not (m.get("metadata") or {}).get("compaction")
    ]
    report.protected_intact = all(m in kept_systems for m in original_systems)

    call_ids = {
        str(call.get("id"))
        for message in result.messages
        for call in (message.get("tool_calls") or ())
        if isinstance(call, dict) and call.get("id")
    }
    result_ids = {
        str(message.get("tool_call_id"))
        for message in result.messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    report.pairing_intact = call_ids == result_ids

    rendered = "\n".join(str(m.get("content") or "") for m in result.messages)
    report.constraints_preserved = all(str(c) in rendered for c in constraints)

    if not report.sound:
        report.notes.append("a structural invariant was violated; this compaction is not usable")
    report.notes.append("token reduction is not by itself an improvement; confirm task quality with a scored run")
    return report
