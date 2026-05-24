"""Quality filters for synthetic Q&A datasets.

A small, composable suite of predicates that drop low-value or unsafe
examples from a generated dataset. Use them via :class:`QualityFilter`
or call the predicate functions directly when you want fine-grained
control.

The filters intentionally **don't** require an embedding model or
extra dependencies for the common path — the refusal detector reuses
the Phase 6 ``RefusalDetector`` (stdlib-only), and length / shape
filters are pure Python. Optional semantic predicates are documented
in :mod:`prompture.dataset.dedup` instead.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .schemas import InstructionPair, QAPair

logger = logging.getLogger(__name__)


# A predicate returns True to KEEP the pair, False to DROP it. ``reason``
# is a short tag a caller can use for telemetry / debugging.
PairPredicate = Callable[[Any], "FilterDecision"]


@dataclass(frozen=True)
class FilterDecision:
    """Outcome of running one predicate against one pair.

    Attributes:
        keep: True when the pair should pass through.
        reason: Short snake_case tag explaining the verdict. Useful in
            telemetry (``"drop:short_answer"``, ``"keep"``).
    """

    keep: bool
    reason: str = "keep"

    @classmethod
    def keep_(cls, reason: str = "keep") -> FilterDecision:
        return cls(True, reason)

    @classmethod
    def drop(cls, reason: str) -> FilterDecision:
        return cls(False, f"drop:{reason}")


# ---------------------------------------------------------------------------
# Helpers — extract the comparable text from any pair shape we support.
# ---------------------------------------------------------------------------


def _question_text(pair: Any) -> str:
    if isinstance(pair, QAPair):
        return pair.question or ""
    if isinstance(pair, InstructionPair):
        return pair.instruction or ""
    if isinstance(pair, dict):
        return str(pair.get("question") or pair.get("instruction") or "")
    return str(pair)


def _answer_text(pair: Any) -> str:
    if isinstance(pair, QAPair):
        return pair.answer or ""
    if isinstance(pair, InstructionPair):
        return pair.output or ""
    if isinstance(pair, dict):
        return str(pair.get("answer") or pair.get("output") or "")
    return ""


# ---------------------------------------------------------------------------
# Built-in predicates
# ---------------------------------------------------------------------------


def length_filter(
    *,
    min_question_chars: int = 8,
    max_question_chars: int | None = 800,
    min_answer_chars: int = 1,
    max_answer_chars: int | None = 4000,
) -> PairPredicate:
    """Reject pairs whose question or answer falls outside the size window.

    The defaults catch the most common synth failures: empty / one-word
    questions and runaway answers that spilled an entire chunk back. Pass
    ``None`` to either ``max_*`` argument to disable the upper bound.
    """

    def predicate(pair: Any) -> FilterDecision:
        q = _question_text(pair).strip()
        a = _answer_text(pair).strip()
        if len(q) < min_question_chars:
            return FilterDecision.drop("short_question")
        if max_question_chars is not None and len(q) > max_question_chars:
            return FilterDecision.drop("long_question")
        if len(a) < min_answer_chars:
            return FilterDecision.drop("short_answer")
        if max_answer_chars is not None and len(a) > max_answer_chars:
            return FilterDecision.drop("long_answer")
        return FilterDecision.keep_()

    return predicate


def refusal_filter(detector: Any = None) -> PairPredicate:
    """Drop pairs whose ANSWER text scans as a refusal.

    Reuses :class:`prompture.refusal.RefusalDetector` (Phase 6). LLMs
    sometimes regurgitate "I can't help with that" into the dataset
    when the source content is borderline — those rows are pure noise
    for a training run. Pass a pre-built detector to override the
    marker list / language.
    """
    if detector is None:
        from ..refusal import RefusalDetector

        detector = RefusalDetector()

    def predicate(pair: Any) -> FilterDecision:
        answer = _answer_text(pair).strip()
        if not answer:
            return FilterDecision.keep_()
        result = detector.detect(answer)
        if getattr(result, "is_refusal", False):
            return FilterDecision.drop("refusal_in_answer")
        return FilterDecision.keep_()

    return predicate


_WHITESPACE_RE = re.compile(r"\s+")


def shape_filter() -> PairPredicate:
    """Reject obvious shape-level junk: empty fields, question without ``?`` *and* without 'how/what/why/etc.', identical Q and A."""

    interrogative = re.compile(r"^\s*(?:who|what|when|where|why|how|which|list|name|describe|explain|define)\b", re.IGNORECASE)

    def predicate(pair: Any) -> FilterDecision:
        q = _question_text(pair).strip()
        a = _answer_text(pair).strip()
        if not q or not a:
            return FilterDecision.drop("empty_field")
        if q.casefold() == a.casefold():
            return FilterDecision.drop("identical_q_and_a")
        # A question with NEITHER a question mark NOR an interrogative
        # prefix is probably a malformed statement masquerading as a Q.
        if "?" not in q and not interrogative.match(q):
            return FilterDecision.drop("not_a_question")
        return FilterDecision.keep_()

    return predicate


# ---------------------------------------------------------------------------
# Composite filter
# ---------------------------------------------------------------------------


@dataclass
class FilterStats:
    """Per-reason counts emitted by :meth:`QualityFilter.filter`."""

    total_in: int = 0
    total_out: int = 0
    dropped_by_reason: dict[str, int] = field(default_factory=dict)

    def drop_rate(self) -> float:
        if self.total_in == 0:
            return 0.0
        return (self.total_in - self.total_out) / self.total_in

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "dropped": self.total_in - self.total_out,
            "drop_rate": round(self.drop_rate(), 4),
            "dropped_by_reason": dict(self.dropped_by_reason),
        }


class QualityFilter:
    """Pipeline of predicates applied to each pair in order.

    The first predicate that returns ``keep=False`` wins (short-circuit).
    Call :meth:`filter` to get a cleaned list back + per-reason stats.

    Args:
        predicates: Predicates in evaluation order. ``None`` (default)
            wires the standard trio: :func:`shape_filter`,
            :func:`length_filter`, :func:`refusal_filter`.
        refusal_detector: Optional pre-built detector forwarded to
            :func:`refusal_filter` when the standard trio is used.
            Ignored when ``predicates`` is explicit.

    Example::

        from prompture.dataset.filters import QualityFilter
        clean, stats = QualityFilter().filter(pairs)
        print(stats.to_dict())
        # {'total_in': 240, 'total_out': 217, 'dropped': 23,
        #  'drop_rate': 0.0958,
        #  'dropped_by_reason': {'drop:short_question': 8, ...}}
    """

    def __init__(
        self,
        predicates: list[PairPredicate] | None = None,
        *,
        refusal_detector: Any = None,
    ) -> None:
        if predicates is None:
            predicates = [shape_filter(), length_filter(), refusal_filter(refusal_detector)]
        self.predicates = list(predicates)

    def evaluate(self, pair: Any) -> FilterDecision:
        """Run every predicate against ``pair``; return the first drop or a clean keep."""
        for pred in self.predicates:
            decision = pred(pair)
            if not decision.keep:
                return decision
        return FilterDecision.keep_()

    def filter(self, pairs: Iterable[Any]) -> tuple[list[Any], FilterStats]:
        """Apply the pipeline to every pair; return ``(kept, stats)``."""
        stats = FilterStats()
        kept: list[Any] = []
        for pair in pairs:
            stats.total_in += 1
            decision = self.evaluate(pair)
            if decision.keep:
                kept.append(pair)
                stats.total_out += 1
            else:
                stats.dropped_by_reason[decision.reason] = (
                    stats.dropped_by_reason.get(decision.reason, 0) + 1
                )
        return kept, stats

    def iter(self, pairs: Iterable[Any]) -> Iterator[Any]:
        """Stream-filter — yields kept pairs without buffering. Stats are not tracked."""
        for pair in pairs:
            if self.evaluate(pair).keep:
                yield pair


__all__ = [
    "FilterDecision",
    "FilterStats",
    "PairPredicate",
    "QualityFilter",
    "length_filter",
    "refusal_filter",
    "shape_filter",
]
