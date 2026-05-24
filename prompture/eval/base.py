"""Shared types and helpers for :mod:`prompture.eval`.

* :class:`EvalDriver` — the minimal LLM-driver protocol every evaluator
  in this package accepts. Any object exposing
  ``generate(prompt, options) -> {"text": str, ...}`` qualifies, which
  matches every sync :class:`prompture.drivers.base.Driver` plus any
  user-supplied stub.
* Result dataclasses (one per evaluator) — frozen, hashable-friendly,
  with a ``to_dict()`` for report serialisation.
* :class:`EvalError` — single error type the package raises for any
  irrecoverable evaluation failure (so callers can ``except`` it
  without catching every underlying SDK error).

Evaluators that score things use Pydantic models to coerce the LLM
output into a structured shape via
:func:`prompture.extract_with_model` — the same mechanism the rest of
Prompture uses. That gives us free retry, validation, schema
tightening (Phase 5), and cache support.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class EvalError(RuntimeError):
    """Raised when an evaluator can't produce a usable result.

    Examples: the rubric is empty, the LLM consistently returns
    unparseable text after retries, a pairwise judge call fails
    twice in a row.
    """


class EvalDriver(Protocol):
    """Minimal LLM-driver contract used by every evaluator."""

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """Structured outcome from :meth:`LLMJudge.evaluate`.

    Attributes:
        score: Overall score on the rubric's 1..``scale`` integer range.
        reasoning: One-paragraph justification the LLM produced.
        per_criterion: ``{criterion_name: integer_score}`` — one entry
            per :class:`LLMJudge` criterion. Absent when the rubric had
            no named criteria.
        passed: ``True`` when ``score >= pass_threshold`` (or ``None``
            when no threshold was supplied).
        meta: Token + cost metadata from the underlying LLM call.
    """

    score: int
    reasoning: str
    per_criterion: dict[str, int] = field(default_factory=dict)
    passed: bool | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseResult:
    """Outcome of comparing two candidates.

    Attributes:
        winner: ``"A"``, ``"B"``, or ``"tie"``. ``"tie"`` is emitted
            when the two judge passes disagree on the winner with no
            confidence margin.
        confidence: Fraction of judge passes that picked ``winner``
            (0.0 – 1.0). With ``swap_positions=True`` the denominator
            is 2; otherwise 1.
        votes: Per-position votes, e.g. ``{"A_first": "A", "B_first": "B"}``.
        reasoning: Aggregated free-text reasoning from each pass.
        meta: Combined token + cost metadata across all judge calls.
    """

    winner: str
    confidence: float
    votes: dict[str, str] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelfConsistencyResult:
    """Outcome of N independent samples + majority vote.

    Attributes:
        consensus: Most-common normalised answer (string). ``None``
            when no samples produced a usable answer.
        agreement: Fraction of samples that matched ``consensus``
            (0.0 – 1.0). High values (>= 0.8) suggest low hallucination
            risk; low values flag uncertainty.
        samples: The N normalised answers in collection order so
            callers can compute their own statistics.
        counts: ``{answer: count}`` distribution.
        meta: Combined token + cost metadata across all sample calls.
    """

    consensus: str | None
    agreement: float
    samples: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FaithfulnessResult:
    """Outcome of RAG faithfulness checking.

    Attributes:
        score: Fraction of claims supported by the evidence
            (0.0 – 1.0). 1.0 means every claim was grounded; 0.0
            means none were.
        supported: List of supported claim strings.
        unsupported: List of unsupported claim strings — the
            primary thing to inspect when the score is low.
        contradicted: List of claims the evidence actively contradicts
            (a subset of unsupported, surfaced separately so callers
            can distinguish "no evidence" from "evidence says no").
        meta: Combined token + cost metadata.
    """

    score: float
    supported: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    contradicted: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _accumulate_meta(target: dict[str, Any], chunk: dict[str, Any] | None) -> None:
    """In-place sum of token / cost / count fields from a per-call ``meta``.

    Used by every evaluator to roll up costs across multiple LLM
    calls into the result's ``meta`` field.
    """
    if not chunk:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(chunk.get(key), (int, float)):
            target[key] = target.get(key, 0) + int(chunk[key])
    if isinstance(chunk.get("cost"), (int, float)):
        target["cost"] = float(target.get("cost", 0.0)) + float(chunk["cost"])
    target["calls"] = target.get("calls", 0) + 1
