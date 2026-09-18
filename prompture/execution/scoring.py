"""Deterministic scorers for the three fixture workloads.

Every scorer here is a pure function of *(case, produced artefacts)*.  None of
them call a model, so a score is reproducible and free — LLM judging, where it is
wanted, is layered on separately via :mod:`prompture.eval`.

What each scorer will and will not claim
----------------------------------------

* ``correct`` is the task-level verdict.  It is ``None`` whenever the check
  cannot be made (no expectation in the fixture, or the run never produced a
  gradeable artefact), and it is never inferred from schema validity.
* ``score`` is partial credit in ``0.0..1.0``.  For extraction it is the
  fraction of expected fields that matched; for the other two workloads it is
  ``1.0``/``0.0`` unless the fixture defines something finer.
* String comparison is exact after normalisation (case-folding, whitespace
  collapse, and — for phone numbers — reduction to digits).  A fixture that
  wants looser matching should express that in ``answer_contains``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .fixtures import FixtureCase
from .outcomes import TaskCategory
from .sandbox import normalize_inventory_state

__all__ = [
    "CaseScore",
    "normalize_text",
    "score_case",
    "score_document_qa",
    "score_extraction",
    "score_tool_task",
]

#: Values a model may emit that all mean "this field is not present".
_NULLISH = {"", "none", "null", "n/a", "na", "unknown", "not specified", "not provided", "-"}

_WS = re.compile(r"\s+")
_PUNCT_EDGES = re.compile(r"^[\s.,;:!?'\"()\[\]-]+|[\s.,;:!?'\"()\[\]-]+$")


@dataclass(frozen=True)
class CaseScore:
    """The graded outcome of one fixture case.

    Attributes:
        correct: Task-level verdict, or ``None`` when it could not be
            determined.
        score: Partial credit in ``0.0..1.0``.
        details: Per-field / per-check breakdown, useful for error analysis and
            for the targeted-repair escalation rules in
            :mod:`prompture.execution.policy`.
        scorer: Which scorer produced this, recorded in reports so a
            deterministic verdict is never confused with an LLM judgement.
    """

    correct: bool | None
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    scorer: str = "deterministic"


def normalize_text(value: Any) -> str:
    """Case-fold, collapse whitespace, and trim edge punctuation."""
    if value is None:
        return ""
    text = _WS.sub(" ", str(value)).strip()
    text = _PUNCT_EDGES.sub("", text)
    return text.casefold()


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, tuple, set)) and not value:
        return True
    return normalize_text(value) in _NULLISH


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _field_matches(name: str, expected: Any, actual: Any) -> bool:
    """Compare one extracted field against its expectation."""
    if _is_nullish(expected):
        return _is_nullish(actual)
    if _is_nullish(actual):
        return False
    if "phone" in name.casefold():
        # Formatting varies wildly; only the digit sequence is meaningful.
        return _digits(expected) == _digits(actual)
    return normalize_text(expected) == normalize_text(actual)


def score_extraction(case: FixtureCase, fields: dict[str, Any] | None) -> CaseScore:
    """Score a typed-extraction case by field-level agreement.

    Args:
        case: The fixture case; ``case.expected["fields"]`` drives the check.
        fields: The extracted object as a plain dict, or ``None`` when the run
            produced nothing (a failed or abstained run).

    Returns:
        A :class:`CaseScore` whose ``score`` is the fraction of expected fields
        that matched and whose ``correct`` is ``True`` only when every one did.
        ``correct`` is ``None`` when the fixture declares no expected fields.
    """
    expected = dict(case.expected.get("fields") or {})
    if not expected:
        return CaseScore(correct=None, score=0.0, details={"reason": "no expected fields in fixture"})
    if fields is None:
        return CaseScore(
            correct=False,
            score=0.0,
            details={"reason": "no output produced", "expected_fields": sorted(expected)},
        )

    per_field: dict[str, bool] = {}
    for name, want in expected.items():
        per_field[name] = _field_matches(name, want, fields.get(name))
    matched = sum(1 for ok in per_field.values() if ok)
    return CaseScore(
        correct=matched == len(expected),
        score=matched / len(expected),
        details={
            "per_field": per_field,
            "wrong_fields": sorted(n for n, ok in per_field.items() if not ok),
            "matched": matched,
            "total": len(expected),
        },
    )


def score_document_qa(
    case: FixtureCase,
    answer: str | None,
    *,
    cited_sources: list[str] | tuple[str, ...] = (),
    abstained: bool = False,
) -> CaseScore:
    """Score an evidence-grounded question.

    Two things are graded independently and then combined:

    1. **Answer content** — every string in ``expected["answer_contains"]`` must
       appear in the answer (normalised substring match).
    2. **Attribution** — at least one of ``expected["supporting_source_ids"]``
       must be cited.  Citing a source is not proof the answer is right; it is
       graded so that a right answer built on the wrong passage is visible.

    For an unanswerable case (``expected["answerable"] is False``) the *only*
    correct behaviour is abstaining.  Producing a fluent answer scores 0 even if
    the prose hedges.

    Args:
        case: The fixture case.
        answer: The produced answer text, or ``None``.
        cited_sources: Source ids the run attributed the answer to.
        abstained: Whether the run explicitly declined for lack of evidence.
    """
    answerable = bool(case.expected.get("answerable", True))
    details: dict[str, Any] = {"answerable": answerable, "abstained": abstained}

    if not answerable:
        details["reason"] = "unanswerable case; abstention is the only correct outcome"
        return CaseScore(correct=abstained, score=1.0 if abstained else 0.0, details=details)

    if abstained:
        details["reason"] = "abstained on an answerable case"
        return CaseScore(correct=False, score=0.0, details=details)

    needles = [str(n) for n in case.expected.get("answer_contains") or []]
    haystack = normalize_text(answer)
    if not haystack:
        details["reason"] = "no answer produced"
        return CaseScore(correct=False, score=0.0, details=details)

    found = {n: (normalize_text(n) in haystack) for n in needles}
    content_ok = all(found.values()) if needles else True
    details["answer_contains"] = found

    wanted_sources = {str(s) for s in case.expected.get("supporting_source_ids") or []}
    cited = {str(s) for s in cited_sources or ()}
    if wanted_sources:
        attribution_ok = bool(wanted_sources & cited)
        details["expected_sources"] = sorted(wanted_sources)
        details["cited_sources"] = sorted(cited)
        details["attribution_ok"] = attribution_ok
    else:
        attribution_ok = True
        details["attribution_ok"] = None

    # Content is the verdict; attribution is half the partial credit so a
    # right-answer/wrong-source run is distinguishable in a report.
    score = (0.5 if content_ok else 0.0) + (0.5 if attribution_ok else 0.0)
    return CaseScore(correct=content_ok and attribution_ok, score=score, details=details)


def score_tool_task(
    case: FixtureCase,
    final_state: dict[str, Any] | None,
    *,
    answer: str | None = None,
    abstained: bool = False,
) -> CaseScore:
    """Score a bounded tool task by comparing world state, not prose.

    A case carrying ``expected["should_abstain"]`` is only correct when the run
    both left the world in the expected state *and* reported that it could not
    complete the request.  That is what stops "I moved the stock" from passing
    when nothing moved.

    Args:
        case: The fixture case.
        final_state: The sandbox world state after the run.
        answer: Optional produced text, checked against
            ``expected["answer_contains"]`` when the fixture has one.
        abstained: Whether the run reported that it could not complete.
    """
    expected_state = case.expected.get("final_state")
    if expected_state is None:
        return CaseScore(correct=None, score=0.0, details={"reason": "no expected final_state in fixture"})
    if final_state is None:
        return CaseScore(correct=False, score=0.0, details={"reason": "no final state captured"})

    want = normalize_inventory_state(expected_state)
    got = normalize_inventory_state(final_state)
    state_ok = want == got
    details: dict[str, Any] = {"state_ok": state_ok, "expected_state": want, "final_state": got}

    checks = [state_ok]

    if case.expected.get("should_abstain"):
        details["should_abstain"] = True
        details["abstained"] = abstained
        checks.append(abstained)

    needles = [str(n) for n in case.expected.get("answer_contains") or []]
    if needles:
        haystack = normalize_text(answer)
        found = {n: (normalize_text(n) in haystack) for n in needles}
        details["answer_contains"] = found
        checks.append(all(found.values()))

    passed = sum(1 for c in checks if c)
    return CaseScore(correct=all(checks), score=passed / len(checks), details=details)


def score_case(
    case: FixtureCase,
    *,
    fields: dict[str, Any] | None = None,
    answer: str | None = None,
    cited_sources: list[str] | tuple[str, ...] = (),
    final_state: dict[str, Any] | None = None,
    abstained: bool = False,
) -> CaseScore:
    """Dispatch to the scorer matching ``case.category``.

    Callers that hold a :class:`~prompture.execution.strategies.base.ExecutionResult`
    should use :func:`prompture.execution.harness.score_result`, which unpacks
    the envelope into these arguments.
    """
    if case.category is TaskCategory.EXTRACTION:
        return score_extraction(case, fields)
    if case.category is TaskCategory.DOCUMENT_QA:
        return score_document_qa(case, answer, cited_sources=cited_sources, abstained=abstained)
    if case.category is TaskCategory.TOOL_TASK:
        return score_tool_task(case, final_state, answer=answer, abstained=abstained)
    return CaseScore(correct=None, score=0.0, details={"reason": f"no scorer for {case.category.value!r}"})
