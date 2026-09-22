"""Base classes for decision drivers ("System One" typed-decision models).

Decision is a first-class modality alongside LLM, embedding, rerank, moderation,
STT, TTS, image-gen and video-gen.  A decision model evaluates a *state* (text or
structured data) against a map of typed *questions* and returns one typed
*answer* per question — no prose, nothing to parse, nothing to hallucinate.

Three question primitives, shared by every provider in this space:

* :class:`Noul` — a yes/no question; returns a calibrated probability of "yes".
* :class:`Choice` — pick one option from a set; returns the winner, the full
  probability distribution, and a confidence.
* :class:`Score` — rate against an ordinal rubric; returns a probability-weighted
  value across the levels, the legend, the distribution, and a confidence.

The wire format is the one TypeSafe published for Jev and that the open
alternatives (Kev, Laya) adopted verbatim, so a single interface covers hosted
APIs, self-hosted servers and in-process weights.

Usage::

    from prompture.drivers.decision_registry import get_decision_driver_for_model
    from prompture.drivers.decision_base import Choice, Noul, Score

    driver = get_decision_driver_for_model("typesafe/jev-latest")
    result = driver.decide(
        state={"subject": "Duplicate charge", "body": "We were billed twice."},
        questions={
            "department": Choice(
                "Which team should handle this?",
                criteria={"billing": "invoices, refunds", "technical": "bugs, outages"},
            ),
            "is_urgent": Noul("Does this convey urgency?"),
            "frustration": Score(
                "How frustrated is the customer?",
                criteria=["Calm", "Frustrated", "Very angry"],
            ),
        },
    )

    result["department"].choice        # -> "billing"
    result["department"].confidence    # -> 0.81
    result["is_urgent"].noul           # -> 0.95
    result["frustration"].score        # -> 1.05

    # Usage / cost metadata for the most recent call is exposed via ``last_usage``.
    print(driver.last_usage)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.decision_driver")

# The three primitives every System One provider implements.
QUESTION_TYPES: tuple[str, ...] = ("noul", "choice", "score")

# ``state`` and ``instructions`` accept a string, a mapping, or a sequence.
StateLike = str | Mapping[str, Any] | Sequence[Any]


# ── Pricing ────────────────────────────────────────────────────────────────

_RATES_DIR = Path(__file__).resolve().parent.parent / "infra" / "rates"
_decision_pricing_cache: dict[str, dict[str, dict[str, float]]] = {}


def _load_decision_pricing(provider: str) -> dict[str, dict[str, float]]:
    """Return ``{model_id: {"input": ..., "output": ...}}`` parsed from the
    provider's rates JSON, keeping only ``"modality": "decision"`` entries.

    Results are cached.  Self-hosted providers (Kev, Laya) ship zero-cost
    entries so that ``pricing_unknown`` stays ``False`` for them — free is a
    known price, not an unknown one.
    """
    if provider in _decision_pricing_cache:
        return _decision_pricing_cache[provider]
    out: dict[str, dict[str, float]] = {}
    path = _RATES_DIR / f"{provider}.json"
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse decision pricing from %s", path)
            raw = {}
        for model_id, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("modality") != "decision":
                continue
            cost = entry.get("cost") or {}
            if not isinstance(cost, dict):
                continue
            slim: dict[str, float] = {}
            for key in ("input", "output"):
                val = cost.get(key)
                if isinstance(val, (int, float)):
                    slim[key] = float(val)
            if slim:
                out[model_id] = slim
    _decision_pricing_cache[provider] = out
    return out


def calculate_decision_cost(
    provider: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> tuple[float, bool]:
    """Calculate USD cost for a decision call.

    Rates in the KB are per *million* tokens.  Jev bills input only (output
    tokens are free), so a missing ``output`` rate is treated as 0.0 rather
    than as unknown.

    Returns:
        ``(cost, pricing_unknown)``.  ``pricing_unknown`` is ``True`` when no
        pricing entry was found for *model*.
    """
    pricing = _load_decision_pricing(provider).get(model)
    if pricing is None:
        return 0.0, True
    cost = (input_tokens / 1_000_000) * pricing.get("input", 0.0)
    cost += (output_tokens / 1_000_000) * pricing.get("output", 0.0)
    return round(cost, 8), False


# ── Questions ──────────────────────────────────────────────────────────────


@dataclass
class Question:
    """Base class for the three question primitives."""

    #: Set by each subclass; serialized as the payload's ``type`` field.
    type: str = field(init=False, default="")

    def to_payload(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class Noul(Question):
    """A yes/no question.  The answer is the probability that it is yes.

    Args:
        instructions: The yes/no question to evaluate.
        criteria: Optional ``{"true": ..., "false": ...}`` descriptions of what
            a yes and a no mean.
    """

    instructions: StateLike
    criteria: Mapping[str, StateLike] | None = None

    def __post_init__(self) -> None:
        self.type = "noul"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = dict(self.criteria)
        return payload


@dataclass
class Choice(Question):
    """Pick one option from a set.

    Args:
        instructions: What the model should decide.
        criteria: ``{option: rubric}``.  Use ``None`` as the rubric when an
            option needs no extra detail.  Providers cap the option count —
            Jev allows 255; the open models degrade well before that, so
            prefer a coarse/fine split or a shortlist for large label sets.
    """

    instructions: StateLike
    criteria: Mapping[str, StateLike | None]

    def __post_init__(self) -> None:
        self.type = "choice"
        if not self.criteria:
            raise ValueError("Choice questions require at least one option in `criteria`.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass
class Score(Question):
    """Rate the state against an ordinal rubric.

    Args:
        instructions: What the model should rate.
        criteria: Ordered level descriptions, lowest first — e.g.
            ``["Calm", "Frustrated", "Very angry"]`` scores 0..2.
    """

    instructions: StateLike
    criteria: Sequence[StateLike]

    def __post_init__(self) -> None:
        self.type = "score"
        if not self.criteria:
            raise ValueError("Score questions require at least one level in `criteria`.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }


QuestionLike = Question | Mapping[str, Any]


def normalize_questions(questions: Mapping[str, QuestionLike]) -> dict[str, dict[str, Any]]:
    """Serialize a question map to the wire payload.

    Accepts :class:`Question` instances and raw dicts interchangeably, so code
    holding a JSON question schema can pass it straight through.

    Raises:
        ValueError: If *questions* is empty, or an entry has an unknown type.
    """
    if not questions:
        raise ValueError("At least one question is required.")
    out: dict[str, dict[str, Any]] = {}
    for qid, q in questions.items():
        if isinstance(q, Question):
            out[qid] = q.to_payload()
            continue
        if not isinstance(q, Mapping):
            raise ValueError(f"Question {qid!r} must be a Question instance or a mapping, got {type(q).__name__}.")
        payload = dict(q)
        qtype = payload.get("type")
        if qtype not in QUESTION_TYPES:
            raise ValueError(f"Question {qid!r} has unknown type {qtype!r}; expected one of {QUESTION_TYPES}.")
        out[qid] = payload
    return out


# ── Answers ────────────────────────────────────────────────────────────────


@dataclass
class Answer:
    """Base class for the three answer shapes."""

    #: Set by each subclass; mirrors the question type that produced it.
    type: str = field(init=False, default="")

    @property
    def value(self) -> Any:  # pragma: no cover - overridden
        """The primitive result, whatever this answer's type is."""
        raise NotImplementedError


@dataclass
class NoulAnswer(Answer):
    """A yes/no answer.

    Attributes:
        noul: Probability the answer is yes, from 0.0 (no) to 1.0 (yes).
    """

    noul: float
    #: Populated only when the provider sends its own ``confidence`` for a
    #: noul.  TypeSafe and Kev do not; Laya does.
    reported_confidence: float | None = None

    def __post_init__(self) -> None:
        self.type = "noul"

    @property
    def value(self) -> float:
        return self.noul

    @property
    def confidence(self) -> float:
        """How certain the model is, on 0..1.

        Prefers the provider's own ``confidence`` when it sends one.  Otherwise
        derived locally as distance from the coin flip, because most of this
        family omits the field for nouls — there the probability *is* the
        answer.  Either way confidence-gating code can treat all three
        primitives uniformly.
        """
        if self.reported_confidence is not None:
            return self.reported_confidence
        return abs(self.noul - 0.5) * 2.0


@dataclass
class ChoiceAnswer(Answer):
    """A single-option answer.

    Attributes:
        choice: The highest-probability option.
        probabilities: Every option mapped to its probability (sums to 1).
        confidence: How certain the model is, derived from ``probabilities``.
    """

    choice: str
    probabilities: dict[str, float]
    confidence: float

    def __post_init__(self) -> None:
        self.type = "choice"

    @property
    def value(self) -> str:
        return self.choice

    def ranked(self) -> list[tuple[str, float]]:
        """Options sorted by descending probability."""
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class ScoreAnswer(Answer):
    """An ordinal-rubric answer.

    Attributes:
        score: Probability-weighted value across the levels; can land between
            levels (e.g. 1.05 on a 0..2 rubric).
        legend: Each level number mapped back to its description.
        probabilities: Each level (string key) mapped to its probability.
        confidence: How certain the model is, derived from ``probabilities``.
    """

    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float

    def __post_init__(self) -> None:
        self.type = "score"

    @property
    def value(self) -> float:
        return self.score

    @property
    def nearest_level(self) -> str:
        """The legend description of the level nearest to ``score``."""
        if not self.legend:
            return ""
        nearest = min(self.legend, key=lambda k: abs(float(k) - self.score))
        return self.legend[nearest]


def parse_answer(qid: str, raw: Mapping[str, Any]) -> Answer:
    """Build a typed :class:`Answer` from one raw answer object.

    Raises:
        ValueError: If the answer's ``type`` is missing or unrecognized.
    """
    atype = raw.get("type")
    if atype == "noul":
        reported = raw.get("confidence")
        return NoulAnswer(
            noul=float(raw.get("noul", 0.0)),
            reported_confidence=float(reported) if reported is not None else None,
        )
    if atype == "choice":
        probs = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
        choice = raw.get("choice")
        if choice is None and probs:
            choice = max(probs, key=lambda k: probs[k])
        return ChoiceAnswer(
            choice=str(choice) if choice is not None else "",
            probabilities=probs,
            confidence=float(raw.get("confidence", 0.0)),
        )
    if atype == "score":
        return ScoreAnswer(
            score=float(raw.get("score", 0.0)),
            legend={str(k): str(v) for k, v in (raw.get("legend") or {}).items()},
            probabilities={str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()},
            confidence=float(raw.get("confidence", 0.0)),
        )
    raise ValueError(f"Answer {qid!r} has unknown type {atype!r}; expected one of {QUESTION_TYPES}.")


@dataclass
class DecisionResponse:
    """The result of one :meth:`DecisionDriver.decide` call.

    Attributes:
        model: The versioned model that answered (aliases resolve here, so log
            this rather than what you requested).
        answers: One typed :class:`Answer` per question, under your own ids.
        usage: Flat usage/cost dict, identical to ``driver.last_usage``.
        raw_response: The provider's unmodified response body.
    """

    model: str
    answers: dict[str, Answer]
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, qid: str) -> Answer:
        return self.answers[qid]

    def __contains__(self, qid: object) -> bool:
        return qid in self.answers

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def values(self) -> dict[str, Any]:
        """``{question_id: primitive value}`` — the answer without the metadata."""
        return {qid: a.value for qid, a in self.answers.items()}


def parse_decision_response(
    resp: Mapping[str, Any],
    *,
    fallback_model: str = "",
) -> tuple[str, dict[str, Answer]]:
    """Parse a raw ``/v1/systemone``-shaped body into ``(model, answers)``."""
    model = str(resp.get("model") or fallback_model)
    raw_answers = resp.get("answers") or {}
    if not isinstance(raw_answers, Mapping):
        raise ValueError(f"Expected an 'answers' object in the response, got {type(raw_answers).__name__}.")
    return model, {qid: parse_answer(qid, raw) for qid, raw in raw_answers.items()}


# ── Drivers ────────────────────────────────────────────────────────────────


class DecisionDriver(ABC):
    """Adapter base for typed-decision ("System One") providers.

    Subclasses implement :meth:`decide` and should populate ``self.last_usage``
    with a dict describing the most recent call::

        {
            "model_name": "provider/model",
            "questions": int,           # how many questions the call carried
            "input_tokens": int,
            "output_tokens": int,
            "total_tokens": int,
            "cost": float,              # USD; 0.0 for self-hosted models
            "pricing_unknown": bool,    # True if no rate entry was found
            "raw_response": dict,
        }

    ``last_usage`` is intentionally a flat dict so callers can persist it
    without coupling to a particular ``UsageEvent`` schema.
    """

    supports_async: bool = False

    callbacks: DriverCallbacks | None = None
    last_usage: dict[str, Any]

    def __init__(self) -> None:
        self.last_usage = {}

    @abstractmethod
    def decide(
        self,
        state: StateLike,
        questions: Mapping[str, QuestionLike],
        **options: Any,
    ) -> DecisionResponse:
        """Evaluate *state* against *questions*.

        Args:
            state: The content to evaluate — a string, or structured data
                (mapping/sequence) for tickets, records, chat logs, or the
                current state of an application.
            questions: ``{question_id: Question | dict}``.  Answers come back
                under the same ids.  Questions share the state but cannot see
                each other.
            **options: Provider-specific options (e.g. ``model``).

        Returns:
            A :class:`DecisionResponse`.
        """
        ...

    def ask(
        self,
        state: StateLike,
        question: QuestionLike,
        **options: Any,
    ) -> Answer:
        """Convenience wrapper for a single unnamed question.

        Prefer :meth:`decide` with a question map when you have more than one —
        every provider here evaluates the whole map against one shared reading
        of the state, so batching is cheaper than looping.
        """
        return self.decide(state, {"answer": question}, **options)["answer"]

    def _fire_callback(self, event: str, payload: dict[str, Any]) -> None:
        """Invoke a single callback, swallowing and logging any exception."""
        if self.callbacks is None:
            return
        cb = getattr(self.callbacks, event, None)
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            logger.exception("Callback %s raised an exception", event)


class AsyncDecisionDriver(ABC):
    """Async adapter base for decision providers.

    Mirrors :class:`DecisionDriver` with an awaitable :meth:`decide`.
    """

    supports_async: bool = True

    callbacks: DriverCallbacks | None = None
    last_usage: dict[str, Any]

    def __init__(self) -> None:
        self.last_usage = {}

    @abstractmethod
    async def decide(
        self,
        state: StateLike,
        questions: Mapping[str, QuestionLike],
        **options: Any,
    ) -> DecisionResponse:
        """Evaluate *state* against *questions* (async)."""
        ...

    async def ask(
        self,
        state: StateLike,
        question: QuestionLike,
        **options: Any,
    ) -> Answer:
        """Convenience wrapper for a single unnamed question (async)."""
        result = await self.decide(state, {"answer": question}, **options)
        return result["answer"]

    def _fire_callback(self, event: str, payload: dict[str, Any]) -> None:
        """Invoke a single callback, swallowing and logging any exception."""
        if self.callbacks is None:
            return
        cb = getattr(self.callbacks, event, None)
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            logger.exception("Callback %s raised an exception", event)
