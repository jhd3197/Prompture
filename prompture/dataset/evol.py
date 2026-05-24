"""Evol-Instruct — LLM-driven evolution of Q&A pairs along quality axes.

Based on `WizardLM's Evol-Instruct <https://arxiv.org/abs/2304.12244>`_:
take an existing (question, answer) pair and ask an LLM to rewrite it
along one of three axes:

* **DEPTH** — make the question more specific, add a constraint, or
  require a more detailed answer.
* **BREADTH** — broaden the question's scope (related topic, sibling
  concept).
* **REASONING** — turn a recall question into one that demands
  multi-step inference.

Each evolution is one LLM call, producing one new pair. Compose with
the existing ``generate_qa_dataset`` output to multiply your training
set by 2-4× without re-running the expensive ingestion pipeline.

Quick start::

    from prompture.dataset.evol import EvolInstruct

    evol = EvolInstruct(model="openai/gpt-4o")
    evolved = evol.evolve_batch(qa_pairs, operators=["depth", "breadth"], rounds=1)
    # len(evolved) == len(qa_pairs) * 2
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..extraction.core import extract_with_model
from .schemas import QAPair

logger = logging.getLogger(__name__)


EvolOperator = Literal["depth", "breadth", "reasoning"]
ALL_OPERATORS: tuple[EvolOperator, ...] = ("depth", "breadth", "reasoning")


# ---------------------------------------------------------------------------
# Operator prompts. ``{question}`` and ``{answer}`` are substituted with
# the source pair. Each operator is a self-contained instruction so the
# LLM produces one new pair in a single call.
# ---------------------------------------------------------------------------

DEFAULT_OPERATOR_PROMPTS: dict[str, str] = {
    "depth": (
        "Rewrite the Q&A pair below to make the question MORE SPECIFIC. Add a "
        "constraint, edge case, or precision requirement that demands a more "
        "detailed answer. Do not invent facts beyond what the original answer "
        "implies.\n\n"
        "ORIGINAL QUESTION: {question}\n"
        "ORIGINAL ANSWER: {answer}\n\n"
        "Return ONE JSON object with `question` and `answer` keys. The new "
        "question must be self-contained; the new answer must remain accurate "
        "and grounded in the same source domain."
    ),
    "breadth": (
        "Rewrite the Q&A pair below to BROADEN its scope to a sibling concept "
        "or adjacent topic that someone studying the original would also "
        "benefit from. The new question must be a genuine sibling — not a "
        "rephrasing of the same fact.\n\n"
        "ORIGINAL QUESTION: {question}\n"
        "ORIGINAL ANSWER: {answer}\n\n"
        "Return ONE JSON object with `question` and `answer` keys. The new "
        "question/answer should fit the same domain but cover different "
        "subject matter."
    ),
    "reasoning": (
        "Rewrite the Q&A pair below into a MULTI-STEP REASONING question. The "
        "new question should require the responder to chain at least two "
        "pieces of information or perform a calculation — not just recall a "
        "single fact.\n\n"
        "ORIGINAL QUESTION: {question}\n"
        "ORIGINAL ANSWER: {answer}\n\n"
        "Return ONE JSON object with `question` and `answer` keys. The new "
        "answer should show the reasoning steps explicitly."
    ),
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class EvolResult:
    """Outcome of an :meth:`EvolInstruct.evolve_batch` call.

    Attributes:
        evolved: List of newly-generated pairs (does NOT include the originals).
        failures: Count of source pairs where every operator failed
            (LLM error / unparseable response / empty content). Useful
            for monitoring evolution quality.
        meta: Aggregated token / cost meta across all LLM calls.
    """

    evolved: list[QAPair] = field(default_factory=list)
    failures: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvolInstruct
# ---------------------------------------------------------------------------


def _parse_pair(text: str) -> QAPair | None:
    """Pull a single ``{"question": ..., "answer": ...}`` object out of LLM text.

    Tolerant of fenced JSON, prose-embedded JSON, and extra keys.
    Returns ``None`` when no usable pair can be recovered.
    """
    text = (text or "").strip()
    if not text:
        return None

    candidates: list[dict[str, Any]] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, dict):
                candidates.append(parsed)
        except json.JSONDecodeError:
            pass

    # Embedded JSON in prose — first balanced ``{...}`` substring.
    if not candidates:
        start = text.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start : i + 1])
                            if isinstance(parsed, dict):
                                candidates.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        break

    for obj in candidates:
        q = str(obj.get("question") or obj.get("instruction") or "").strip()
        a = str(obj.get("answer") or obj.get("output") or "").strip()
        if q and a:
            return QAPair(question=q, answer=a)
    return None


class EvolInstruct:
    """Evolve Q&A pairs along Evol-Instruct's depth / breadth / reasoning axes.

    Args:
        model: Model string passed to :func:`extract_with_model` — same
            ``"provider/model"`` form used everywhere else.
        operator_prompts: Override the operator prompt templates. Each
            template must contain ``{question}`` and ``{answer}``.
            Defaults to :data:`DEFAULT_OPERATOR_PROMPTS`.
        options: Extra options forwarded to ``extract_with_model``.
            Bumping ``temperature`` to ~0.7 produces more diverse
            evolutions; ``0.3`` keeps them tight to the source.
        seed: Optional seed for the per-pair operator chooser when a
            single operator argument is omitted from :meth:`evolve_pair`.
    """

    def __init__(
        self,
        *,
        model: str,
        operator_prompts: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        prompts = operator_prompts if operator_prompts is not None else DEFAULT_OPERATOR_PROMPTS
        for op in ALL_OPERATORS:
            if op not in prompts:
                raise ValueError(f"operator_prompts missing required key {op!r}.")
            tpl = prompts[op]
            if "{question}" not in tpl or "{answer}" not in tpl:
                raise ValueError(
                    f"operator_prompts[{op!r}] must contain '{{question}}' and '{{answer}}' placeholders."
                )
        self.model = model
        self.operator_prompts = dict(prompts)
        self.options = dict(options) if options is not None else {"temperature": 0.7, "max_tokens": 512}
        self._rng = random.Random(seed)  # nosec B311 - non-cryptographic dataset sampling

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve_pair(self, pair: QAPair, operator: EvolOperator | None = None) -> tuple[QAPair | None, dict[str, Any]]:
        """Run a single operator against one pair.

        When ``operator`` is ``None`` an operator is chosen uniformly at
        random from :data:`ALL_OPERATORS` using the configured seed.
        Returns ``(evolved_pair_or_None, meta)``. ``None`` indicates the
        LLM call failed or produced unparseable text.
        """
        op = operator or self._rng.choice(ALL_OPERATORS)
        prompt = self.operator_prompts[op].format(question=pair.question, answer=pair.answer)
        meta: dict[str, Any] = {}
        try:
            result = extract_with_model(
                QAPair,
                prompt,
                model_name=self.model,
                options=self.options,
            )
        except Exception as exc:
            logger.warning("EvolInstruct %s evolution failed: %s", op, exc)
            return None, meta

        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(key), (int, float)):
                    meta[key] = meta.get(key, 0) + int(usage[key])
            if isinstance(usage.get("cost"), (int, float)):
                meta["cost"] = float(meta.get("cost", 0.0)) + float(usage["cost"])

        # Prefer the validated Pydantic model the extractor returns.
        evolved: QAPair | None = result.get("model") if isinstance(result, dict) else None
        if not isinstance(evolved, QAPair):
            # Fall back to JSON parsing of the raw string in case
            # extract_with_model normalised an unexpected shape.
            evolved = _parse_pair(result.get("json_string", "") if isinstance(result, dict) else "")
        if evolved is None:
            return None, meta
        # Drop trivial "evolutions" that are identical to the source.
        if evolved.question.strip().casefold() == pair.question.strip().casefold():
            logger.debug("EvolInstruct %s produced an identical question; dropping.", op)
            return None, meta
        return evolved, meta

    def evolve_batch(
        self,
        pairs: Sequence[QAPair],
        *,
        operators: list[EvolOperator] | None = None,
        rounds: int = 1,
    ) -> EvolResult:
        """Evolve every input pair through each requested operator, ``rounds`` times.

        Output count = ``len(pairs) * len(operators) * rounds`` minus
        any failures. The originals are NOT included — concatenate the
        result with the input set if you want both.

        Args:
            pairs: Source pairs.
            operators: Which axes to apply. ``None`` (default) runs all
                three. Specify a subset to control cost.
            rounds: How many sequential evolution passes per operator.
                Round 2+ evolves the round-1 output, so it intensifies
                whichever axis you picked. Defaults to 1.
        """
        ops = operators or list(ALL_OPERATORS)
        for op in ops:
            if op not in ALL_OPERATORS:
                raise ValueError(f"Unknown operator {op!r}. Choose from {ALL_OPERATORS}.")

        result = EvolResult()
        meta: dict[str, Any] = {"calls": 0}

        for source in pairs:
            had_any_success = False
            for op in ops:
                current = source
                for _ in range(rounds):
                    evolved, call_meta = self.evolve_pair(current, op)
                    meta["calls"] += 1
                    if call_meta:
                        for k, v in call_meta.items():
                            meta[k] = (meta.get(k) or 0) + v if isinstance(v, (int, float)) else v
                    if evolved is None:
                        break
                    result.evolved.append(evolved)
                    had_any_success = True
                    current = evolved  # later rounds compound on the latest
            if not had_any_success:
                result.failures += 1

        result.meta = meta
        return result


__all__ = [
    "ALL_OPERATORS",
    "DEFAULT_OPERATOR_PROMPTS",
    "EvolInstruct",
    "EvolOperator",
    "EvolResult",
]
