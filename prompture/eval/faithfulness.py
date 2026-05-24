"""RAG-answer faithfulness evaluator.

Given a generated RAG answer plus the retrieved evidence chunks,
this evaluator measures how well the answer is grounded in the
evidence. The pipeline is two LLM calls per evaluation:

1. **Decomposition** — break the answer into atomic claims (one
   verifiable fact per claim).
2. **Verification** — for each claim, ask the judge whether the
   evidence ``supports``, ``contradicts``, or is ``silent`` on it.

Score = ``supported / total_claims``. A score of 1.0 means every
claim was directly supported; a low score plus a non-empty
``contradicted`` list flags the most dangerous failure mode
(hallucinated facts that the evidence explicitly refutes).

Both prompts are short and use a forced-choice JSON shape so the
parsing is robust to weaker judges.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import EvalDriver, EvalError, FaithfulnessResult, _accumulate_meta

#: Default decomposition prompt. ``{answer}`` is substituted.
DEFAULT_DECOMPOSITION_PROMPT = """Break the ANSWER below into a list of atomic claims. Each claim must:
- contain exactly one verifiable fact
- stand on its own (no pronouns referring outside the claim)
- be written in the same tense and voice as the answer

ANSWER:
{answer}

Output a JSON array of strings, one claim per element, with no other text:
["claim 1", "claim 2", ...]
"""


#: Default verification prompt. ``{claim}`` and ``{evidence}`` are
#: substituted per claim.
DEFAULT_VERIFICATION_PROMPT = """You are checking whether a claim is grounded in retrieved evidence.

EVIDENCE:
{evidence}

CLAIM:
{claim}

Respond with exactly one of these three labels — nothing else, no punctuation:
SUPPORTED   (the evidence directly states or strongly implies the claim)
CONTRADICTED   (the evidence states the opposite)
SILENT   (the evidence neither supports nor contradicts the claim)
"""


_VERDICT_RE = re.compile(r"\b(SUPPORTED|CONTRADICTED|SILENT)\b", re.IGNORECASE)


def _parse_claims(text: str) -> list[str]:
    """Pull a list of claim strings out of the decomposition response.

    Accepts:
    - a JSON array (the documented happy path)
    - bullet/numbered lists when the model ignored the JSON instruction
    """
    text = (text or "").strip()
    if not text:
        return []

    # JSON happy path.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return [str(c).strip() for c in parsed if isinstance(c, (str, int, float)) and str(c).strip()]

    # Fenced JSON.
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if isinstance(c, (str, int, float)) and str(c).strip()]
        except json.JSONDecodeError:
            pass

    # Embedded JSON array.
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if isinstance(c, (str, int, float)) and str(c).strip()]
        except json.JSONDecodeError:
            pass

    # Bullet / numbered list fallback.
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"^\s*(?:\d+[.)]|[-*])\s+", "", line.strip())
        stripped = stripped.strip(" \"'")
        if stripped:
            lines.append(stripped)
    return lines


def _parse_verdict(text: str) -> str | None:
    """Extract the verdict label from the verification response."""
    m = _VERDICT_RE.search(text or "")
    if not m:
        return None
    return m.group(1).upper()


def _format_evidence(evidence: list[str]) -> str:
    """Format the evidence list as numbered passages for the judge."""
    if not evidence:
        return "(no evidence provided)"
    return "\n\n".join(f"[{i + 1}] {chunk.strip()}" for i, chunk in enumerate(evidence) if chunk.strip())


class FaithfulnessEvaluator:
    """Measure how well a RAG answer is grounded in retrieved evidence.

    Args:
        driver: LLM driver used for both decomposition and per-claim
            verification.
        decomposition_prompt: Template overriding
            :data:`DEFAULT_DECOMPOSITION_PROMPT`. Must include
            ``{answer}``.
        verification_prompt: Template overriding
            :data:`DEFAULT_VERIFICATION_PROMPT`. Must include ``{claim}``
            and ``{evidence}``.
        llm_options: Forwarded to ``driver.generate``. Defaults to
            ``{"temperature": 0.0, "max_tokens": 512}``.
        treat_silent_as: How to score ``SILENT`` verdicts. ``"unsupported"``
            (default) — silent claims count against the score; this is
            the conservative choice and matches RAGAS faithfulness.
            ``"neutral"`` — silent claims are excluded from the
            denominator entirely.

    Example::

        evaluator = FaithfulnessEvaluator(driver=my_driver)
        result = evaluator.evaluate(
            answer="Paris is the capital of France and home to the Eiffel Tower.",
            evidence=[
                "Paris is the capital city of France.",
                "Lyon is France's third-largest city.",
            ],
        )
        # FaithfulnessResult(
        #   score=0.5,
        #   supported=["Paris is the capital of France."],
        #   unsupported=["Paris is home to the Eiffel Tower."],
        #   contradicted=[],
        # )
    """

    def __init__(
        self,
        driver: EvalDriver,
        *,
        decomposition_prompt: str = DEFAULT_DECOMPOSITION_PROMPT,
        verification_prompt: str = DEFAULT_VERIFICATION_PROMPT,
        llm_options: dict[str, Any] | None = None,
        treat_silent_as: str = "unsupported",
    ) -> None:
        if "{answer}" not in decomposition_prompt:
            raise EvalError("decomposition_prompt must include {answer}.")
        for ph in ("{claim}", "{evidence}"):
            if ph not in verification_prompt:
                raise EvalError(f"verification_prompt must include {ph}.")
        if treat_silent_as not in {"unsupported", "neutral"}:
            raise EvalError(f"treat_silent_as must be 'unsupported' or 'neutral'; got {treat_silent_as!r}.")
        self.driver = driver
        self.decomposition_prompt = decomposition_prompt
        self.verification_prompt = verification_prompt
        self.llm_options = llm_options if llm_options is not None else {
            "temperature": 0.0,
            "max_tokens": 512,
        }
        self.treat_silent_as = treat_silent_as

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, answer: str, evidence: list[str]) -> FaithfulnessResult:
        """Score ``answer`` against ``evidence``.

        Decomposes the answer into atomic claims, verifies each
        against the evidence, and returns a :class:`FaithfulnessResult`
        with the supported / unsupported / contradicted partitions
        and the rolled-up score.

        Raises:
            EvalError: When the decomposition step produces zero
                usable claims (e.g. the LLM returned junk).
        """
        meta: dict[str, Any] = {}
        claims = self._decompose(answer, meta=meta)
        if not claims:
            raise EvalError("Decomposition produced no claims; cannot score faithfulness.")

        evidence_text = _format_evidence(evidence)

        supported: list[str] = []
        unsupported: list[str] = []
        contradicted: list[str] = []
        silent: list[str] = []

        for claim in claims:
            verdict = self._verify(claim, evidence_text, meta=meta)
            if verdict == "SUPPORTED":
                supported.append(claim)
            elif verdict == "CONTRADICTED":
                contradicted.append(claim)
                unsupported.append(claim)
            else:
                # SILENT or unparseable
                silent.append(claim)
                if self.treat_silent_as == "unsupported":
                    unsupported.append(claim)

        # Score denominator depends on treat_silent_as.
        if self.treat_silent_as == "neutral":
            denominator = len(supported) + len(unsupported)
        else:
            denominator = len(claims)
        score = (len(supported) / denominator) if denominator else 0.0

        return FaithfulnessResult(
            score=score,
            supported=supported,
            unsupported=unsupported,
            contradicted=contradicted,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _decompose(self, answer: str, *, meta: dict[str, Any]) -> list[str]:
        rendered = self.decomposition_prompt.format(answer=answer)
        try:
            response = self.driver.generate(rendered, dict(self.llm_options))
        except Exception as exc:
            raise EvalError(f"Decomposition driver call failed: {exc}") from exc
        if isinstance(response, dict):
            _accumulate_meta(meta, response.get("meta"))
            text = response.get("text") or ""
        else:
            text = str(response or "")
        return _parse_claims(text)

    def _verify(self, claim: str, evidence_text: str, *, meta: dict[str, Any]) -> str:
        rendered = self.verification_prompt.format(claim=claim, evidence=evidence_text)
        try:
            response = self.driver.generate(rendered, dict(self.llm_options))
        except Exception as exc:
            raise EvalError(f"Verification driver call failed: {exc}") from exc
        if isinstance(response, dict):
            _accumulate_meta(meta, response.get("meta"))
            text = response.get("text") or ""
        else:
            text = str(response or "")
        verdict = _parse_verdict(text)
        return verdict or "SILENT"
