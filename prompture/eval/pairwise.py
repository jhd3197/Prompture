"""Pairwise A/B preference judge with position-bias correction.

LLM judges have a well-documented positional bias — given two candidates,
they're more likely to pick the first one regardless of quality. The
fix is to ask twice with the candidates swapped and combine the votes:

* both passes agree → that's the winner with confidence 1.0
* passes disagree → the LLM is genuinely uncertain → ``"tie"``,
  confidence 0.5
* with ``swap_positions=False`` (cheaper, less accurate) we trust a
  single pass and report confidence 1.0

Use this when the question is "which of these two is better"; use
:class:`LLMJudge` for "how good is this one".
"""

from __future__ import annotations

import re
from typing import Any

from .base import EvalDriver, EvalError, PairwiseResult, _accumulate_meta

#: Default pairwise prompt. ``{input}``, ``{a}``, ``{b}`` are
#: substituted. The instruction explicitly forbids "tie" because at
#: the per-pass level we want a forced choice; the wrapper aggregates
#: disagreement into "tie" itself.
DEFAULT_PAIRWISE_PROMPT = """You are choosing the better of two LLM responses to the same input.

INPUT:
{input}

RESPONSE A:
{a}

RESPONSE B:
{b}

Pick exactly one — A or B — that better answers the input. You must choose; saying both are equal is not allowed. Briefly justify your choice in 1-2 sentences.

Respond on two lines, no markdown:
CHOICE: A
REASON: <one or two sentences>
"""

_CHOICE_RE = re.compile(r"\bCHOICE\s*[:=]\s*([AB])\b", re.IGNORECASE)


def _parse_choice(text: str) -> tuple[str | None, str]:
    """Extract ``("A" | "B" | None, reason)`` from a judge response.

    Tolerant about formatting — accepts ``CHOICE: A``, ``Choice = B``,
    ``a`` / ``b`` standalone, or the word ``Response A`` / ``Response B``
    if no ``CHOICE`` line is present.
    """
    text = (text or "").strip()
    if not text:
        return None, ""

    m = _CHOICE_RE.search(text)
    if m:
        choice = m.group(1).upper()
        reason_match = re.search(r"\bREASON\s*[:=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else ""
        return choice, reason

    # Fall back: look for "Response A" / "Response B" first occurrence.
    fallback = re.search(r"\bRESPONSE\s+([AB])\b", text, flags=re.IGNORECASE)
    if fallback:
        return fallback.group(1).upper(), text

    # Last resort: a single isolated A / B character on the first line.
    first_line = text.splitlines()[0].strip()
    if first_line.upper() in {"A", "B"}:
        return first_line.upper(), "\n".join(text.splitlines()[1:]).strip()

    return None, text


class PairwiseJudge:
    """A/B preference judge with optional position-bias correction.

    Args:
        driver: LLM driver matching :class:`EvalDriver`.
        swap_positions: When True (default), runs two judge passes per
            ``compare`` call — once with (A, B) and once with (B, A) —
            and combines votes to neutralise position bias. When False,
            only one pass is run (half the cost, less accurate).
        prompt: Template overriding :data:`DEFAULT_PAIRWISE_PROMPT`.
            Must contain ``{input}``, ``{a}``, ``{b}``.
        llm_options: Forwarded to ``driver.generate``. Defaults to
            ``{"temperature": 0.0, "max_tokens": 200}``.

    Example::

        judge = PairwiseJudge(driver=my_driver)
        result = judge.compare(
            input_text="Summarise quantum entanglement in one sentence.",
            output_a="Two particles share a state.",
            output_b="When two particles become entangled, ...",
        )
        # PairwiseResult(winner='B', confidence=1.0, votes={...})
    """

    def __init__(
        self,
        driver: EvalDriver,
        *,
        swap_positions: bool = True,
        prompt: str = DEFAULT_PAIRWISE_PROMPT,
        llm_options: dict[str, Any] | None = None,
    ) -> None:
        for placeholder in ("{input}", "{a}", "{b}"):
            if placeholder not in prompt:
                raise EvalError(f"Pairwise prompt is missing required placeholder {placeholder}.")
        self.driver = driver
        self.swap_positions = swap_positions
        self.prompt = prompt
        self.llm_options = llm_options if llm_options is not None else {
            "temperature": 0.0,
            "max_tokens": 200,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        input_text: str,
        output_a: str,
        output_b: str,
    ) -> PairwiseResult:
        """Decide which of ``output_a`` / ``output_b`` is better.

        When ``swap_positions=True`` (default), runs the judge twice
        with the candidates in opposite orders so positional bias
        cancels. The aggregated result is ``"A"`` or ``"B"`` when both
        passes agree, ``"tie"`` otherwise.
        """
        votes: dict[str, str] = {}
        reasoning: dict[str, str] = {}
        meta: dict[str, Any] = {}

        # First pass: A presented first.
        choice_ab, reason_ab = self._judge_once(input_text, output_a, output_b, meta=meta)
        if choice_ab is not None:
            votes["A_first"] = choice_ab
            reasoning["A_first"] = reason_ab

        if not self.swap_positions:
            return self._build_result_single(votes, reasoning, meta)

        # Second pass: B presented first; remap "A"->B, "B"->A so the
        # ``winner`` field always refers to the originals.
        choice_ba_raw, reason_ba = self._judge_once(input_text, output_b, output_a, meta=meta)
        if choice_ba_raw is not None:
            remapped = {"A": "B", "B": "A"}.get(choice_ba_raw, choice_ba_raw)
            votes["B_first"] = remapped
            reasoning["B_first"] = reason_ba

        return self._build_result_swapped(votes, reasoning, meta)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _judge_once(
        self,
        input_text: str,
        a_content: str,
        b_content: str,
        *,
        meta: dict[str, Any],
    ) -> tuple[str | None, str]:
        rendered = self.prompt.format(input=input_text, a=a_content, b=b_content)
        try:
            response = self.driver.generate(rendered, dict(self.llm_options))
        except Exception as exc:
            raise EvalError(f"Pairwise driver call failed: {exc}") from exc
        if isinstance(response, dict):
            _accumulate_meta(meta, response.get("meta"))
            text = response.get("text") or ""
        else:
            text = str(response or "")
        return _parse_choice(text)

    def _build_result_single(
        self,
        votes: dict[str, str],
        reasoning: dict[str, str],
        meta: dict[str, Any],
    ) -> PairwiseResult:
        if not votes:
            raise EvalError("Pairwise judge produced no parseable choice.")
        winner = next(iter(votes.values()))
        return PairwiseResult(
            winner=winner,
            confidence=1.0,
            votes=votes,
            reasoning=reasoning,
            meta=meta,
        )

    def _build_result_swapped(
        self,
        votes: dict[str, str],
        reasoning: dict[str, str],
        meta: dict[str, Any],
    ) -> PairwiseResult:
        # Tally the two votes (each one already remapped to the original-position frame).
        if not votes:
            raise EvalError("Pairwise judge produced no parseable choices on either pass.")

        counts: dict[str, int] = {"A": 0, "B": 0}
        for choice in votes.values():
            if choice in counts:
                counts[choice] += 1

        a, b = counts["A"], counts["B"]
        total = a + b
        if total == 0:
            raise EvalError("Pairwise judge returned only invalid choices.")

        if a > b:
            winner = "A"
            confidence = a / total
        elif b > a:
            winner = "B"
            confidence = b / total
        else:
            winner = "tie"
            confidence = 0.5

        return PairwiseResult(
            winner=winner,
            confidence=confidence,
            votes=votes,
            reasoning=reasoning,
            meta=meta,
        )
