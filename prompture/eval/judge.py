"""Single-output rubric-based LLM judge.

``LLMJudge`` asks a strong LLM to score one candidate output against
a rubric of criteria, on a 1..N integer scale, with a free-text
justification. The prompt is short, deterministic (``temperature=0``
by default), and parses cleanly into a Pydantic model so callers
never have to massage free text.

This is the workhorse of the eval package — it underpins prompt-A/B
testing, regression detection, and any "did the output get worse?"
loop. Use :class:`PairwiseJudge` instead when the question is "which
of these two is better" rather than "how good is this one".
"""

from __future__ import annotations

import json
import re
from typing import Any

from .._internal.json_encoder import PromptureJSONEncoder
from .base import EvalDriver, EvalError, JudgeResult, _accumulate_meta

#: Default rubric prompt. ``{input}``, ``{output}``, ``{reference}``,
#: ``{criteria_block}``, and ``{scale}`` are substituted. ``{reference}``
#: is replaced with an empty string when no reference was supplied —
#: that's why the template wraps it in a labelled section that gracefully
#: degenerates when empty.
DEFAULT_RUBRIC_PROMPT = """You are an impartial evaluator scoring an LLM output.

Score the OUTPUT on every criterion below using an integer from 1 to {scale}, where 1 is the worst and {scale} is the best. Then assign an overall integer score on the same 1..{scale} range. Finally, write a one-paragraph justification (2-4 sentences).

CRITERIA:
{criteria_block}

INPUT:
{input}

OUTPUT TO EVALUATE:
{output}
{reference}

Respond with a single JSON object that matches this exact shape (no markdown, no preamble):
{response_shape}

Use double quotes for every string. Per-criterion scores must use the criterion name verbatim as the key.
"""


#: Substituted into the prompt only when a reference is supplied.
_REFERENCE_BLOCK_TEMPLATE = "\nREFERENCE ANSWER (for comparison, not strict equality):\n{reference}\n"


def _criteria_block(criteria: list[str]) -> str:
    """Render a list of criteria as a bulleted block.

    Single-line criteria become ``- name``. A criterion that contains
    a colon is split into ``name: description`` so authors can attach
    rubric details inline (e.g. ``"factuality: claims must match the
    reference"``).
    """
    lines: list[str] = []
    for c in criteria:
        c = c.strip()
        if not c:
            continue
        if ":" in c:
            name, desc = c.split(":", 1)
            lines.append(f"- {name.strip()}: {desc.strip()}")
        else:
            lines.append(f"- {c}")
    return "\n".join(lines) if lines else "- overall quality"


def _criterion_keys(criteria: list[str]) -> list[str]:
    """Pull out just the canonical criterion names (text before ``:``)."""
    keys: list[str] = []
    for c in criteria:
        c = c.strip()
        if not c:
            continue
        keys.append(c.split(":", 1)[0].strip())
    return keys


def _response_shape(criteria: list[str], scale: int) -> str:
    """Build the JSON shape hint we embed in the prompt itself.

    Inlining the expected shape (rather than relying solely on
    ``json_mode``) helps weaker models produce parseable output even
    when the driver doesn't support strict JSON mode.
    """
    keys = _criterion_keys(criteria) or ["overall_quality"]
    per_criterion: dict[str, str] = {k: f"<1..{scale} integer>" for k in keys}
    shape = {
        "score": f"<1..{scale} integer>",
        "per_criterion": per_criterion,
        "reasoning": "<one paragraph string>",
    }
    return json.dumps(shape, indent=2, cls=PromptureJSONEncoder)


class LLMJudge:
    """Rubric-based single-output judge.

    Args:
        driver: Any sync LLM driver matching :class:`EvalDriver`. The
            best results come from a capable model (``gpt-4o``,
            ``claude-sonnet-4-6``, etc.) — judge calls are a small
            fraction of total cost in most evaluation runs.
        criteria: List of named criteria the judge scores. Each entry
            may be ``"name"`` or ``"name: description"``. Defaults to
            ``["overall_quality"]``.
        scale: Maximum integer score (inclusive). The minimum is
            always 1. Defaults to 5.
        pass_threshold: When set, every :class:`JudgeResult` carries a
            boolean ``passed`` field == ``score >= pass_threshold``.
        prompt: Template overriding :data:`DEFAULT_RUBRIC_PROMPT`. Must
            include ``{input}``, ``{output}``, ``{reference}``,
            ``{criteria_block}``, ``{scale}``, and ``{response_shape}``.
        llm_options: Options forwarded to ``driver.generate``. Defaults
            to ``{"temperature": 0.0, "max_tokens": 400}``.

    Example::

        from prompture.eval import LLMJudge

        judge = LLMJudge(
            driver=my_driver,
            criteria=["factuality: claims must match the reference",
                      "clarity",
                      "brevity"],
            scale=10,
            pass_threshold=7,
        )
        result = judge.evaluate(
            input_text="What is the capital of France?",
            output="Paris is the capital of France.",
            reference="Paris",
        )
        print(result.score, result.per_criterion, result.passed)
    """

    def __init__(
        self,
        driver: EvalDriver,
        *,
        criteria: list[str] | None = None,
        scale: int = 5,
        pass_threshold: int | None = None,
        prompt: str = DEFAULT_RUBRIC_PROMPT,
        llm_options: dict[str, Any] | None = None,
    ) -> None:
        criteria_list = list(criteria) if criteria is not None else ["overall_quality"]
        if not criteria_list:
            raise EvalError("LLMJudge requires at least one criterion.")
        if scale < 2:
            raise EvalError(f"scale must be >= 2; got {scale}.")
        required_placeholders = ("{input}", "{output}", "{reference}", "{criteria_block}", "{scale}", "{response_shape}")
        for placeholder in required_placeholders:
            if placeholder not in prompt:
                raise EvalError(f"Prompt template is missing required placeholder {placeholder}.")

        self.driver = driver
        self.criteria = criteria_list
        self.scale = scale
        self.pass_threshold = pass_threshold
        self.prompt = prompt
        self.llm_options = llm_options if llm_options is not None else {
            "temperature": 0.0,
            "max_tokens": 400,
        }
        self._criterion_keys = _criterion_keys(self.criteria)
        self._criteria_block = _criteria_block(self.criteria)
        self._response_shape = _response_shape(self.criteria, self.scale)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        input_text: str,
        output: str,
        reference: str | None = None,
    ) -> JudgeResult:
        """Score one ``output`` against the rubric.

        ``reference`` is optional — when supplied it's shown to the
        judge with a "for comparison, not strict equality" hint so
        the judge can detect factuality issues without penalising
        cosmetic differences.

        Raises:
            EvalError: When the LLM call fails or the response can't
                be parsed into the expected shape even after the
                downstream extraction's own retry.
        """
        rendered = self._render_prompt(input_text, output, reference)
        try:
            response = self.driver.generate(rendered, dict(self.llm_options))
        except Exception as exc:
            raise EvalError(f"Judge driver call failed: {exc}") from exc

        text = (response.get("text") if isinstance(response, dict) else "") or ""
        parsed = self._parse_judge_response(text)

        score = self._clamp_score(parsed.get("score"))
        reasoning = str(parsed.get("reasoning") or "").strip()
        raw_per = parsed.get("per_criterion") or {}
        per_criterion: dict[str, int] = {}
        if isinstance(raw_per, dict):
            for key in self._criterion_keys:
                val = raw_per.get(key)
                if val is None:
                    # Some models lowercase / normalise keys — try case-insensitive.
                    val = next(
                        (raw_per[k] for k in raw_per if isinstance(k, str) and k.lower() == key.lower()),
                        None,
                    )
                clamped = self._clamp_score(val)
                if clamped is not None:
                    per_criterion[key] = clamped

        passed: bool | None = None
        if self.pass_threshold is not None and score is not None:
            passed = score >= self.pass_threshold

        meta: dict[str, Any] = {}
        if isinstance(response, dict):
            _accumulate_meta(meta, response.get("meta"))

        if score is None:
            raise EvalError(
                f"Judge response did not include a usable 'score' field. "
                f"Got: {text[:160]!r}"
            )

        return JudgeResult(
            score=score,
            reasoning=reasoning,
            per_criterion=per_criterion,
            passed=passed,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _render_prompt(self, input_text: str, output: str, reference: str | None) -> str:
        reference_block = _REFERENCE_BLOCK_TEMPLATE.format(reference=reference) if reference else ""
        return self.prompt.format(
            input=input_text,
            output=output,
            reference=reference_block,
            criteria_block=self._criteria_block,
            scale=self.scale,
            response_shape=self._response_shape,
        )

    def _clamp_score(self, value: Any) -> int | None:
        """Coerce ``value`` to an int in ``[1, self.scale]``.

        Accepts ints, floats, strings like ``"4"`` or ``"4/5"``.
        Returns ``None`` when ``value`` can't be coerced.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return None  # avoid the int(True)=1 trap
        if isinstance(value, (int, float)):
            ival = int(value)
            return max(1, min(self.scale, ival))
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            # Handle "4/5" style — take the numerator.
            num_match = re.match(r"^-?\d+", stripped)
            if num_match:
                try:
                    return max(1, min(self.scale, int(num_match.group())))
                except ValueError:
                    return None
        return None

    def _parse_judge_response(self, text: str) -> dict[str, Any]:
        """Pull a JSON object out of the judge's response text.

        Handles three common shapes:

        * raw JSON: ``{"score": 4, ...}``
        * fenced JSON: ``\\`\\`\\`json\\n{...}\\n\\`\\`\\```
        * JSON embedded in prose: we extract the first ``{...}`` span.
        """
        text = text.strip()
        if not text:
            return {}

        # Fast path: whole response is JSON.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Strip ```json fences if present.
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fence_match:
            try:
                parsed = json.loads(fence_match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Last resort: extract the first balanced ``{...}`` substring.
        start = text.find("{")
        if start == -1:
            return {}
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
                    candidate = text[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        return {}
        return {}
