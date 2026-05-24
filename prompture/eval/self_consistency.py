"""N-sample self-consistency evaluator.

Sample the same prompt ``n`` times at non-zero temperature and report:

* the **consensus** — the most-common normalised answer
* an **agreement ratio** — fraction of samples that match the consensus
* the full sample list + count distribution

Low agreement is a strong hallucination signal: the model is rolling
the dice rather than reading off a stable belief. Useful for any
extraction or short-form-answer task where you'd like a confidence
score alongside the answer.

Two modes:

* **Free-form** (default): pass an LLM driver. Each sample is one
  ``driver.generate`` call. Useful for short-answer questions.
* **Wrapped**: supply a callable ``sampler() -> str`` and the
  evaluator just runs your callable N times. Useful when each sample
  involves multiple LLM calls (e.g. a chain) and you want the
  outer-most answer compared.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from .base import EvalDriver, EvalError, SelfConsistencyResult, _accumulate_meta


def _default_normalize(text: str) -> str:
    """Cheap default normaliser: lowercased, whitespace-collapsed, trimmed.

    Removes trailing punctuation so ``"42."`` and ``"42"`` collapse
    to the same bucket. Keeps internal numeric tokens intact.
    """
    s = (text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(" .!?,;:'\"")
    return s


class SelfConsistencyEvaluator:
    """Sample N times, return majority vote + agreement.

    Args:
        driver: LLM driver to sample from. Ignored when ``sampler`` is
            supplied. One of ``driver`` or ``sampler`` must be set.
        n_samples: How many times to sample. Defaults to 5. Higher
            gives a tighter agreement estimate but scales linearly in
            cost. Use a temperature > 0 in ``llm_options`` so samples
            actually differ.
        temperature: Override ``llm_options['temperature']`` when set.
            Defaults to 0.7, which is high enough to surface real
            uncertainty without going off the rails. Ignored when a
            custom ``sampler`` is given.
        normalize: Function ``str -> str`` that collapses sample text
            into the bucket key used for the vote. Defaults to
            :func:`_default_normalize`. Pass ``lambda s: s.strip()`` to
            keep exact-match comparison.
        llm_options: Forwarded to ``driver.generate`` per sample.
        sampler: Callable producing one sample. When supplied, the
            evaluator just runs it N times and ``driver`` is unused.

    Example::

        evaluator = SelfConsistencyEvaluator(driver=my_driver, n_samples=5)
        result = evaluator.evaluate("What is 17 * 23?")
        if result.agreement >= 0.8:
            print(f"Confident answer: {result.consensus}")
        else:
            print(f"Low agreement ({result.agreement:.0%}) — counts: {result.counts}")
    """

    def __init__(
        self,
        driver: EvalDriver | None = None,
        *,
        n_samples: int = 5,
        temperature: float = 0.7,
        normalize: Callable[[str], str] = _default_normalize,
        llm_options: dict[str, Any] | None = None,
        sampler: Callable[[], str] | None = None,
    ) -> None:
        if n_samples < 1:
            raise EvalError(f"n_samples must be >= 1; got {n_samples}.")
        if driver is None and sampler is None:
            raise EvalError("SelfConsistencyEvaluator requires either driver= or sampler=.")
        self.driver = driver
        self.n_samples = n_samples
        self.temperature = temperature
        self.normalize = normalize
        self.sampler = sampler
        base_opts = dict(llm_options or {})
        # Default temperature only when caller didn't pin one.
        base_opts.setdefault("temperature", temperature)
        base_opts.setdefault("max_tokens", 256)
        self.llm_options = base_opts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, prompt: str | None = None) -> SelfConsistencyResult:
        """Run the sampler N times and return the majority vote.

        ``prompt`` is required when using the driver path; ignored
        (but accepted for API symmetry) when ``sampler`` is set.

        Raises:
            EvalError: When zero samples produced usable text.
        """
        if self.sampler is None:
            if not prompt or not prompt.strip():
                raise EvalError("prompt is required when using driver-based sampling.")
            sampler = self._driver_sampler(prompt)
        else:
            sampler = self.sampler

        meta: dict[str, Any] = {}
        normalised: list[str] = []

        for _ in range(self.n_samples):
            text, per_call_meta = self._invoke_sampler(sampler)
            if per_call_meta:
                _accumulate_meta(meta, per_call_meta)
            normalised_text = self.normalize(text)
            if normalised_text:
                normalised.append(normalised_text)

        if not normalised:
            raise EvalError("All self-consistency samples returned empty text.")

        counter = Counter(normalised)
        consensus, top_count = counter.most_common(1)[0]
        agreement = top_count / len(normalised)

        return SelfConsistencyResult(
            consensus=consensus,
            agreement=agreement,
            samples=normalised,
            counts=dict(counter),
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _driver_sampler(self, prompt: str) -> Callable[[], tuple[str, dict[str, Any] | None]]:
        """Build a sampler closure that calls ``driver.generate(prompt, ...)``.

        Returns a callable that, on each invocation, runs one driver
        call and returns ``(text, meta_or_none)``.
        """
        if self.driver is None:  # narrowing for type checkers
            raise EvalError("driver is required for the driver-based sampler.")

        driver = self.driver
        options = self.llm_options

        def _sample() -> tuple[str, dict[str, Any] | None]:
            response = driver.generate(prompt, dict(options))
            if isinstance(response, dict):
                return str(response.get("text") or ""), response.get("meta")
            return str(response or ""), None

        return _sample

    def _invoke_sampler(self, sampler: Callable[..., Any]) -> tuple[str, dict[str, Any] | None]:
        """Run a sampler and normalise its return shape to ``(text, meta_or_none)``."""
        result = sampler()
        if isinstance(result, tuple) and len(result) == 2:
            text, meta = result
            return str(text or ""), meta if isinstance(meta, dict) else None
        if isinstance(result, dict):
            return str(result.get("text") or ""), result.get("meta")
        return str(result or ""), None
