"""Driver-level refusal evaluation.

:class:`RefusalEvaluator` runs a prompt set through a Prompture driver
(sync or async) and reports the refusal rate, category breakdown, and
keeps a small sample of representative responses.  Useful for:

* Comparing alignment across providers in a fallback chain.
* Validating decensored / abliterated models — measure refusal rate
  before and after the modification with the same prompt set.
* CI-time regression guards: fail if refusal rate on a benign prompt
  set drifts above a threshold (i.e. the model has become over-aligned).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .detector import RefusalCategory, RefusalDetector, RefusalResult

logger = logging.getLogger("prompture.refusal")


@dataclass
class RefusalReport:
    """Aggregate result from :meth:`RefusalEvaluator.evaluate_*`.

    Attributes:
        total: Number of responses scored.
        refusals: Number flagged as refusals.
        refusal_rate: ``refusals / total``, or ``0.0`` when total is 0.
        by_category: Count of decisive categories across all
            responses.  Includes non-refusing categories
            (``deflection``, ``safety_disclaimer``) when they fired.
        confidence_avg: Mean confidence across all flagged refusals.
        samples: Up to ``max_samples`` ``(prompt, response, result)``
            triples, sampled per refusing category.  Use this to
            eyeball false positives / negatives.
        duration_seconds: Wall-clock time of the evaluation run.
        model: Identifier of the driver evaluated, when known.
    """

    total: int = 0
    refusals: int = 0
    refusal_rate: float = 0.0
    by_category: dict[str, int] = field(default_factory=dict)
    confidence_avg: float = 0.0
    samples: list[tuple[str, str, RefusalResult]] = field(default_factory=list)
    duration_seconds: float = 0.0
    model: str | None = None

    # Convenience for display / debugging.
    def __str__(self) -> str:  # pragma: no cover — cosmetic
        head = f"RefusalReport(model={self.model!r}, total={self.total}, "
        head += f"refusals={self.refusals}, rate={self.refusal_rate:.1%})"
        if self.by_category:
            head += "\n  by_category:"
            for k, v in sorted(self.by_category.items(), key=lambda kv: -kv[1]):
                head += f"\n    {k}: {v}"
        return head


class RefusalEvaluator:
    """Run a refusal benchmark against responses or a live driver.

    Args:
        detector: A pre-built :class:`RefusalDetector`.  Defaults to
            one with English + Spanish markers.
        max_samples: Cap on the number of ``(prompt, response, result)``
            triples kept on the :class:`RefusalReport`.  When more
            refusals are found, samples are kept round-robin across
            categories so each refusal category is represented.
            Default 10.
    """

    def __init__(
        self,
        detector: RefusalDetector | None = None,
        *,
        max_samples: int = 10,
    ) -> None:
        self.detector = detector or RefusalDetector()
        self.max_samples = max(0, max_samples)

    # ------------------------------------------------------------------
    # Score pre-existing responses
    # ------------------------------------------------------------------

    def evaluate_responses(
        self,
        responses: Sequence[str],
        *,
        prompts: Sequence[str] | None = None,
        model: str | None = None,
    ) -> RefusalReport:
        """Score a pre-computed list of *responses*.

        Args:
            responses: The model outputs to classify.
            prompts: Optional matching prompt list — recorded on each
                sample for context.
            model: Optional model identifier recorded on the report.
        """
        start = time.monotonic()
        results = self.detector.detect_many(responses)
        report = self._build_report(
            prompts=prompts,
            responses=responses,
            results=results,
            model=model,
        )
        report.duration_seconds = time.monotonic() - start
        return report

    # ------------------------------------------------------------------
    # Drive a real provider
    # ------------------------------------------------------------------

    def evaluate_driver(
        self,
        model: Any,
        prompts: Sequence[str],
        *,
        options: dict[str, Any] | None = None,
    ) -> RefusalReport:
        """Run *prompts* through *model* and score the responses.

        Args:
            model: A model string (``"provider/name"``) or a driver
                instance exposing ``generate(prompt, options)``.
            prompts: The prompt set to send.
            options: Driver options passed through to each call.
        """
        from ..drivers import get_driver_for_model

        driver = model if hasattr(model, "generate") else get_driver_for_model(model)
        model_id = model if isinstance(model, str) else getattr(driver, "model", None)

        start = time.monotonic()
        responses: list[str] = []
        for prompt in prompts:
            try:
                result = driver.generate(prompt, options or {})
                responses.append(_extract_text(result))
            except Exception as exc:  # pragma: no cover — protect long runs
                logger.warning("Driver call failed for prompt: %s", exc)
                responses.append("")  # Empty response → counts as refusal by default

        results = self.detector.detect_many(responses)
        report = self._build_report(
            prompts=prompts,
            responses=responses,
            results=results,
            model=model_id,
        )
        report.duration_seconds = time.monotonic() - start
        return report

    async def evaluate_driver_async(
        self,
        model: Any,
        prompts: Sequence[str],
        *,
        options: dict[str, Any] | None = None,
        concurrency: int = 4,
    ) -> RefusalReport:
        """Async variant of :meth:`evaluate_driver` with bounded concurrency.

        Args:
            model: A model string or an async driver instance.
            prompts: The prompt set to send.
            options: Driver options.
            concurrency: Max number of in-flight requests.  Default 4.
        """
        from ..drivers import get_async_driver_for_model

        driver = model if hasattr(model, "generate") else get_async_driver_for_model(model)
        model_id = model if isinstance(model, str) else getattr(driver, "model", None)

        sem = asyncio.Semaphore(concurrency)

        async def _call(prompt: str) -> str:
            async with sem:
                try:
                    result = await driver.generate(prompt, options or {})
                    return _extract_text(result)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Async driver call failed: %s", exc)
                    return ""

        start = time.monotonic()
        responses = await asyncio.gather(*(_call(p) for p in prompts))
        results = self.detector.detect_many(responses)
        report = self._build_report(
            prompts=prompts,
            responses=responses,
            results=results,
            model=model_id,
        )
        report.duration_seconds = time.monotonic() - start
        return report

    # ------------------------------------------------------------------
    # Report assembly
    # ------------------------------------------------------------------

    def _build_report(
        self,
        *,
        prompts: Sequence[str] | None,
        responses: Sequence[str],
        results: Sequence[RefusalResult],
        model: str | None,
    ) -> RefusalReport:
        total = len(results)
        refusals = sum(1 for r in results if r.is_refusal)
        confidence_sum = sum(r.confidence for r in results if r.is_refusal)
        cat_counter: Counter[str] = Counter()
        for r in results:
            if r.category is not None:
                cat_counter[r.category.value] += 1

        samples = self._select_samples(
            prompts=prompts or [""] * total,
            responses=responses,
            results=results,
        )

        return RefusalReport(
            total=total,
            refusals=refusals,
            refusal_rate=(refusals / total) if total else 0.0,
            by_category=dict(cat_counter),
            confidence_avg=(confidence_sum / refusals) if refusals else 0.0,
            samples=samples,
            duration_seconds=0.0,  # set by caller
            model=model,
        )

    def _select_samples(
        self,
        *,
        prompts: Sequence[str],
        responses: Sequence[str],
        results: Sequence[RefusalResult],
    ) -> list[tuple[str, str, RefusalResult]]:
        if self.max_samples <= 0:
            return []
        # Group flagged responses by category so the sample set covers
        # the spectrum instead of repeating one type.
        by_cat: dict[RefusalCategory, list[tuple[str, str, RefusalResult]]] = {}
        for prompt, response, result in zip(prompts, responses, results):
            if not result.is_refusal or result.category is None:
                continue
            by_cat.setdefault(result.category, []).append((prompt, response, result))

        # Round-robin take across categories until we hit max_samples.
        samples: list[tuple[str, str, RefusalResult]] = []
        iters = {cat: iter(items) for cat, items in by_cat.items() if items}
        while iters and len(samples) < self.max_samples:
            for cat in list(iters):
                try:
                    samples.append(next(iters[cat]))
                except StopIteration:
                    iters.pop(cat)
                if len(samples) >= self.max_samples:
                    break
        return samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(driver_result: Any) -> str:
    """Pull the response text out of a driver's response dict.

    Prompture drivers return ``{"text": "...", "meta": {...}}``.  We
    keep this defensive in case a driver hands back a bare string.
    """
    if isinstance(driver_result, str):
        return driver_result
    if isinstance(driver_result, dict):
        return str(driver_result.get("text", ""))
    return str(driver_result)


# Surface RefusalCategory so callers can `from prompture.refusal import
# evaluator; evaluator.RefusalCategory` without a separate import line.
__all__ = ["RefusalCategory", "RefusalEvaluator", "RefusalReport"]
