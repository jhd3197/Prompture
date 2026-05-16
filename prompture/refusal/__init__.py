"""Detect and measure LLM refusals across drivers.

Provides a fast, configurable refusal detector plus a driver-level
evaluator that reports refusal rate, category breakdown, and sample
responses.  Useful for:

* **Comparing alignment** across providers in your fallback chain.
* **Filtering or alerting** when an agent's response is a refusal.
* **Validating decensored / abliterated models** (e.g. those produced
  with `Heretic <https://github.com/p-e-w/heretic>`_) by measuring
  refusal rate before and after the modification.

The detector is a clean-room MIT implementation — it uses public
marker phrases and a position-weighted scoring heuristic, with no
external dependencies.

Quick start::

    from prompture.refusal import RefusalDetector

    detector = RefusalDetector()
    result = detector.detect("I'm sorry, but I cannot help with that.")
    print(result.is_refusal, result.confidence, result.category)

    # Measure a driver against a prompt list
    from prompture.refusal import RefusalEvaluator
    report = RefusalEvaluator().evaluate_driver(
        "openai/gpt-4o-mini",
        prompts=["...", "..."],
    )
    print(f"Refusal rate: {report.refusal_rate:.1%}")
"""

from .detector import (
    RefusalCategory,
    RefusalDetector,
    RefusalResult,
    is_refusal,
)
from .evaluator import RefusalEvaluator, RefusalReport
from .markers import DEFAULT_MARKERS

__all__ = [
    "DEFAULT_MARKERS",
    "RefusalCategory",
    "RefusalDetector",
    "RefusalEvaluator",
    "RefusalReport",
    "RefusalResult",
    "is_refusal",
]
