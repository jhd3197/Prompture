"""LLM-as-judge evaluation primitives.

Four orthogonal evaluators that all share the LLM driver protocol used
throughout Prompture::

    driver.generate(prompt, options) -> {"text": str, "meta": {...}}

* :class:`LLMJudge` — score one output against a rubric of criteria.
* :class:`PairwiseJudge` — pick the better of two outputs with
  position-bias correction.
* :class:`SelfConsistencyEvaluator` — sample N independent answers and
  return the majority vote plus an agreement ratio.
* :class:`FaithfulnessEvaluator` — decompose a RAG answer into atomic
  claims and check each one against the retrieved evidence.

All evaluators return ``Result`` dataclasses with a small, stable
shape so downstream tooling can render reports without parsing free
text.
"""

from __future__ import annotations

from .base import (
    EvalDriver,
    EvalError,
    FaithfulnessResult,
    JudgeResult,
    PairwiseResult,
    SelfConsistencyResult,
)
from .faithfulness import FaithfulnessEvaluator
from .judge import LLMJudge
from .pairwise import PairwiseJudge
from .self_consistency import SelfConsistencyEvaluator

__all__ = [
    "EvalDriver",
    "EvalError",
    "FaithfulnessEvaluator",
    "FaithfulnessResult",
    "JudgeResult",
    "LLMJudge",
    "PairwiseJudge",
    "PairwiseResult",
    "SelfConsistencyEvaluator",
    "SelfConsistencyResult",
]
