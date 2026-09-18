"""The three fixed execution strategies.

Each one is an adapter over machinery Prompture already has, not a new engine:

``direct``
    :class:`~prompture.execution.strategies.direct.DirectStrategy` — one attempt
    plus validation, over :func:`prompture.extraction.core.ask_for_json`, with a
    separately bounded schema-repair pass.
``retrieve_verify``
    :class:`~prompture.execution.strategies.retrieve.RetrieveAndVerifyStrategy` —
    retrieval (any :class:`prompture.rag.Retriever`), a grounded typed answer,
    structural citation checking, and an optional claim-level check via
    :class:`prompture.eval.FaithfulnessEvaluator`.
``draft_critique``
    :class:`~prompture.execution.strategies.critique.DraftAndCritiqueStrategy` —
    the existing :class:`prompture.agents.review_loop.AsyncReviewLoop`, driven
    with a structured reviewer verdict.

All three return the same
:class:`~prompture.execution.types.ExecutionResult` envelope, aggregate usage
across every nested call, check budgets before starting more work, and terminate
for an explicit, inspectable reason.
"""

from __future__ import annotations

from .base import ExecutionStrategy, StrategyContext, StrategyError, run_sync
from .critique import DraftAndCritiqueStrategy, draft_and_critique
from .direct import DirectStrategy, direct
from .registry import (
    BUILTIN_STRATEGIES,
    get_strategy,
    list_strategies,
    register_strategy,
    unregister_strategy,
)
from .retrieve import RetrieveAndVerifyStrategy, retrieve_and_verify

__all__ = [
    "BUILTIN_STRATEGIES",
    "DirectStrategy",
    "DraftAndCritiqueStrategy",
    "ExecutionStrategy",
    "RetrieveAndVerifyStrategy",
    "StrategyContext",
    "StrategyError",
    "direct",
    "draft_and_critique",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "retrieve_and_verify",
    "run_sync",
    "unregister_strategy",
]
