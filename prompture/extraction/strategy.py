"""Structured output strategy selection for extraction.

Defines the three extraction strategies and the auto-selection logic
that picks the best approach based on driver/provider capabilities.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class StructuredOutputStrategy(str, Enum):
    """How Prompture obtains structured JSON from an LLM.

    The three strategies form a reliability cascade — each successive
    strategy works with less capable models:

    - **PROVIDER_NATIVE**: Use the provider's built-in JSON mode or
      JSON schema enforcement.  Most reliable when available.
    - **TOOL_CALL**: Encode the schema as a tool/function definition
      and extract JSON from the tool call arguments.  Works with models
      that support function calling but lack native JSON mode.
    - **PROMPTED_REPAIR**: Prompt the model for JSON and repair
      malformed output via AI cleanup.  Universal fallback — works
      with any model, including GPT-3 class.
    """

    PROVIDER_NATIVE = "provider_native"
    TOOL_CALL = "tool_call"
    PROMPTED_REPAIR = "prompted_repair"


def auto_select_strategy(
    model_str: str,
    *,
    driver: Any | None = None,
) -> StructuredOutputStrategy:
    """Pick the best strategy for *model_str* based on capabilities.

    Resolution order matches :func:`ProviderCapabilities.best_strategy`:
    provider_native > tool_call > prompted_repair.
    """
    from ..infra.capabilities import get_capabilities

    caps = get_capabilities(model_str, driver=driver)
    name = caps.best_strategy()
    return StructuredOutputStrategy(name)


def resolve_strategy(
    strategy: str | StructuredOutputStrategy | None,
    model_str: str,
    *,
    driver: Any | None = None,
) -> StructuredOutputStrategy:
    """Normalize a user-supplied strategy value.

    Accepts ``"auto"`` (or ``None``), a strategy name string, or an
    enum member.  Returns a concrete :class:`StructuredOutputStrategy`.
    """
    if strategy is None or strategy == "auto":
        return auto_select_strategy(model_str, driver=driver)
    if isinstance(strategy, StructuredOutputStrategy):
        return strategy
    return StructuredOutputStrategy(strategy)
