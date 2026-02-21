"""Budget enforcement for Prompture agents and conversations.

Provides cost/token budget tracking, pre-flight estimation, and
policy-based enforcement (hard stop, warn, degrade to cheaper model).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..exceptions import BudgetExceededError

logger = logging.getLogger("prompture.budget")

# Threshold at which ``degrade`` policy proactively switches models.
_DEGRADE_THRESHOLD = 0.80


class BudgetPolicy(enum.Enum):
    """How to react when a budget limit is approached or exceeded."""

    hard_stop = "hard_stop"
    warn_and_continue = "warn_and_continue"
    degrade = "degrade"


@dataclass(frozen=True)
class BudgetState:
    """Snapshot of budget consumption vs. limits."""

    cost_used: float = 0.0
    tokens_used: int = 0
    max_cost: float | None = None
    max_tokens: int | None = None

    # -- computed helpers ------------------------------------------------

    @property
    def exceeded(self) -> bool:
        """True when any hard limit has been reached."""
        if self.max_cost is not None and self.cost_used >= self.max_cost:
            return True
        if self.max_tokens is not None and self.tokens_used >= self.max_tokens:
            return True
        return False

    @property
    def cost_remaining(self) -> float | None:
        if self.max_cost is None:
            return None
        return max(0.0, self.max_cost - self.cost_used)

    @property
    def tokens_remaining(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def cost_fraction(self) -> float | None:
        """Fraction of cost budget consumed (0.0 – 1.0+)."""
        if self.max_cost is None or self.max_cost == 0:
            return None
        return self.cost_used / self.max_cost


# -- estimation ----------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count for *text*.

    Uses tiktoken ``cl100k_base`` when available, otherwise falls back to
    a ~4 chars/token heuristic.
    """
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate USD cost for a call using cached model rates.

    Returns ``0.0`` when rates are unavailable.
    """
    from .model_rates import get_model_rates

    provider, _, model_id = model.partition("/")
    if not model_id:
        return 0.0

    rates = get_model_rates(provider, model_id)
    if rates is None:
        return 0.0

    input_rate = rates.get("input", 0.0)  # per 1M tokens
    output_rate = rates.get("output", 0.0)  # per 1M tokens

    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


# -- enforcement ----------------------------------------------------------


def enforce_budget(
    budget_state: BudgetState,
    policy: BudgetPolicy,
    *,
    fallback_models: list[str] | None = None,
    current_model: str = "",
    on_model_fallback: Callable[[str, str, Any], None] | None = None,
) -> str | None:
    """Apply *policy* given *budget_state*.

    Returns:
        ``None`` when no action is needed (budget OK or warn-only).
        A new model string when ``degrade`` selects a fallback.

    Raises:
        BudgetExceededError: For ``hard_stop`` when the budget is exceeded.
    """
    cost_frac = budget_state.cost_fraction
    exceeded = budget_state.exceeded

    if policy is BudgetPolicy.hard_stop:
        if exceeded:
            raise BudgetExceededError(
                f"Budget exceeded (cost={budget_state.cost_used:.4f}, tokens={budget_state.tokens_used})",
                budget_state=budget_state,
            )
        return None

    if policy is BudgetPolicy.warn_and_continue:
        if exceeded:
            logger.warning(
                "Budget exceeded — continuing per warn_and_continue policy (cost=%.4f, tokens=%d)",
                budget_state.cost_used,
                budget_state.tokens_used,
            )
        return None

    if policy is BudgetPolicy.degrade:
        # Switch proactively at the threshold, or when already exceeded
        should_switch = exceeded or (cost_frac is not None and cost_frac >= _DEGRADE_THRESHOLD)
        if not should_switch:
            return None

        if not fallback_models:
            raise BudgetExceededError(
                f"Budget threshold reached but no fallback models configured "
                f"(cost={budget_state.cost_used:.4f}, tokens={budget_state.tokens_used})",
                budget_state=budget_state,
            )

        # Pick the first fallback that isn't the current model
        new_model: str | None = None
        for candidate in fallback_models:
            if candidate != current_model:
                new_model = candidate
                break

        if new_model is None:
            # All fallbacks are the same as current — nothing to switch to
            logger.warning(
                "Budget threshold reached but no different fallback available; continuing with %s",
                current_model,
            )
            return None

        logger.info(
            "Budget degrade: switching from %s to %s (cost_fraction=%.2f)",
            current_model,
            new_model,
            cost_frac or 0.0,
        )

        if on_model_fallback is not None:
            on_model_fallback(current_model, new_model, budget_state)

        return new_model

    return None  # pragma: no cover — unreachable with current enum
