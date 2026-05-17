"""Budget enforcement for Prompture agents and conversations.

Provides cost/token budget tracking, pre-flight estimation, and
policy-based enforcement (hard stop, warn, degrade to cheaper model).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any, Callable, Union

from ..exceptions import BudgetExceededError

logger = logging.getLogger("prompture.budget")

# Threshold at which ``degrade`` policy proactively switches models.
_DEGRADE_THRESHOLD = 0.80


class BudgetPolicy(enum.Enum):
    """How to react when a budget limit is approached or exceeded."""

    hard_stop = "hard_stop"
    warn_and_continue = "warn_and_continue"
    degrade = "degrade"


def resolve_budget_policy(value: BudgetPolicy | str | None) -> BudgetPolicy | None:
    """Coerce a string or BudgetPolicy enum to BudgetPolicy.

    Accepts enum members, their string values (``"hard_stop"``), or ``None``.
    """
    if value is None or isinstance(value, BudgetPolicy):
        return value
    if isinstance(value, str):
        try:
            return BudgetPolicy(value)
        except ValueError:
            valid = ", ".join(p.value for p in BudgetPolicy)
            raise ValueError(f"Invalid budget policy {value!r}. Valid: {valid}") from None
    raise TypeError(f"Expected BudgetPolicy, str, or None — got {type(value).__name__}")


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
        return bool(self.max_tokens is not None and self.tokens_used >= self.max_tokens)

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
    except Exception:
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


# -- pre-flight estimate from text ---------------------------------------


@dataclass(frozen=True)
class CostEstimate:
    """Forecast for a single LLM call.

    Attributes:
        model: The model string the estimate was computed for.
        input_tokens: Estimated input token count.
        output_tokens: Estimated output token count.
        total_tokens: Sum of input + output.
        input_cost: USD cost of the input portion.
        output_cost: USD cost of the output portion.
        total_cost: USD sum.
        rates_available: ``False`` when no pricing data was found for
            *model* — costs in that case are ``0.0`` and should not
            be treated as authoritative.
        currency: Always ``"USD"`` for now.
        token_counter: ``"tiktoken"`` when the precise encoder ran,
            otherwise ``"heuristic"`` (~4 chars/token fallback).
    """

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    rates_available: bool
    currency: str = "USD"
    token_counter: str = "heuristic"


def _tokens_for(text_or_count: Union[str, int]) -> tuple[int, str]:
    """Resolve a ``str | int`` to ``(token_count, counter_name)``."""
    if isinstance(text_or_count, int):
        return max(0, text_or_count), "exact"
    text = text_or_count or ""
    if not text:
        return 0, "exact"
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), "tiktoken"
    except Exception:
        return max(1, len(text) // 4), "heuristic"


def estimate_call_cost(
    model: str,
    prompt: Union[str, int],
    completion: Union[str, int, None] = None,
    *,
    expected_completion_tokens: int = 500,
) -> CostEstimate:
    """Forecast cost and token usage for a single LLM call.

    Accepts text or pre-counted tokens for both prompt and completion.
    When *completion* is omitted, ``expected_completion_tokens`` is
    used as a rough estimate of the response length.

    Args:
        model: Model string in ``"provider/model"`` form.
        prompt: Either the prompt text (counted with tiktoken when
            available, char-heuristic otherwise) or an already-counted
            token integer.
        completion: Either the expected completion text, an integer
            token count, or ``None`` to fall back to
            ``expected_completion_tokens``.
        expected_completion_tokens: Default completion-length guess
            when *completion* is ``None``.  Defaults to 500 — a
            mid-sized assistant reply.

    Returns:
        A :class:`CostEstimate` carrying token counts and USD cost.

    Example::

        from prompture import estimate_call_cost

        est = estimate_call_cost("openai/gpt-4o", prompt="…", completion=200)
        if est.total_cost > 0.10:
            raise RuntimeError("Too expensive — skipping.")
    """
    in_tokens, in_counter = _tokens_for(prompt)
    if completion is None:
        out_tokens, out_counter = expected_completion_tokens, "exact"
    else:
        out_tokens, out_counter = _tokens_for(completion)

    # Use whichever counter is least-precise to label the estimate.
    # (Local name avoids the literal ``token`` in the variable name —
    # bandit's B105 false-positives on it.)
    if "heuristic" in (in_counter, out_counter):
        counter_label = "heuristic"
    elif "tiktoken" in (in_counter, out_counter):
        counter_label = "tiktoken"
    else:
        counter_label = "exact"

    from .model_rates import get_model_rates

    provider, _, model_id = model.partition("/")
    rates = get_model_rates(provider, model_id) if model_id else None
    rates_available = bool(rates and (rates.get("input") or rates.get("output")))

    if rates_available:
        input_rate = rates.get("input", 0.0)
        output_rate = rates.get("output", 0.0)
        input_cost = (in_tokens * input_rate) / 1_000_000
        output_cost = (out_tokens * output_rate) / 1_000_000
    else:
        input_cost = 0.0
        output_cost = 0.0

    return CostEstimate(
        model=model,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        total_tokens=in_tokens + out_tokens,
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        total_cost=round(input_cost + output_cost, 6),
        rates_available=rates_available,
        token_counter=counter_label,
    )


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
