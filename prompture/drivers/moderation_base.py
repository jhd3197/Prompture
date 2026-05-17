"""Base classes for moderation drivers.

Moderation is a first-class modality alongside LLM, embedding, STT, TTS,
image-gen, video-gen, and rerank.  Moderation providers take one or more text
inputs and return per-category flags + confidence scores indicating whether
the content violates the provider's content policy.

Usage::

    from prompture.drivers.moderation_registry import get_moderation_driver_for_model

    driver = get_moderation_driver_for_model("openai/omni-moderation-latest")
    result = driver.moderate("I will hurt someone")
    print(result.flagged, result.categories["violence"], result.category_scores)

    # Batch input → list of results
    results = driver.moderate(["benign text", "violent text"])

    # Usage / cost metadata for the most recent call is exposed via ``last_usage``.
    print(driver.last_usage)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.moderation_driver")


_RATES_DIR = Path(__file__).resolve().parent.parent / "infra" / "rates"
_moderation_pricing_cache: dict[str, dict[str, dict[str, float]]] = {}


def _load_moderation_pricing(provider: str) -> dict[str, dict[str, float]]:
    """Return ``{model_id: {"per_request": ..., "per_million_tokens": ...}}``
    parsed from the provider's rates JSON.  Results are cached.

    Moderation pricing models don't fit the standard input/output rate shape
    used by ``LocalKBPricingSource``, so we read the raw JSON directly here.
    """
    if provider in _moderation_pricing_cache:
        return _moderation_pricing_cache[provider]
    out: dict[str, dict[str, float]] = {}
    path = _RATES_DIR / f"{provider}.json"
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse moderation pricing from %s", path)
            raw = {}
        for model_id, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("modality") != "moderation":
                continue
            cost = entry.get("cost") or {}
            if not isinstance(cost, dict):
                continue
            slim: dict[str, float] = {}
            for key in ("per_request", "per_million_tokens", "input"):
                val = cost.get(key)
                if isinstance(val, (int, float)):
                    slim[key] = float(val)
            # Even a fully-zero entry (e.g. OpenAI moderation, which is free)
            # must be recorded so callers can distinguish "known free" from
            # "unknown pricing".  Use a sentinel ``_known`` marker.
            slim["_known"] = 1.0
            out[model_id] = slim
    _moderation_pricing_cache[provider] = out
    return out


def calculate_moderation_cost(
    provider: str,
    model: str,
    *,
    requests: int = 0,
    total_tokens: int = 0,
) -> tuple[float, bool]:
    """Calculate USD cost for a moderation call.

    Returns ``(cost, pricing_unknown)``.  ``pricing_unknown`` is ``True`` when
    no pricing entry was found in the KB.  A zero-cost entry (e.g. OpenAI
    moderation is free) is still considered *known* — ``pricing_unknown`` is
    ``False`` and ``cost`` is ``0.0``.
    """
    pricing = _load_moderation_pricing(provider).get(model)
    if not pricing:
        return 0.0, True
    cost = 0.0
    if "per_request" in pricing and requests > 0:
        cost += requests * pricing["per_request"]
    per_million = pricing.get("per_million_tokens", pricing.get("input", 0.0))
    if per_million and total_tokens > 0:
        cost += (total_tokens / 1_000_000) * per_million
    return round(cost, 6), False


@dataclass
class ModerationResult:
    """A single moderation result.

    Attributes:
        flagged: ``True`` if any category triggered.
        categories: Per-category boolean flag (e.g. ``{"violence": True, ...}``).
        category_scores: Per-category confidence score in ``[0.0, 1.0]``.
            Higher = more confident the content falls into that category.
        raw_response: Optional raw provider response for the single item
            (provider-specific).
    """

    flagged: bool
    categories: dict[str, bool] = field(default_factory=dict)
    category_scores: dict[str, float] = field(default_factory=dict)
    raw_response: dict[str, Any] | None = None


class ModerationDriver(ABC):
    """Adapter base for content-moderation providers.

    Subclasses implement :meth:`moderate` and should populate
    ``self.last_usage`` with a dict describing the most recent call::

        {
            "model_name": "provider/model",
            "input_count": int,
            "total_tokens": int,
            "cost": float,
            "pricing_unknown": bool,
            "raw_response": dict,
        }
    """

    supports_async: bool = False

    callbacks: DriverCallbacks | None = None
    last_usage: dict[str, Any]

    def __init__(self) -> None:
        self.last_usage = {}

    @abstractmethod
    def moderate(
        self,
        input: str | list[str],
        **options: Any,
    ) -> ModerationResult | list[ModerationResult]:
        """Run content moderation on one or more text inputs.

        Args:
            input: Either a single string (returns a single
                :class:`ModerationResult`) or a list of strings (returns a
                ``list[ModerationResult]`` of the same length).
            **options: Provider-specific options (e.g. ``model=`` override).

        Returns:
            A single :class:`ModerationResult` when *input* is a string, or a
            list of results when *input* is a list.
        """
        ...

    def _fire_callback(self, event: str, payload: dict[str, Any]) -> None:
        """Invoke a single callback, swallowing and logging any exception."""
        if self.callbacks is None:
            return
        cb = getattr(self.callbacks, event, None)
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            logger.exception("Callback %s raised an exception", event)


class AsyncModerationDriver(ABC):
    """Async adapter base for moderation providers.

    Mirrors :class:`ModerationDriver` with an awaitable :meth:`moderate`.
    """

    supports_async: bool = True

    callbacks: DriverCallbacks | None = None
    last_usage: dict[str, Any]

    def __init__(self) -> None:
        self.last_usage = {}

    @abstractmethod
    async def moderate(
        self,
        input: str | list[str],
        **options: Any,
    ) -> ModerationResult | list[ModerationResult]:
        """Run content moderation on one or more text inputs (async)."""
        ...

    def _fire_callback(self, event: str, payload: dict[str, Any]) -> None:
        """Invoke a single callback, swallowing and logging any exception."""
        if self.callbacks is None:
            return
        cb = getattr(self.callbacks, event, None)
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            logger.exception("Callback %s raised an exception", event)
