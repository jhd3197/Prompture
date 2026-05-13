"""Base classes for rerank drivers.

Rerank is a first-class modality alongside LLM, embedding, STT, TTS, image-gen,
and video-gen.  Rerank providers take a *query* and a list of candidate
*documents* and return them re-ordered by relevance.

Usage::

    from prompture.drivers.rerank_registry import get_rerank_driver_for_model

    driver = get_rerank_driver_for_model("cohere/rerank-v3.5")
    results = driver.rerank(
        query="What is the capital of France?",
        documents=["Paris is the capital of France.", "Berlin is in Germany."],
        top_n=2,
        return_documents=True,
    )
    for r in results:
        print(r.index, r.relevance_score, r.document)

    # Usage / cost metadata for the most recent call is exposed via ``last_usage``.
    print(driver.last_usage)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.rerank_driver")


_RATES_DIR = Path(__file__).resolve().parent.parent / "infra" / "rates"
_rerank_pricing_cache: dict[str, dict[str, dict[str, float]]] = {}


def _load_rerank_pricing(provider: str) -> dict[str, dict[str, float]]:
    """Return ``{model_id: {"per_search_unit": ..., "per_million_tokens": ...}}``
    parsed from the provider's rates JSON.  Results are cached.

    Rerank pricing models don't fit the standard input/output rate shape used
    by ``LocalKBPricingSource``, so we read the raw JSON directly here.
    """
    if provider in _rerank_pricing_cache:
        return _rerank_pricing_cache[provider]
    out: dict[str, dict[str, float]] = {}
    path = _RATES_DIR / f"{provider}.json"
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse rerank pricing from %s", path)
            raw = {}
        for model_id, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("modality") != "rerank":
                continue
            cost = entry.get("cost") or {}
            if not isinstance(cost, dict):
                continue
            slim: dict[str, float] = {}
            for key in ("per_search_unit", "per_million_tokens", "input"):
                val = cost.get(key)
                if isinstance(val, (int, float)):
                    slim[key] = float(val)
            if slim:
                out[model_id] = slim
    _rerank_pricing_cache[provider] = out
    return out


def calculate_rerank_cost(
    provider: str,
    model: str,
    *,
    search_units: int = 0,
    total_tokens: int = 0,
) -> tuple[float, bool]:
    """Calculate USD cost for a rerank call.

    Returns ``(cost, pricing_unknown)``.  ``pricing_unknown`` is ``True`` when
    no pricing entry was found.
    """
    pricing = _load_rerank_pricing(provider).get(model)
    if not pricing:
        return 0.0, True
    cost = 0.0
    if "per_search_unit" in pricing and search_units > 0:
        cost += search_units * pricing["per_search_unit"]
    per_million = pricing.get("per_million_tokens", pricing.get("input", 0.0))
    if per_million and total_tokens > 0:
        cost += (total_tokens / 1_000_000) * per_million
    return round(cost, 6), False


@dataclass
class RerankResult:
    """A single rerank result row.

    Attributes:
        index: Position of the document in the *original* input list.
        relevance_score: Provider-defined relevance score (typically 0.0–1.0;
            higher = more relevant).  Comparable only within a single result list.
        document: The original document text.  Populated only when the caller
            passes ``return_documents=True``; otherwise ``None``.
    """

    index: int
    relevance_score: float
    document: str | None = None


class RerankDriver(ABC):
    """Adapter base for rerank providers.

    Subclasses implement :meth:`rerank` and should populate ``self.last_usage``
    with a dict describing the most recent call::

        {
            "model_name": "provider/model",
            "search_units": int,        # provider-specific billable units
            "total_tokens": int,        # when applicable
            "cost": float,              # USD; 0.0 when pricing unknown
            "pricing_unknown": bool,    # True if no rate entry was found
            "raw_response": dict,
        }

    ``last_usage`` is intentionally a flat dict so callers can persist it
    without coupling to a particular ``UsageEvent`` schema.
    """

    supports_async: bool = False

    callbacks: DriverCallbacks | None = None
    last_usage: dict[str, Any]

    def __init__(self) -> None:
        self.last_usage = {}

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_documents: bool = False,
        **options: Any,
    ) -> list[RerankResult]:
        """Re-rank *documents* by relevance to *query*.

        Args:
            query: The search query string.
            documents: The candidate documents to score.
            top_n: If set, only return the top-N most relevant results.
                When ``None``, all documents are returned (re-ordered).
            return_documents: When ``True``, populate ``RerankResult.document``
                with the original text.  When ``False``, leave it as ``None``
                to save bandwidth.
            **options: Provider-specific options (e.g. ``max_chunks_per_doc``).

        Returns:
            A list of :class:`RerankResult`, sorted by descending
            ``relevance_score``.
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


class AsyncRerankDriver(ABC):
    """Async adapter base for rerank providers.

    Mirrors :class:`RerankDriver` with an awaitable :meth:`rerank`.
    """

    supports_async: bool = True

    callbacks: DriverCallbacks | None = None
    last_usage: dict[str, Any]

    def __init__(self) -> None:
        self.last_usage = {}

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        return_documents: bool = False,
        **options: Any,
    ) -> list[RerankResult]:
        """Re-rank *documents* by relevance to *query* (async)."""
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
