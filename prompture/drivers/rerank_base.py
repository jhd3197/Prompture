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

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.rerank_driver")


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
