"""Async Laya decision driver — wraps the sync in-process driver with asyncio.to_thread."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .decision_base import AsyncDecisionDriver, DecisionResponse, QuestionLike, StateLike
from .laya_decision_driver import LayaDecisionDriver


class AsyncLayaDecisionDriver(AsyncDecisionDriver):
    """Async wrapper around :class:`LayaDecisionDriver`.

    Laya runs the weights in-process with no native async API, so we delegate
    to ``asyncio.to_thread()`` to avoid blocking the event loop.
    """

    supports_async = True

    PROVIDER = LayaDecisionDriver.PROVIDER
    DEFAULT_MODEL = LayaDecisionDriver.DEFAULT_MODEL
    KNOWN_MODELS: tuple[str, ...] = LayaDecisionDriver.KNOWN_MODELS

    def __init__(
        self,
        model: str | None = None,
        preload: bool = True,
        device: str | None = None,
    ):
        super().__init__()
        self.model = model or self.DEFAULT_MODEL
        self._sync_driver = LayaDecisionDriver(model=model, preload=preload, device=device)

    async def decide(
        self,
        state: StateLike,
        questions: Mapping[str, QuestionLike],
        **options: Any,
    ) -> DecisionResponse:
        result = await asyncio.to_thread(self._sync_driver.decide, state, questions, **options)
        self.last_usage = self._sync_driver.last_usage
        return result

    async def unload(self) -> None:
        """Release the resident checkpoints and free memory."""
        await asyncio.to_thread(self._sync_driver.unload)
