"""Async driver for any endpoint speaking the ``/v1/systemone`` wire format.

Mirrors :class:`~prompture.drivers.systemone_compatible_driver.SystemOneCompatibleDriver`;
config, payload building and response parsing come from the shared
:class:`SystemOneTransportMixin`, so only the HTTP call differs.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping
from typing import Any

import httpx

from .decision_base import AsyncDecisionDriver, DecisionResponse, QuestionLike, StateLike
from .systemone_compatible_driver import SystemOneTransportMixin

logger = logging.getLogger(__name__)


class AsyncSystemOneCompatibleDriver(SystemOneTransportMixin, AsyncDecisionDriver):
    """Async decision driver for ``POST {base_url}/v1/systemone`` endpoints."""

    supports_async = True

    async def decide(
        self,
        state: StateLike,
        questions: Mapping[str, QuestionLike],
        **options: Any,
    ) -> DecisionResponse:
        payload, model = self._build_payload(state, questions, options)
        self._fire_callback("on_request", {"provider": self.PROVIDER, "model": model, "payload": payload})

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.url, headers=self.headers, json=payload)
                response.raise_for_status()
                resp = response.json()
        except httpx.HTTPStatusError as e:
            body = ""
            with contextlib.suppress(Exception):
                body = e.response.text
            error_msg = f"{self.PROVIDER} decision API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            self._fire_callback("on_error", {"provider": self.PROVIDER, "model": model, "error": error_msg})
            raise RuntimeError(error_msg) from e
        except httpx.HTTPError as e:
            error_msg = f"{self.PROVIDER} decision API request failed: {e!s}"
            self._fire_callback("on_error", {"provider": self.PROVIDER, "model": model, "error": error_msg})
            raise RuntimeError(error_msg) from e

        result = self._finalize(resp, model, len(payload["questions"]))
        self._fire_callback("on_response", {"provider": self.PROVIDER, "model": model, "usage": self.last_usage})
        return result
