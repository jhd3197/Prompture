"""Shared driver for any endpoint speaking the ``/v1/systemone`` wire format.

TypeSafe published this shape for Jev, and the open implementations adopted it
verbatim — Kev serves a byte-identical endpoint from a local process.  So the
same way :mod:`openai_compatible_driver` covers every OpenAI-shaped endpoint,
this module covers every System One-shaped one: hosted, self-hosted, or
whatever ships next.

Subclass and set the class attributes, or instantiate directly with an explicit
``base_url``::

    driver = SystemOneCompatibleDriver(
        base_url="http://127.0.0.1:8009",
        api_key="local",
        model="kev-latest",
    )
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Mapping
from typing import Any

import requests

from .decision_base import (
    DecisionDriver,
    DecisionResponse,
    QuestionLike,
    StateLike,
    calculate_decision_cost,
    normalize_questions,
    parse_decision_response,
)

logger = logging.getLogger(__name__)

#: Path appended to a bare host, so callers can pass either form.
SYSTEMONE_PATH = "/v1/systemone"


def build_systemone_url(base_url: str) -> str:
    """Normalize *base_url* to a full ``/v1/systemone`` URL.

    Accepts ``https://api.typesafe.ai``, ``https://api.typesafe.ai/``, or the
    full endpoint already spelled out.
    """
    url = base_url.rstrip("/")
    if url.endswith(SYSTEMONE_PATH):
        return url
    return url + SYSTEMONE_PATH


class SystemOneTransportMixin:
    """Config, payload building and response parsing shared by sync + async.

    Only the HTTP call itself differs between the two, so everything else
    lives here and both drivers stay honest about the same wire format.
    """

    #: Provider name used for pricing lookups and ``model_name`` strings.
    PROVIDER: str = "systemone"
    DEFAULT_BASE_URL: str = ""
    DEFAULT_MODEL: str = ""
    #: Environment variable consulted when no ``api_key`` is passed.
    ENV_KEY: str = ""
    #: Environment variable consulted when no ``base_url`` is passed.
    ENV_BASE_URL: str = ""
    #: Hosted providers reject unauthenticated calls; local servers do not care.
    REQUIRES_API_KEY: bool = True

    last_usage: dict[str, Any]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        super().__init__()
        self.api_key = api_key or (os.getenv(self.ENV_KEY) if self.ENV_KEY else None)
        if self.REQUIRES_API_KEY and not self.api_key:
            raise ValueError(f"{self.PROVIDER} API key not found. Set the {self.ENV_KEY} env var.")
        resolved_base = (
            base_url or (os.getenv(self.ENV_BASE_URL) if self.ENV_BASE_URL else None) or self.DEFAULT_BASE_URL
        )
        if not resolved_base:
            raise ValueError(f"No base URL for {self.PROVIDER}. Pass base_url= or set the {self.ENV_BASE_URL} env var.")
        self.base_url = resolved_base
        self.url = build_systemone_url(resolved_base)
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _build_payload(
        self,
        state: StateLike,
        questions: Mapping[str, QuestionLike],
        options: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Return ``(payload, model)`` for a decide call."""
        model = options.pop("model", self.model)
        payload: dict[str, Any] = {
            "state": state,
            "model": model,
            "questions": normalize_questions(questions),
        }
        payload.update(options)
        return payload, model

    def _finalize(
        self,
        resp: dict[str, Any],
        model: str,
        question_count: int,
    ) -> DecisionResponse:
        """Parse a raw response body and record usage."""
        answered_model, answers = parse_decision_response(resp, fallback_model=model)

        usage = resp.get("usage", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)

        # Bill against the alias the caller asked for, falling back to the
        # versioned id the server reports when the alias has no rate entry.
        cost, pricing_unknown = calculate_decision_cost(
            self.PROVIDER, model, input_tokens=input_tokens, output_tokens=output_tokens
        )
        if pricing_unknown and answered_model and answered_model != model:
            cost, pricing_unknown = calculate_decision_cost(
                self.PROVIDER,
                answered_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        self.last_usage = {
            "model_name": f"{self.PROVIDER}/{model}",
            "questions": question_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "pricing_unknown": pricing_unknown,
            "raw_response": resp,
        }
        return DecisionResponse(
            model=answered_model,
            answers=answers,
            usage=dict(self.last_usage),
            raw_response=resp,
        )


class SystemOneCompatibleDriver(SystemOneTransportMixin, DecisionDriver):
    """Decision driver for ``POST {base_url}/v1/systemone`` endpoints."""

    supports_async = False

    def decide(
        self,
        state: StateLike,
        questions: Mapping[str, QuestionLike],
        **options: Any,
    ) -> DecisionResponse:
        payload, model = self._build_payload(state, questions, options)
        self._fire_callback("on_request", {"provider": self.PROVIDER, "model": model, "payload": payload})

        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            body = ""
            if e.response is not None:
                with contextlib.suppress(Exception):
                    body = e.response.text
            error_msg = f"{self.PROVIDER} decision API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            self._fire_callback("on_error", {"provider": self.PROVIDER, "model": model, "error": error_msg})
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            error_msg = f"{self.PROVIDER} decision API request failed: {e!s}"
            self._fire_callback("on_error", {"provider": self.PROVIDER, "model": model, "error": error_msg})
            raise RuntimeError(error_msg) from e

        result = self._finalize(resp, model, len(payload["questions"]))
        self._fire_callback("on_response", {"provider": self.PROVIDER, "model": model, "usage": self.last_usage})
        return result
