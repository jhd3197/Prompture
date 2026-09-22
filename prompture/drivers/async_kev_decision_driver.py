"""Async Kev decision driver — self-hosted Jev-compatible decision models."""

from __future__ import annotations

from .async_systemone_compatible_driver import AsyncSystemOneCompatibleDriver
from .kev_decision_driver import KevDecisionDriver


class AsyncKevDecisionDriver(AsyncSystemOneCompatibleDriver):
    """Async self-hosted Kev decision driver."""

    PROVIDER = KevDecisionDriver.PROVIDER
    DEFAULT_BASE_URL = KevDecisionDriver.DEFAULT_BASE_URL
    DEFAULT_MODEL = KevDecisionDriver.DEFAULT_MODEL
    ENV_KEY = KevDecisionDriver.ENV_KEY
    ENV_BASE_URL = KevDecisionDriver.ENV_BASE_URL
    REQUIRES_API_KEY = False

    KNOWN_MODELS: tuple[str, ...] = KevDecisionDriver.KNOWN_MODELS
