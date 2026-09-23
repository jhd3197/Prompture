"""Async TypeSafe decision driver — the hosted Jev "System One" model."""

from __future__ import annotations

from .async_systemone_compatible_driver import AsyncSystemOneCompatibleDriver
from .typesafe_decision_driver import TypeSafeDecisionDriver


class AsyncTypeSafeDecisionDriver(AsyncSystemOneCompatibleDriver):
    """Async hosted TypeSafe Jev decision driver."""

    PROVIDER = TypeSafeDecisionDriver.PROVIDER
    DEFAULT_BASE_URL = TypeSafeDecisionDriver.DEFAULT_BASE_URL
    DEFAULT_MODEL = TypeSafeDecisionDriver.DEFAULT_MODEL
    ENV_KEY = TypeSafeDecisionDriver.ENV_KEY
    ENV_BASE_URL = TypeSafeDecisionDriver.ENV_BASE_URL
    REQUIRES_API_KEY = True

    KNOWN_MODELS: tuple[str, ...] = TypeSafeDecisionDriver.KNOWN_MODELS
