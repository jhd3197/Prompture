"""Decision driver factory functions.

Provides high-level factory functions for instantiating decision drivers
by model string.  Built-in driver registration is handled centrally by
``provider_descriptors.register_all_builtin_drivers()``.

Usage:
    from prompture.drivers.decision_registry import get_decision_driver_for_model

    driver = get_decision_driver_for_model("typesafe/jev-latest")
    result = driver.decide("Payouts have failed for 3 days.", {"urgent": Noul("Urgent?")})
"""

from typing import Any, cast

# ── Public registry dicts (live views) ────────────────────────────────────
from .decision_base import AsyncDecisionDriver, DecisionDriver
from .registry import (
    _ASYNC_DECISION_REGISTRY,
    _DECISION_REGISTRY,
    get_async_decision_driver_factory,
    get_decision_driver_factory,
)

DECISION_DRIVER_REGISTRY = _DECISION_REGISTRY
ASYNC_DECISION_DRIVER_REGISTRY = _ASYNC_DECISION_REGISTRY


# ── Factory functions ─────────────────────────────────────────────────────


def get_decision_driver_for_model(model_str: str, **options: Any) -> DecisionDriver:
    """Instantiate a sync decision driver from a ``"provider/model"`` string.

    Args:
        model_str: e.g. ``"typesafe/jev-latest"``, ``"kev/kev-4b"`` or
            ``"laya/router"``.
        **options: Constructor options for the driver, overriding settings,
            e.g. ``preload=False, device="cuda:1"`` for Laya.

    Returns:
        A configured decision driver instance.
    """
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_decision_driver_factory(provider)
    return cast(DecisionDriver, factory(model_id, **options))


def get_async_decision_driver_for_model(model_str: str, **options: Any) -> AsyncDecisionDriver:
    """Instantiate an async decision driver from a ``"provider/model"`` string.

    ``options`` are passed to the driver constructor, as in the sync factory.
    """
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_decision_driver_factory(provider)
    return cast(AsyncDecisionDriver, factory(model_id, **options))
