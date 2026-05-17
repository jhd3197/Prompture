"""Moderation driver factory functions.

Provides high-level factory functions for instantiating moderation drivers
by model string.  Built-in driver registration is handled centrally by
``provider_descriptors.register_all_builtin_drivers()``.

Usage:
    from prompture.drivers.moderation_registry import get_moderation_driver_for_model

    driver = get_moderation_driver_for_model("openai/omni-moderation-latest")
    result = driver.moderate("text to check")
"""

from typing import cast

# ── Public registry dicts (live views) ────────────────────────────────────
from .moderation_base import AsyncModerationDriver, ModerationDriver
from .registry import (
    _ASYNC_MODERATION_REGISTRY,
    _MODERATION_REGISTRY,
    get_async_moderation_driver_factory,
    get_moderation_driver_factory,
)

MODERATION_DRIVER_REGISTRY = _MODERATION_REGISTRY
ASYNC_MODERATION_DRIVER_REGISTRY = _ASYNC_MODERATION_REGISTRY


# ── Factory functions ─────────────────────────────────────────────────────


def get_moderation_driver_for_model(model_str: str) -> ModerationDriver:
    """Instantiate a sync moderation driver from a ``"provider/model"`` string.

    Args:
        model_str: e.g. ``"openai/omni-moderation-latest"`` or
            ``"mistral/mistral-moderation-latest"``.

    Returns:
        A configured moderation driver instance.
    """
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_moderation_driver_factory(provider)
    return cast(ModerationDriver, factory(model_id))


def get_async_moderation_driver_for_model(model_str: str) -> AsyncModerationDriver:
    """Instantiate an async moderation driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_moderation_driver_factory(provider)
    return cast(AsyncModerationDriver, factory(model_id))
