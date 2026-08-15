"""Lipsync driver factory functions.

Provides high-level factory functions for instantiating lipsync drivers by
model string. Built-in driver registration is handled centrally by
``provider_descriptors.register_all_builtin_drivers()``.
"""

from typing import cast

from .lipsync_base import AsyncLipsyncDriver, LipsyncDriver
from .registry import (
    get_async_lipsync_driver_factory,
    get_lipsync_driver_factory,
)


def get_lipsync_driver_for_model(model_str: str) -> LipsyncDriver:
    """Instantiate a sync lipsync driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_lipsync_driver_factory(provider)
    return cast(LipsyncDriver, factory(model_id))


def get_async_lipsync_driver_for_model(model_str: str) -> AsyncLipsyncDriver:
    """Instantiate an async lipsync driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_lipsync_driver_factory(provider)
    return cast(AsyncLipsyncDriver, factory(model_id))
