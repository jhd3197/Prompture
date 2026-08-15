"""Music driver factory functions.

Provides high-level factory functions for instantiating music generation
drivers by model string. Built-in driver registration is handled centrally by
``provider_descriptors.register_all_builtin_drivers()``.
"""

from typing import cast

from .music_base import AsyncMusicGenDriver, MusicGenDriver
from .registry import (
    get_async_music_driver_factory,
    get_music_driver_factory,
)


def get_music_driver_for_model(model_str: str) -> MusicGenDriver:
    """Instantiate a sync music driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_music_driver_factory(provider)
    return cast(MusicGenDriver, factory(model_id))


def get_async_music_driver_for_model(model_str: str) -> AsyncMusicGenDriver:
    """Instantiate an async music driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_music_driver_factory(provider)
    return cast(AsyncMusicGenDriver, factory(model_id))
