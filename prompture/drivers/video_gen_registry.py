"""Video generation driver factory functions.

Provides high-level factory functions for instantiating video generation
drivers by model string. Built-in driver registration is handled centrally by
``provider_descriptors.register_all_builtin_drivers()``.
"""

from typing import cast

from .async_video_gen_base import AsyncVideoGenDriver
from .registry import (
    get_async_video_gen_driver_factory,
    get_video_gen_driver_factory,
)
from .video_gen_base import VideoGenDriver


def get_video_gen_driver_for_model(model_str: str) -> VideoGenDriver:
    """Instantiate a sync video generation driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_video_gen_driver_factory(provider)
    return cast(VideoGenDriver, factory(model_id))


def get_async_video_gen_driver_for_model(model_str: str) -> AsyncVideoGenDriver:
    """Instantiate an async video generation driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_video_gen_driver_factory(provider)
    return cast(AsyncVideoGenDriver, factory(model_id))
