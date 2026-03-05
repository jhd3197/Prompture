"""Image generation driver factory functions.

Provides high-level factory functions for instantiating image gen drivers
by model string.  Built-in driver registration is handled centrally by
``provider_descriptors.register_all_builtin_drivers()``.

Usage:
    from prompture.drivers.img_gen_registry import get_img_gen_driver_for_model

    driver = get_img_gen_driver_for_model("openai/dall-e-3")
    result = driver.generate_image("a cat on a surfboard", {"size": "1024x1024"})
"""

from typing import cast

from .async_img_gen_base import AsyncImageGenDriver
from .img_gen_base import ImageGenDriver
from .registry import (
    get_async_img_gen_driver_factory,
    get_img_gen_driver_factory,
)

# ── Factory functions ─────────────────────────────────────────────────────


def get_img_gen_driver_for_model(model_str: str) -> ImageGenDriver:
    """Instantiate a sync image gen driver from a ``"provider/model"`` string.

    Args:
        model_str: e.g. ``"openai/dall-e-3"`` or ``"stability/stable-image-core"``.

    Returns:
        A configured image gen driver instance.
    """
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_img_gen_driver_factory(provider)
    return cast(ImageGenDriver, factory(model_id))


def get_async_img_gen_driver_for_model(model_str: str) -> AsyncImageGenDriver:
    """Instantiate an async image gen driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_img_gen_driver_factory(provider)
    return cast(AsyncImageGenDriver, factory(model_id))
