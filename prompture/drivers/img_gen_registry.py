"""Image generation driver registration and factory functions.

Registers built-in image gen drivers (OpenAI, Google, Stability, Grok) and provides
high-level factory functions for instantiating image gen drivers by model string.

Usage:
    from prompture.drivers.img_gen_registry import get_img_gen_driver_for_model

    driver = get_img_gen_driver_for_model("openai/dall-e-3")
    result = driver.generate_image("a cat on a surfboard", {"size": "1024x1024"})
"""

from ..infra.settings import settings
from .async_google_img_gen_driver import AsyncGoogleImageGenDriver
from .async_grok_img_gen_driver import AsyncGrokImageGenDriver
from .async_openai_img_gen_driver import AsyncOpenAIImageGenDriver
from .async_stability_img_gen_driver import AsyncStabilityImageGenDriver
from .google_img_gen_driver import GoogleImageGenDriver
from .grok_img_gen_driver import GrokImageGenDriver
from .openai_img_gen_driver import OpenAIImageGenDriver
from .registry import (
    get_async_img_gen_driver_factory,
    get_img_gen_driver_factory,
    register_async_img_gen_driver,
    register_img_gen_driver,
)
from .stability_img_gen_driver import StabilityImageGenDriver

# ── Register built-in OpenAI image gen drivers ────────────────────────────

register_img_gen_driver(
    "openai",
    lambda model=None: OpenAIImageGenDriver(
        api_key=settings.openai_api_key,
        model=model or "dall-e-3",
    ),
    overwrite=True,
)

register_async_img_gen_driver(
    "openai",
    lambda model=None: AsyncOpenAIImageGenDriver(
        api_key=settings.openai_api_key,
        model=model or "dall-e-3",
    ),
    overwrite=True,
)

# ── Register built-in Google image gen drivers ────────────────────────────

register_img_gen_driver(
    "google",
    lambda model=None: GoogleImageGenDriver(
        api_key=settings.google_api_key,
        model=model or "imagen-3.0-generate-002",
    ),
    overwrite=True,
)

register_async_img_gen_driver(
    "google",
    lambda model=None: AsyncGoogleImageGenDriver(
        api_key=settings.google_api_key,
        model=model or "imagen-3.0-generate-002",
    ),
    overwrite=True,
)

# ── Register built-in Stability AI image gen drivers ──────────────────────

_stability_api_key = getattr(settings, "stability_api_key", None)
_stability_endpoint = getattr(settings, "stability_endpoint", None)

register_img_gen_driver(
    "stability",
    lambda model=None: StabilityImageGenDriver(
        api_key=_stability_api_key,
        model=model or "stable-image-core",
        endpoint=_stability_endpoint,
    ),
    overwrite=True,
)

register_async_img_gen_driver(
    "stability",
    lambda model=None: AsyncStabilityImageGenDriver(
        api_key=_stability_api_key,
        model=model or "stable-image-core",
        endpoint=_stability_endpoint,
    ),
    overwrite=True,
)

# ── Register built-in Grok/xAI image gen drivers ─────────────────────────

register_img_gen_driver(
    "grok",
    lambda model=None: GrokImageGenDriver(
        api_key=settings.grok_api_key,
        model=model or "grok-2-image",
    ),
    overwrite=True,
)

register_async_img_gen_driver(
    "grok",
    lambda model=None: AsyncGrokImageGenDriver(
        api_key=settings.grok_api_key,
        model=model or "grok-2-image",
    ),
    overwrite=True,
)


# ── Factory functions ─────────────────────────────────────────────────────


def get_img_gen_driver_for_model(model_str: str):
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
    return factory(model_id)


def get_async_img_gen_driver_for_model(model_str: str):
    """Instantiate an async image gen driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_img_gen_driver_factory(provider)
    return factory(model_id)
