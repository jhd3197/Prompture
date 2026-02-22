"""Audio driver registration and factory functions.

Registers built-in STT/TTS drivers (OpenAI, ElevenLabs) and provides
high-level factory functions for instantiating audio drivers by model string.

Usage:
    from prompture.drivers.audio_registry import get_stt_driver_for_model

    stt = get_stt_driver_for_model("openai/whisper-1")
    result = stt.transcribe(audio_bytes, {})
"""

from ..infra.settings import settings
from .async_elevenlabs_stt_driver import AsyncElevenLabsSTTDriver
from .async_elevenlabs_tts_driver import AsyncElevenLabsTTSDriver
from .async_openai_stt_driver import AsyncOpenAISTTDriver
from .async_openai_tts_driver import AsyncOpenAITTSDriver
from .async_stt_base import AsyncSTTDriver
from .async_tts_base import AsyncTTSDriver
from .elevenlabs_stt_driver import ElevenLabsSTTDriver
from .elevenlabs_tts_driver import ElevenLabsTTSDriver
from .openai_stt_driver import OpenAISTTDriver
from .openai_tts_driver import OpenAITTSDriver
from .registry import (
    get_async_stt_driver_factory,
    get_async_tts_driver_factory,
    get_stt_driver_factory,
    get_tts_driver_factory,
    register_async_stt_driver,
    register_async_tts_driver,
    register_stt_driver,
    register_tts_driver,
)
from .stt_base import STTDriver
from .tts_base import TTSDriver

# ── Register built-in OpenAI audio drivers ────────────────────────────────

register_stt_driver(
    "openai",
    lambda model=None: OpenAISTTDriver(  # type: ignore[misc]
        api_key=settings.openai_api_key,
        model=model or "whisper-1",
    ),
    overwrite=True,
)

register_async_stt_driver(
    "openai",
    lambda model=None: AsyncOpenAISTTDriver(  # type: ignore[misc]
        api_key=settings.openai_api_key,
        model=model or "whisper-1",
    ),
    overwrite=True,
)

register_tts_driver(
    "openai",
    lambda model=None: OpenAITTSDriver(  # type: ignore[misc]
        api_key=settings.openai_api_key,
        model=model or "tts-1",
    ),
    overwrite=True,
)

register_async_tts_driver(
    "openai",
    lambda model=None: AsyncOpenAITTSDriver(  # type: ignore[misc]
        api_key=settings.openai_api_key,
        model=model or "tts-1",
    ),
    overwrite=True,
)

# ── Register built-in ElevenLabs audio drivers ────────────────────────────

_elevenlabs_api_key = getattr(settings, "elevenlabs_api_key", None)
_elevenlabs_endpoint = getattr(settings, "elevenlabs_endpoint", "https://api.elevenlabs.io/v1")
_elevenlabs_tts_model = getattr(settings, "elevenlabs_tts_model", "eleven_multilingual_v2")

register_stt_driver(
    "elevenlabs",
    lambda model=None: ElevenLabsSTTDriver(  # type: ignore[misc]
        api_key=_elevenlabs_api_key,
        model=model or "scribe_v1",
        endpoint=_elevenlabs_endpoint,
    ),
    overwrite=True,
)

register_async_stt_driver(
    "elevenlabs",
    lambda model=None: AsyncElevenLabsSTTDriver(  # type: ignore[misc]
        api_key=_elevenlabs_api_key,
        model=model or "scribe_v1",
        endpoint=_elevenlabs_endpoint,
    ),
    overwrite=True,
)

register_tts_driver(
    "elevenlabs",
    lambda model=None: ElevenLabsTTSDriver(  # type: ignore[misc]
        api_key=_elevenlabs_api_key,
        model=model or _elevenlabs_tts_model,
        endpoint=_elevenlabs_endpoint,
    ),
    overwrite=True,
)

register_async_tts_driver(
    "elevenlabs",
    lambda model=None: AsyncElevenLabsTTSDriver(  # type: ignore[misc]
        api_key=_elevenlabs_api_key,
        model=model or _elevenlabs_tts_model,
        endpoint=_elevenlabs_endpoint,
    ),
    overwrite=True,
)


# ── Factory functions ─────────────────────────────────────────────────────


def get_stt_driver_for_model(model_str: str) -> STTDriver:
    """Instantiate a sync STT driver from a ``"provider/model"`` string.

    Args:
        model_str: e.g. ``"openai/whisper-1"`` or ``"elevenlabs/scribe_v1"``.

    Returns:
        A configured STT driver instance.
    """
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_stt_driver_factory(provider)
    return factory(model_id)


def get_async_stt_driver_for_model(model_str: str) -> AsyncSTTDriver:
    """Instantiate an async STT driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_stt_driver_factory(provider)
    return factory(model_id)


def get_tts_driver_for_model(model_str: str) -> TTSDriver:
    """Instantiate a sync TTS driver from a ``"provider/model"`` string.

    Args:
        model_str: e.g. ``"openai/tts-1"`` or ``"elevenlabs/eleven_multilingual_v2"``.

    Returns:
        A configured TTS driver instance.
    """
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_tts_driver_factory(provider)
    return factory(model_id)


def get_async_tts_driver_for_model(model_str: str) -> AsyncTTSDriver:
    """Instantiate an async TTS driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_tts_driver_factory(provider)
    return factory(model_id)
