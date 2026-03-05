"""Audio driver factory functions.

Provides high-level factory functions for instantiating STT/TTS drivers
by model string.  Built-in driver registration is handled centrally by
``provider_descriptors.register_all_builtin_drivers()``.

Usage:
    from prompture.drivers.audio_registry import get_stt_driver_for_model

    stt = get_stt_driver_for_model("openai/whisper-1")
    result = stt.transcribe(audio_bytes, {})
"""

from typing import cast

from .async_stt_base import AsyncSTTDriver
from .async_tts_base import AsyncTTSDriver
from .registry import (
    get_async_stt_driver_factory,
    get_async_tts_driver_factory,
    get_stt_driver_factory,
    get_tts_driver_factory,
)
from .stt_base import STTDriver
from .tts_base import TTSDriver

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
    return cast(STTDriver, factory(model_id))


def get_async_stt_driver_for_model(model_str: str) -> AsyncSTTDriver:
    """Instantiate an async STT driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_stt_driver_factory(provider)
    return cast(AsyncSTTDriver, factory(model_id))


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
    return cast(TTSDriver, factory(model_id))


def get_async_tts_driver_for_model(model_str: str) -> AsyncTTSDriver:
    """Instantiate an async TTS driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_tts_driver_factory(provider)
    return cast(AsyncTTSDriver, factory(model_id))
