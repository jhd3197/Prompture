"""Tests for audio driver registry — STT/TTS registration, lookup, listing."""

from __future__ import annotations

import pytest

from prompture.drivers.registry import (
    _reset_registries,
    get_async_stt_driver_factory,
    get_async_tts_driver_factory,
    get_stt_driver_factory,
    get_tts_driver_factory,
    is_async_stt_driver_registered,
    is_async_tts_driver_registered,
    is_stt_driver_registered,
    is_tts_driver_registered,
    list_registered_async_stt_drivers,
    list_registered_async_tts_drivers,
    list_registered_stt_drivers,
    list_registered_tts_drivers,
    register_async_stt_driver,
    register_async_tts_driver,
    register_stt_driver,
    register_tts_driver,
    unregister_async_stt_driver,
    unregister_async_tts_driver,
    unregister_stt_driver,
    unregister_tts_driver,
)


class DummySTT:
    def __init__(self, model=None):
        self.model = model


class DummyTTS:
    def __init__(self, model=None):
        self.model = model


@pytest.fixture(autouse=True)
def _clean_registries():
    """Reset all registries before each test, restore after."""
    _reset_registries()
    yield
    _reset_registries()


# ── STT Registration ──────────────────────────────────────────────────────


class TestSTTRegistration:
    def test_register_and_lookup(self):
        register_stt_driver("test_stt", lambda model=None: DummySTT(model))
        assert is_stt_driver_registered("test_stt")
        factory = get_stt_driver_factory("test_stt")
        driver = factory("whisper-1")
        assert isinstance(driver, DummySTT)
        assert driver.model == "whisper-1"

    def test_case_insensitive(self):
        register_stt_driver("TestSTT", lambda model=None: DummySTT(model))
        assert is_stt_driver_registered("teststt")
        assert is_stt_driver_registered("TESTSTT")

    def test_duplicate_raises(self):
        register_stt_driver("dup", lambda model=None: DummySTT())
        with pytest.raises(ValueError, match="already registered"):
            register_stt_driver("dup", lambda model=None: DummySTT())

    def test_overwrite(self):
        register_stt_driver("dup", lambda model=None: DummySTT("old"))
        register_stt_driver("dup", lambda model=None: DummySTT("new"), overwrite=True)
        driver = get_stt_driver_factory("dup")()
        assert driver.model == "new"

    def test_unregister(self):
        register_stt_driver("temp", lambda model=None: DummySTT())
        assert unregister_stt_driver("temp")
        assert not is_stt_driver_registered("temp")
        assert not unregister_stt_driver("temp")  # already gone

    def test_list(self):
        register_stt_driver("b_stt", lambda model=None: DummySTT())
        register_stt_driver("a_stt", lambda model=None: DummySTT())
        result = list_registered_stt_drivers()
        assert result == ["a_stt", "b_stt"]

    def test_lookup_missing_raises(self):
        with pytest.raises(ValueError, match="Unsupported STT"):
            get_stt_driver_factory("nonexistent")


# ── Async STT Registration ────────────────────────────────────────────────


class TestAsyncSTTRegistration:
    def test_register_and_lookup(self):
        register_async_stt_driver("test_async", lambda model=None: DummySTT(model))
        assert is_async_stt_driver_registered("test_async")
        factory = get_async_stt_driver_factory("test_async")
        driver = factory("whisper-1")
        assert driver.model == "whisper-1"

    def test_unregister(self):
        register_async_stt_driver("temp", lambda model=None: DummySTT())
        assert unregister_async_stt_driver("temp")
        assert not is_async_stt_driver_registered("temp")

    def test_list(self):
        register_async_stt_driver("z_async", lambda model=None: DummySTT())
        register_async_stt_driver("a_async", lambda model=None: DummySTT())
        result = list_registered_async_stt_drivers()
        assert result == ["a_async", "z_async"]


# ── TTS Registration ──────────────────────────────────────────────────────


class TestTTSRegistration:
    def test_register_and_lookup(self):
        register_tts_driver("test_tts", lambda model=None: DummyTTS(model))
        assert is_tts_driver_registered("test_tts")
        factory = get_tts_driver_factory("test_tts")
        driver = factory("tts-1")
        assert isinstance(driver, DummyTTS)
        assert driver.model == "tts-1"

    def test_duplicate_raises(self):
        register_tts_driver("dup", lambda model=None: DummyTTS())
        with pytest.raises(ValueError, match="already registered"):
            register_tts_driver("dup", lambda model=None: DummyTTS())

    def test_overwrite(self):
        register_tts_driver("dup", lambda model=None: DummyTTS("old"))
        register_tts_driver("dup", lambda model=None: DummyTTS("new"), overwrite=True)
        driver = get_tts_driver_factory("dup")()
        assert driver.model == "new"

    def test_unregister(self):
        register_tts_driver("temp", lambda model=None: DummyTTS())
        assert unregister_tts_driver("temp")
        assert not is_tts_driver_registered("temp")

    def test_list(self):
        register_tts_driver("b_tts", lambda model=None: DummyTTS())
        register_tts_driver("a_tts", lambda model=None: DummyTTS())
        result = list_registered_tts_drivers()
        assert result == ["a_tts", "b_tts"]

    def test_lookup_missing_raises(self):
        with pytest.raises(ValueError, match="Unsupported TTS"):
            get_tts_driver_factory("nonexistent")


# ── Async TTS Registration ────────────────────────────────────────────────


class TestAsyncTTSRegistration:
    def test_register_and_lookup(self):
        register_async_tts_driver("test_async", lambda model=None: DummyTTS(model))
        assert is_async_tts_driver_registered("test_async")
        factory = get_async_tts_driver_factory("test_async")
        driver = factory("tts-1-hd")
        assert driver.model == "tts-1-hd"

    def test_unregister(self):
        register_async_tts_driver("temp", lambda model=None: DummyTTS())
        assert unregister_async_tts_driver("temp")
        assert not is_async_tts_driver_registered("temp")

    def test_list(self):
        register_async_tts_driver("z_async", lambda model=None: DummyTTS())
        register_async_tts_driver("a_async", lambda model=None: DummyTTS())
        result = list_registered_async_tts_drivers()
        assert result == ["a_async", "z_async"]


# ── AudioCostMixin ────────────────────────────────────────────────────────


class TestAudioCostMixin:
    def test_stt_cost(self):
        from prompture.infra.cost_mixin import AudioCostMixin

        class TestDriver(AudioCostMixin):
            AUDIO_PRICING = {"whisper-1": {"per_second": 0.0001}}

        driver = TestDriver()
        cost = driver._calculate_audio_cost("openai", "whisper-1", duration_seconds=60)
        assert cost == pytest.approx(0.006, abs=1e-6)

    def test_tts_cost(self):
        from prompture.infra.cost_mixin import AudioCostMixin

        class TestDriver(AudioCostMixin):
            AUDIO_PRICING = {"tts-1": {"per_character": 0.000015}}

        driver = TestDriver()
        cost = driver._calculate_audio_cost("openai", "tts-1", characters=1_000_000)
        assert cost == pytest.approx(15.0, abs=1e-6)

    def test_unknown_model_zero_cost(self):
        from prompture.infra.cost_mixin import AudioCostMixin

        class TestDriver(AudioCostMixin):
            AUDIO_PRICING = {}

        driver = TestDriver()
        cost = driver._calculate_audio_cost("openai", "unknown", duration_seconds=60)
        assert cost == 0.0


# ── STT/TTS Base Class Contracts ──────────────────────────────────────────


class TestSTTBaseContract:
    def test_sync_transcribe_raises(self):
        from prompture.drivers.stt_base import STTDriver

        driver = STTDriver()
        with pytest.raises(NotImplementedError):
            driver.transcribe(b"audio", {})

    def test_async_transcribe_raises(self):
        import asyncio

        from prompture.drivers.async_stt_base import AsyncSTTDriver

        driver = AsyncSTTDriver()
        with pytest.raises(NotImplementedError):
            asyncio.run(driver.transcribe(b"audio", {}))


class TestTTSBaseContract:
    def test_sync_synthesize_raises(self):
        from prompture.drivers.tts_base import TTSDriver

        driver = TTSDriver()
        with pytest.raises(NotImplementedError):
            driver.synthesize("hello", {})

    def test_sync_stream_raises(self):
        from prompture.drivers.tts_base import TTSDriver

        driver = TTSDriver()
        with pytest.raises(NotImplementedError):
            driver.synthesize_stream("hello", {})

    def test_async_synthesize_raises(self):
        import asyncio

        from prompture.drivers.async_tts_base import AsyncTTSDriver

        driver = AsyncTTSDriver()
        with pytest.raises(NotImplementedError):
            asyncio.run(driver.synthesize("hello", {}))


# ── Hook wrappers ─────────────────────────────────────────────────────────


class TestSTTHooks:
    def test_transcribe_with_hooks_fires_callbacks(self):
        from prompture.drivers.stt_base import STTDriver
        from prompture.infra.callbacks import DriverCallbacks

        calls = []

        class TestSTT(STTDriver):
            def transcribe(self, audio, options):
                return {
                    "text": "hello",
                    "segments": [],
                    "language": "en",
                    "meta": {"duration_seconds": 1.0, "cost": 0.0},
                }

        cb = DriverCallbacks(
            on_request=lambda p: calls.append(("request", p)),
            on_response=lambda p: calls.append(("response", p)),
        )
        driver = TestSTT()
        driver.callbacks = cb

        result = driver.transcribe_with_hooks(b"audio", {})
        assert result["text"] == "hello"
        assert len(calls) == 2
        assert calls[0][0] == "request"
        assert calls[1][0] == "response"


class TestTTSHooks:
    def test_synthesize_with_hooks_fires_callbacks(self):
        from prompture.drivers.tts_base import TTSDriver
        from prompture.infra.callbacks import DriverCallbacks

        calls = []

        class TestTTS(TTSDriver):
            def synthesize(self, text, options):
                return {
                    "audio": b"fake audio",
                    "media_type": "audio/mpeg",
                    "meta": {"characters": len(text), "cost": 0.0},
                }

        cb = DriverCallbacks(
            on_request=lambda p: calls.append(("request", p)),
            on_response=lambda p: calls.append(("response", p)),
        )
        driver = TestTTS()
        driver.callbacks = cb

        result = driver.synthesize_with_hooks("hello", {})
        assert result["audio"] == b"fake audio"
        assert len(calls) == 2
        assert calls[0][0] == "request"
        assert calls[0][1]["text_length"] == 5
        assert calls[1][0] == "response"
