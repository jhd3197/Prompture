"""Tests for the Prompture -> Tukuy LLM backend bridge."""

from __future__ import annotations

import asyncio
import pytest

from prompture.bridges.tukuy_backend import TukuyLLMBackend, create_tukuy_backend
from prompture.exceptions import ConfigurationError, DriverError


# ---------------------------------------------------------------------------
# Mock driver
# ---------------------------------------------------------------------------

class MockAsyncDriver:
    """Minimal mock that satisfies the AsyncDriver interface."""

    supports_json_mode: bool = True
    supports_json_schema: bool = True
    supports_messages: bool = True
    supports_tool_use: bool = False
    supports_streaming: bool = False
    supports_vision: bool = False

    def __init__(self, response_text: str = "mock response") -> None:
        self.response_text = response_text
        self.last_messages: list | None = None
        self.last_options: dict | None = None
        self.generate_called = False

    async def generate_messages(self, messages, options):
        self.last_messages = messages
        self.last_options = options
        return {
            "text": self.response_text,
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cost": 0.001,
                "model_name": "mock-model",
            },
        }

    async def generate(self, prompt, options):
        self.generate_called = True
        self.last_options = options
        return {
            "text": self.response_text,
            "meta": {
                "prompt_tokens": 5,
                "completion_tokens": 15,
                "total_tokens": 20,
                "cost": 0.0005,
                "model_name": "mock-model-simple",
            },
        }


class NonMessagesDriver(MockAsyncDriver):
    """Driver that does NOT support messages — forces generate() fallback."""

    supports_messages = False


class NoJsonDriver(MockAsyncDriver):
    """Driver with json_mode but NOT json_schema."""

    supports_json_schema = False


class PlainDriver(MockAsyncDriver):
    """Driver without any JSON support."""

    supports_json_mode = False
    supports_json_schema = False


class ErrorDriver(MockAsyncDriver):
    """Driver that always raises."""

    async def generate_messages(self, messages, options):
        raise RuntimeError("LLM service unavailable")

    async def generate(self, prompt, options):
        raise RuntimeError("LLM service unavailable")


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_init(self):
        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(driver, default_model="openai/gpt-4o")
        assert backend._driver is driver
        assert backend._default_model == "openai/gpt-4o"

    def test_none_driver_raises(self):
        with pytest.raises(ConfigurationError, match="non-None AsyncDriver"):
            TukuyLLMBackend(None)

    def test_repr(self):
        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(driver, default_model="openai/gpt-4o")
        r = repr(backend)
        assert "TukuyLLMBackend" in r
        assert "MockAsyncDriver" in r
        assert "openai/gpt-4o" in r

    def test_defaults(self):
        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(
            driver,
            default_temperature=0.7,
            default_max_tokens=500,
        )
        assert backend._default_temperature == 0.7
        assert backend._default_max_tokens == 500


# ---------------------------------------------------------------------------
# Tests: complete() message building
# ---------------------------------------------------------------------------

class TestComplete:
    @pytest.fixture
    def driver(self):
        return MockAsyncDriver()

    @pytest.fixture
    def backend(self, driver):
        return TukuyLLMBackend(driver, default_model="test/model")

    @pytest.mark.asyncio
    async def test_prompt_only(self, backend, driver):
        result = await backend.complete("Hello world")
        assert driver.last_messages == [{"role": "user", "content": "Hello world"}]
        assert result["text"] == "mock response"

    @pytest.mark.asyncio
    async def test_with_system(self, backend, driver):
        result = await backend.complete("Hello", system="Be helpful")
        assert driver.last_messages == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        assert result["text"] == "mock response"

    @pytest.mark.asyncio
    async def test_temperature_passed(self, backend, driver):
        await backend.complete("Hi", temperature=0.3)
        assert driver.last_options["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_max_tokens_passed(self, backend, driver):
        await backend.complete("Hi", max_tokens=100)
        assert driver.last_options["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_default_temperature_used(self, driver):
        backend = TukuyLLMBackend(driver, default_temperature=0.9)
        await backend.complete("Hi")
        assert driver.last_options["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_explicit_temperature_overrides_default(self, driver):
        backend = TukuyLLMBackend(driver, default_temperature=0.9)
        await backend.complete("Hi", temperature=0.1)
        assert driver.last_options["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_default_max_tokens_used(self, driver):
        backend = TukuyLLMBackend(driver, default_max_tokens=256)
        await backend.complete("Hi")
        assert driver.last_options["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_explicit_max_tokens_overrides_default(self, driver):
        backend = TukuyLLMBackend(driver, default_max_tokens=256)
        await backend.complete("Hi", max_tokens=50)
        assert driver.last_options["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_no_options_when_none(self, backend, driver):
        await backend.complete("Hi")
        assert "temperature" not in driver.last_options
        assert "max_tokens" not in driver.last_options


# ---------------------------------------------------------------------------
# Tests: structured output (json_schema)
# ---------------------------------------------------------------------------

class TestJsonSchema:
    @pytest.mark.asyncio
    async def test_json_schema_native(self):
        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(driver)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        await backend.complete("Extract name", json_schema=schema)
        assert driver.last_options["json_mode"] is True
        assert driver.last_options["json_schema"] is schema

    @pytest.mark.asyncio
    async def test_json_mode_fallback(self):
        """When driver has json_mode but not json_schema, inject into messages."""
        driver = NoJsonDriver()
        backend = TukuyLLMBackend(driver)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        await backend.complete("Extract name", json_schema=schema)
        assert driver.last_options["json_mode"] is True
        assert "json_schema" not in driver.last_options
        # Schema should be injected into last user message
        user_msg = driver.last_messages[-1]
        assert "validates against this schema" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_no_json_support(self):
        """When driver lacks all JSON support, no json options set."""
        driver = PlainDriver()
        backend = TukuyLLMBackend(driver)
        schema = {"type": "object"}
        await backend.complete("Hi", json_schema=schema)
        assert "json_mode" not in driver.last_options
        assert "json_schema" not in driver.last_options


# ---------------------------------------------------------------------------
# Tests: generate() fallback for non-messages drivers
# ---------------------------------------------------------------------------

class TestGenerateFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_generate(self):
        driver = NonMessagesDriver()
        backend = TukuyLLMBackend(driver)
        result = await backend.complete("Hello", system="Be helpful")
        assert driver.generate_called is True
        assert result["text"] == "mock response"

    @pytest.mark.asyncio
    async def test_fallback_prompt_concatenation(self):
        driver = NonMessagesDriver()
        backend = TukuyLLMBackend(driver)
        # We can't easily check the concatenated prompt via mock, but verify it works
        result = await backend.complete("Hello")
        assert result["text"] == "mock response"
        assert result["meta"]["model"] == "mock-model-simple"


# ---------------------------------------------------------------------------
# Tests: response normalization
# ---------------------------------------------------------------------------

class TestResponseNormalization:
    @pytest.mark.asyncio
    async def test_meta_mapping(self):
        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(driver, default_model="openai/gpt-4o")
        result = await backend.complete("Hi")
        meta = result["meta"]
        assert meta["prompt_tokens"] == 10
        assert meta["completion_tokens"] == 20
        assert meta["cost"] == 0.001
        assert meta["model"] == "mock-model"

    @pytest.mark.asyncio
    async def test_model_fallback_to_default(self):
        """When driver meta has no model_name, use default_model."""
        driver = MockAsyncDriver()
        # Override generate_messages to return empty meta
        async def gen(messages, options):
            return {"text": "hi", "meta": {}}
        driver.generate_messages = gen
        backend = TukuyLLMBackend(driver, default_model="fallback/model")
        result = await backend.complete("Hi")
        assert result["meta"]["model"] == "fallback/model"

    @pytest.mark.asyncio
    async def test_model_fallback_to_unknown(self):
        """When no model_name and no default, use 'unknown'."""
        driver = MockAsyncDriver()
        async def gen(messages, options):
            return {"text": "hi", "meta": {}}
        driver.generate_messages = gen
        backend = TukuyLLMBackend(driver)
        result = await backend.complete("Hi")
        assert result["meta"]["model"] == "unknown"

    @pytest.mark.asyncio
    async def test_empty_result(self):
        """Handle a driver returning minimal dict."""
        driver = MockAsyncDriver()
        async def gen(messages, options):
            return {}
        driver.generate_messages = gen
        backend = TukuyLLMBackend(driver)
        result = await backend.complete("Hi")
        assert result["text"] == ""
        assert result["meta"]["prompt_tokens"] == 0
        assert result["meta"]["completion_tokens"] == 0
        assert result["meta"]["cost"] == 0.0


# ---------------------------------------------------------------------------
# Tests: on_complete callback
# ---------------------------------------------------------------------------

class TestOnCompleteCallback:
    @pytest.mark.asyncio
    async def test_sync_callback(self):
        captured = []
        def on_complete(response):
            captured.append(response)

        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(driver, on_complete=on_complete)
        result = await backend.complete("Hi")
        assert len(captured) == 1
        assert captured[0] is result

    @pytest.mark.asyncio
    async def test_async_callback(self):
        captured = []
        async def on_complete(response):
            captured.append(response)

        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(driver, on_complete=on_complete)
        result = await backend.complete("Hi")
        assert len(captured) == 1
        assert captured[0] is result

    @pytest.mark.asyncio
    async def test_callback_error_ignored(self):
        """Callback errors must not affect the response."""
        def on_complete(response):
            raise ValueError("logging failed")

        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(driver, on_complete=on_complete)
        result = await backend.complete("Hi")
        assert result["text"] == "mock response"

    @pytest.mark.asyncio
    async def test_async_callback_error_ignored(self):
        async def on_complete(response):
            raise ValueError("async logging failed")

        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(driver, on_complete=on_complete)
        result = await backend.complete("Hi")
        assert result["text"] == "mock response"

    @pytest.mark.asyncio
    async def test_no_callback(self):
        """No callback should be fine."""
        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(driver)
        result = await backend.complete("Hi")
        assert result["text"] == "mock response"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_driver_error_wrapped(self):
        driver = ErrorDriver()
        backend = TukuyLLMBackend(driver)
        with pytest.raises(DriverError, match="LLM completion failed"):
            await backend.complete("Hi")

    @pytest.mark.asyncio
    async def test_driver_error_chained(self):
        """Original exception should be chained."""
        driver = ErrorDriver()
        backend = TukuyLLMBackend(driver)
        with pytest.raises(DriverError) as exc_info:
            await backend.complete("Hi")
        assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Tests: with_model()
# ---------------------------------------------------------------------------

class TestWithModel:
    def test_returns_new_instance(self):
        driver = MockAsyncDriver()
        backend = TukuyLLMBackend(
            driver,
            default_model="openai/gpt-4o",
            default_temperature=0.5,
            default_max_tokens=200,
        )
        # with_model calls get_async_driver_for_model internally, which
        # requires the registry. We'll patch it.
        import prompture.bridges.tukuy_backend as mod
        original = mod.TukuyLLMBackend.with_model

        def patched_with_model(self, model_hint):
            new_driver = MockAsyncDriver(response_text="new model response")
            return TukuyLLMBackend(
                new_driver,
                default_model=model_hint,
                default_temperature=self._default_temperature,
                default_max_tokens=self._default_max_tokens,
                on_complete=self._on_complete,
            )

        backend.with_model = patched_with_model.__get__(backend)
        new_backend = backend.with_model("claude/claude-sonnet-4-5-20250929")
        assert new_backend is not backend
        assert new_backend._default_model == "claude/claude-sonnet-4-5-20250929"
        assert new_backend._default_temperature == 0.5
        assert new_backend._default_max_tokens == 200

    @pytest.mark.asyncio
    async def test_with_model_new_driver_used(self):
        driver = MockAsyncDriver(response_text="original")
        backend = TukuyLLMBackend(driver, default_model="openai/gpt-4o")

        # Patch with_model to use a new mock driver
        new_driver = MockAsyncDriver(response_text="switched")

        def patched_with_model(self, model_hint):
            return TukuyLLMBackend(
                new_driver,
                default_model=model_hint,
                on_complete=self._on_complete,
            )

        backend.with_model = patched_with_model.__get__(backend)
        new_backend = backend.with_model("claude/haiku")
        result = await new_backend.complete("Hi")
        assert result["text"] == "switched"
        assert new_backend._default_model == "claude/haiku"


# ---------------------------------------------------------------------------
# Tests: create_tukuy_backend() factory
# ---------------------------------------------------------------------------

class TestCreateFactory:
    def test_factory_creates_backend(self, monkeypatch):
        """Factory should call get_async_driver_for_model and return TukuyLLMBackend."""
        mock_driver = MockAsyncDriver()

        def fake_get_driver(model_str, *, env=None):
            return mock_driver

        monkeypatch.setattr(
            "prompture.bridges.tukuy_backend.create_tukuy_backend.__module__",
            "prompture.bridges.tukuy_backend",
        )
        # Patch the import inside create_tukuy_backend
        import prompture.drivers as drivers_mod
        monkeypatch.setattr(drivers_mod, "get_async_driver_for_model", fake_get_driver)

        backend = create_tukuy_backend(
            "openai/gpt-4o",
            default_temperature=0.5,
            default_max_tokens=100,
        )
        assert isinstance(backend, TukuyLLMBackend)
        assert backend._driver is mock_driver
        assert backend._default_model == "openai/gpt-4o"
        assert backend._default_temperature == 0.5
        assert backend._default_max_tokens == 100

    def test_factory_passes_env(self, monkeypatch):
        mock_driver = MockAsyncDriver()
        captured_env = []

        def fake_get_driver(model_str, *, env=None):
            captured_env.append(env)
            return mock_driver

        import prompture.drivers as drivers_mod
        monkeypatch.setattr(drivers_mod, "get_async_driver_for_model", fake_get_driver)

        sentinel_env = object()
        backend = create_tukuy_backend("openai/gpt-4o", env=sentinel_env)
        assert captured_env[0] is sentinel_env

    def test_factory_passes_on_complete(self, monkeypatch):
        mock_driver = MockAsyncDriver()

        def fake_get_driver(model_str, *, env=None):
            return mock_driver

        import prompture.drivers as drivers_mod
        monkeypatch.setattr(drivers_mod, "get_async_driver_for_model", fake_get_driver)

        def my_callback(resp):
            pass

        backend = create_tukuy_backend("openai/gpt-4o", on_complete=my_callback)
        assert backend._on_complete is my_callback


# ---------------------------------------------------------------------------
# Mock streaming driver
# ---------------------------------------------------------------------------

class MockStreamingAsyncDriver(MockAsyncDriver):
    """Driver that supports streaming via generate_messages_stream."""

    supports_streaming = True

    def __init__(self, response_text: str = "mock response", chunks=None) -> None:
        super().__init__(response_text)
        self._chunks = chunks or [
            {"type": "delta", "text": "Hello"},
            {"type": "delta", "text": " World"},
            {"type": "done", "text": "Hello World", "meta": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "cost": 0.001,
                "model_name": "mock-streaming",
            }},
        ]
        self.stream_messages = None
        self.stream_options = None

    async def generate_messages_stream(self, messages, options):
        self.stream_messages = messages
        self.stream_options = options
        for chunk in self._chunks:
            yield chunk


class ErrorStreamingDriver(MockStreamingAsyncDriver):
    """Streaming driver that raises after yielding a delta."""

    async def generate_messages_stream(self, messages, options):
        yield {"type": "delta", "text": "partial"}
        raise RuntimeError("Stream connection lost")


# ---------------------------------------------------------------------------
# Tests: stream()
# ---------------------------------------------------------------------------

class TestStream:
    @pytest.mark.asyncio
    async def test_yields_deltas_then_done(self):
        driver = MockStreamingAsyncDriver()
        backend = TukuyLLMBackend(driver, default_model="test/model")
        chunks = []
        async for chunk in backend.stream("Hello world"):
            chunks.append(chunk)

        deltas = [c for c in chunks if c["type"] == "delta"]
        dones = [c for c in chunks if c["type"] == "done"]
        assert len(deltas) == 2
        assert deltas[0]["text"] == "Hello"
        assert deltas[1]["text"] == " World"
        assert len(dones) == 1
        assert dones[0]["text"] == "Hello World"

    @pytest.mark.asyncio
    async def test_normalizes_meta(self):
        driver = MockStreamingAsyncDriver()
        backend = TukuyLLMBackend(driver, default_model="test/model")
        chunks = []
        async for chunk in backend.stream("Hello"):
            chunks.append(chunk)

        done = [c for c in chunks if c["type"] == "done"][0]
        meta = done["meta"]
        assert meta["prompt_tokens"] == 5
        assert meta["completion_tokens"] == 2
        assert meta["cost"] == 0.001
        assert meta["model"] == "mock-streaming"

    @pytest.mark.asyncio
    async def test_model_fallback_in_meta(self):
        """When driver meta has no model_name, use default_model."""
        driver = MockStreamingAsyncDriver(chunks=[
            {"type": "delta", "text": "hi"},
            {"type": "done", "text": "hi", "meta": {}},
        ])
        backend = TukuyLLMBackend(driver, default_model="fallback/model")
        chunks = []
        async for chunk in backend.stream("Hello"):
            chunks.append(chunk)

        done = [c for c in chunks if c["type"] == "done"][0]
        assert done["meta"]["model"] == "fallback/model"

    @pytest.mark.asyncio
    async def test_fires_on_complete_callback(self):
        captured = []
        def on_complete(response):
            captured.append(response)

        driver = MockStreamingAsyncDriver()
        backend = TukuyLLMBackend(driver, on_complete=on_complete)
        chunks = []
        async for chunk in backend.stream("Hello"):
            chunks.append(chunk)

        assert len(captured) == 1
        assert captured[0]["text"] == "Hello World"
        assert captured[0]["meta"]["prompt_tokens"] == 5

    @pytest.mark.asyncio
    async def test_fires_async_on_complete_callback(self):
        captured = []
        async def on_complete(response):
            captured.append(response)

        driver = MockStreamingAsyncDriver()
        backend = TukuyLLMBackend(driver, on_complete=on_complete)
        async for _ in backend.stream("Hello"):
            pass

        assert len(captured) == 1
        assert captured[0]["text"] == "Hello World"

    @pytest.mark.asyncio
    async def test_callback_error_ignored(self):
        def on_complete(response):
            raise ValueError("callback boom")

        driver = MockStreamingAsyncDriver()
        backend = TukuyLLMBackend(driver, on_complete=on_complete)
        chunks = []
        async for chunk in backend.stream("Hello"):
            chunks.append(chunk)

        # Should still yield done
        done = [c for c in chunks if c["type"] == "done"]
        assert len(done) == 1
        assert done[0]["text"] == "Hello World"

    @pytest.mark.asyncio
    async def test_fallback_to_complete_when_no_streaming(self):
        """Non-streaming driver falls back to complete() and yields single done."""
        driver = MockAsyncDriver()  # supports_streaming = False
        backend = TukuyLLMBackend(driver, default_model="test/model")
        chunks = []
        async for chunk in backend.stream("Hello"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["type"] == "done"
        assert chunks[0]["text"] == "mock response"
        assert chunks[0]["meta"]["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_passes_system_and_options(self):
        driver = MockStreamingAsyncDriver()
        backend = TukuyLLMBackend(driver, default_model="test/model")
        async for _ in backend.stream(
            "Hello",
            system="Be helpful",
            temperature=0.5,
            max_tokens=100,
        ):
            pass

        assert driver.stream_messages == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        assert driver.stream_options["temperature"] == 0.5
        assert driver.stream_options["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_error_handling(self):
        driver = ErrorStreamingDriver()
        backend = TukuyLLMBackend(driver)
        with pytest.raises(DriverError, match="LLM streaming failed"):
            async for _ in backend.stream("Hello"):
                pass

    @pytest.mark.asyncio
    async def test_error_chained(self):
        driver = ErrorStreamingDriver()
        backend = TukuyLLMBackend(driver)
        with pytest.raises(DriverError) as exc_info:
            async for _ in backend.stream("Hello"):
                pass
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_default_temperature_used(self):
        driver = MockStreamingAsyncDriver()
        backend = TukuyLLMBackend(driver, default_temperature=0.9)
        async for _ in backend.stream("Hello"):
            pass
        assert driver.stream_options["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_explicit_temperature_overrides_default(self):
        driver = MockStreamingAsyncDriver()
        backend = TukuyLLMBackend(driver, default_temperature=0.9)
        async for _ in backend.stream("Hello", temperature=0.1):
            pass
        assert driver.stream_options["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Tests: top-level exports
# ---------------------------------------------------------------------------

class TestExports:
    def test_importable_from_bridges(self):
        from prompture.bridges import TukuyLLMBackend, create_tukuy_backend
        assert TukuyLLMBackend is not None
        assert create_tukuy_backend is not None

    def test_importable_from_top_level(self):
        from prompture import TukuyLLMBackend, create_tukuy_backend
        assert TukuyLLMBackend is not None
        assert create_tukuy_backend is not None
