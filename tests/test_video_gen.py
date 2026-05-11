"""Tests for video generation registry, base classes, costs, and Grok driver."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from prompture.drivers.registry import (
    _ASYNC_VIDEO_GEN_REGISTRY,
    _VIDEO_GEN_REGISTRY,
    get_async_video_gen_driver_factory,
    get_video_gen_driver_factory,
    is_async_video_gen_driver_registered,
    is_video_gen_driver_registered,
    list_registered_async_video_gen_drivers,
    list_registered_video_gen_drivers,
    register_async_video_gen_driver,
    register_video_gen_driver,
    unregister_async_video_gen_driver,
    unregister_video_gen_driver,
)


class DummyVideoGen:
    """Dummy video gen driver for registry tests."""

    def __init__(self, model=None):
        self.model = model


@pytest.fixture(autouse=True)
def _clean_video_gen_registries():
    """Reset only video gen registries before each test, restore after."""
    saved = (
        dict(_VIDEO_GEN_REGISTRY),
        dict(_ASYNC_VIDEO_GEN_REGISTRY),
    )
    _VIDEO_GEN_REGISTRY.clear()
    _ASYNC_VIDEO_GEN_REGISTRY.clear()
    yield
    _VIDEO_GEN_REGISTRY.clear()
    _ASYNC_VIDEO_GEN_REGISTRY.clear()
    _VIDEO_GEN_REGISTRY.update(saved[0])
    _ASYNC_VIDEO_GEN_REGISTRY.update(saved[1])


class TestVideoGenRegistration:
    """Tests for sync video generation registration."""

    def test_register_and_lookup(self):
        register_video_gen_driver("test_video", lambda model=None: DummyVideoGen(model))
        assert is_video_gen_driver_registered("test_video")
        factory = get_video_gen_driver_factory("test_video")
        driver = factory("grok-imagine-video")
        assert isinstance(driver, DummyVideoGen)
        assert driver.model == "grok-imagine-video"

    def test_case_insensitive(self):
        register_video_gen_driver("TestVideo", lambda model=None: DummyVideoGen(model))
        assert is_video_gen_driver_registered("testvideo")
        assert is_video_gen_driver_registered("TESTVIDEO")

    def test_duplicate_raises(self):
        register_video_gen_driver("dup", lambda model=None: DummyVideoGen())
        with pytest.raises(ValueError, match="already registered"):
            register_video_gen_driver("dup", lambda model=None: DummyVideoGen())

    def test_overwrite(self):
        register_video_gen_driver("dup", lambda model=None: DummyVideoGen("old"))
        register_video_gen_driver("dup", lambda model=None: DummyVideoGen("new"), overwrite=True)
        driver = get_video_gen_driver_factory("dup")()
        assert driver.model == "new"

    def test_unregister(self):
        register_video_gen_driver("temp", lambda model=None: DummyVideoGen())
        assert unregister_video_gen_driver("temp")
        assert not is_video_gen_driver_registered("temp")
        assert not unregister_video_gen_driver("temp")

    def test_list(self):
        register_video_gen_driver("b_video", lambda model=None: DummyVideoGen())
        register_video_gen_driver("a_video", lambda model=None: DummyVideoGen())
        assert list_registered_video_gen_drivers() == ["a_video", "b_video"]

    def test_lookup_missing_raises(self):
        with pytest.raises(ValueError, match="Unsupported video gen"):
            get_video_gen_driver_factory("nonexistent")


class TestAsyncVideoGenRegistration:
    """Tests for async video generation registration."""

    def test_register_and_lookup(self):
        register_async_video_gen_driver("test_async", lambda model=None: DummyVideoGen(model))
        assert is_async_video_gen_driver_registered("test_async")
        factory = get_async_video_gen_driver_factory("test_async")
        driver = factory("grok-imagine-video")
        assert driver.model == "grok-imagine-video"

    def test_unregister(self):
        register_async_video_gen_driver("temp", lambda model=None: DummyVideoGen())
        assert unregister_async_video_gen_driver("temp")
        assert not is_async_video_gen_driver_registered("temp")

    def test_list(self):
        register_async_video_gen_driver("z_async", lambda model=None: DummyVideoGen())
        register_async_video_gen_driver("a_async", lambda model=None: DummyVideoGen())
        assert list_registered_async_video_gen_drivers() == ["a_async", "z_async"]


class TestVideoCostMixin:
    """Tests for VideoCostMixin."""

    def test_per_second_cost(self):
        from prompture.infra.cost_mixin import VideoCostMixin

        class TestDriver(VideoCostMixin):
            VIDEO_PRICING = {"model": {"per_second": 0.05}}

        driver = TestDriver()
        assert driver._calculate_video_cost("test", "model", duration_seconds=10) == pytest.approx(0.5)

    def test_resolution_specific_per_second_cost(self):
        from prompture.infra.cost_mixin import VideoCostMixin

        class TestDriver(VideoCostMixin):
            VIDEO_PRICING = {"model": {"per_second": 0.05, "per_second_by_resolution": {"720p": 0.07}}}

        driver = TestDriver()
        cost = driver._calculate_video_cost("test", "model", duration_seconds=10, resolution="720p")
        assert cost == pytest.approx(0.7)

    def test_per_video_cost(self):
        from prompture.infra.cost_mixin import VideoCostMixin

        class TestDriver(VideoCostMixin):
            VIDEO_PRICING = {"model": {"per_video": 1.25}}

        driver = TestDriver()
        assert driver._calculate_video_cost("test", "model", n=2) == pytest.approx(2.5)

    def test_unknown_model_zero_cost(self):
        from prompture.infra.cost_mixin import VideoCostMixin

        class TestDriver(VideoCostMixin):
            VIDEO_PRICING = {}

        driver = TestDriver()
        assert driver._calculate_video_cost("test", "unknown", duration_seconds=10) == 0.0


class TestVideoGenBaseContract:
    """Tests for video generation base contracts."""

    def test_sync_generate_video_raises(self):
        from prompture.drivers.video_gen_base import VideoGenDriver

        driver = VideoGenDriver()
        with pytest.raises(NotImplementedError):
            driver.generate_video("a cat", {})

    def test_async_generate_video_raises(self):
        from prompture.drivers.async_video_gen_base import AsyncVideoGenDriver

        driver = AsyncVideoGenDriver()
        with pytest.raises(NotImplementedError):
            asyncio.run(driver.generate_video("a cat", {}))

    def test_sync_class_attributes(self):
        from prompture.drivers.video_gen_base import VideoGenDriver

        driver = VideoGenDriver()
        assert driver.supports_polling is True
        assert driver.supports_image_input is False
        assert driver.callbacks is None


class TestVideoGenHooks:
    """Tests for video generation hook wrappers."""

    def test_sync_generate_with_hooks_fires_callbacks(self):
        from prompture.drivers.video_gen_base import VideoGenDriver
        from prompture.infra.callbacks import DriverCallbacks
        from prompture.media.video import video_from_url

        calls = []

        class TestVideo(VideoGenDriver):
            def generate_video(self, prompt, options):
                return {
                    "videos": [video_from_url("https://example.com/video.mp4")],
                    "meta": {"video_count": 1, "cost": 0.0},
                }

        driver = TestVideo()
        driver.callbacks = DriverCallbacks(
            on_request=lambda p: calls.append(("request", p)),
            on_response=lambda p: calls.append(("response", p)),
        )

        result = driver.generate_video_with_hooks("a cat", {})
        assert result["meta"]["video_count"] == 1
        assert len(calls) == 2
        assert calls[0][1]["prompt_length"] == 5
        assert calls[1][1]["video_count"] == 1

    def test_async_generate_with_hooks_fires_callbacks(self):
        from prompture.drivers.async_video_gen_base import AsyncVideoGenDriver
        from prompture.infra.callbacks import DriverCallbacks
        from prompture.media.video import video_from_url

        calls = []

        class TestAsyncVideo(AsyncVideoGenDriver):
            async def generate_video(self, prompt, options):
                return {
                    "videos": [video_from_url("https://example.com/video.mp4")],
                    "meta": {"video_count": 1, "cost": 0.0},
                }

        driver = TestAsyncVideo()
        driver.callbacks = DriverCallbacks(
            on_request=lambda p: calls.append(("request", p)),
            on_response=lambda p: calls.append(("response", p)),
        )

        result = asyncio.run(driver.generate_video_with_hooks("a cat", {}))
        assert result["meta"]["video_count"] == 1
        assert len(calls) == 2


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class TestGrokVideoGenDriver:
    """Tests for Grok video generation driver."""

    def test_attributes(self):
        from prompture.drivers.grok_video_gen_driver import GrokVideoGenDriver

        assert "grok-imagine-video" in GrokVideoGenDriver.KNOWN_MODELS
        assert "16:9" in GrokVideoGenDriver.supported_aspect_ratios
        assert "720p" in GrokVideoGenDriver.supported_resolutions
        assert GrokVideoGenDriver.supports_image_input is True
        assert GrokVideoGenDriver.supports_audio is True

    @patch("prompture.drivers.grok_video_gen_driver.httpx.get")
    @patch("prompture.drivers.grok_video_gen_driver.httpx.post")
    def test_generate_video_polls_and_returns_url(self, mock_post, mock_get):
        from prompture.drivers.grok_video_gen_driver import GrokVideoGenDriver

        mock_post.return_value = _Response(200, {"request_id": "req_123"})
        mock_get.return_value = _Response(
            200,
            {"status": "done", "video": {"url": "https://vidgen.x.ai/video.mp4", "duration": 8}},
        )

        driver = GrokVideoGenDriver(api_key="xai-test")
        result = driver.generate_video(
            "cinematic launch",
            {"duration": 8, "resolution": "720p", "poll_interval": 0, "timeout": 1},
        )

        assert result["videos"][0].url == "https://vidgen.x.ai/video.mp4"
        assert result["meta"]["video_count"] == 1
        assert result["meta"]["request_id"] == "req_123"
        assert result["meta"]["resolution"] == "720p"
        assert result["meta"]["cost"] == pytest.approx(0.56)
        mock_post.assert_called_once()
        mock_get.assert_called_once()

    @patch("prompture.drivers.grok_video_gen_driver.httpx.post")
    def test_generate_video_without_poll_returns_request_id(self, mock_post):
        from prompture.drivers.grok_video_gen_driver import GrokVideoGenDriver

        mock_post.return_value = _Response(200, {"request_id": "req_123", "status": "pending"})
        driver = GrokVideoGenDriver(api_key="xai-test")
        result = driver.generate_video("cinematic launch", {"poll": False})

        assert result["videos"] == []
        assert result["meta"]["request_id"] == "req_123"
        assert result["meta"]["status"] == "pending"

    def test_missing_key_raises(self):
        from prompture.drivers.grok_video_gen_driver import GrokVideoGenDriver

        with patch.dict("os.environ", {}, clear=True):
            driver = GrokVideoGenDriver(api_key=None)
            with pytest.raises(RuntimeError, match="GROK_API_KEY or XAI_API_KEY"):
                driver.generate_video("test", {})

    def test_xai_api_key_alias_is_supported(self):
        from prompture.drivers.grok_video_gen_driver import GrokVideoGenDriver

        with patch.dict("os.environ", {"XAI_API_KEY": "xai-test"}, clear=True):
            driver = GrokVideoGenDriver(api_key=None)
            assert driver.api_key == "xai-test"

    def test_image_and_reference_images_conflict(self):
        from prompture.drivers.grok_video_gen_driver import GrokVideoGenDriver

        driver = GrokVideoGenDriver(api_key="xai-test")
        with pytest.raises(ValueError, match="cannot be used together"):
            driver.generate_video(
                "test",
                {
                    "image": "https://example.com/a.png",
                    "reference_images": ["https://example.com/b.png"],
                },
            )

    def test_reference_images_limit(self):
        from prompture.drivers.grok_video_gen_driver import GrokVideoGenDriver

        driver = GrokVideoGenDriver(api_key="xai-test")
        with pytest.raises(ValueError, match="at most 7"):
            driver.generate_video("test", {"reference_images": ["https://example.com/a.png"] * 8})

    def test_reference_images_duration_limit(self):
        from prompture.drivers.grok_video_gen_driver import GrokVideoGenDriver

        driver = GrokVideoGenDriver(api_key="xai-test")
        with pytest.raises(ValueError, match="10 seconds or less"):
            driver.generate_video("test", {"reference_images": ["https://example.com/a.png"], "duration": 11})

    def test_factory_resolves_builtin_driver(self):
        from prompture.drivers import get_video_gen_driver_for_model
        from prompture.drivers.grok_video_gen_driver import GrokVideoGenDriver
        from prompture.drivers.provider_descriptors import register_all_builtin_drivers

        register_all_builtin_drivers()
        driver = get_video_gen_driver_for_model("grok/grok-imagine-video")
        assert isinstance(driver, GrokVideoGenDriver)
        assert driver.model == "grok-imagine-video"


class TestVideoDiscovery:
    """Tests for video generation model discovery."""

    @patch("prompture.infra.discovery.get_available_models", return_value=[])
    @patch.dict("os.environ", {"GROK_API_KEY": "xai-test"})
    def test_grok_video_model_discovered(self, mock_models):
        from prompture.infra.discovery import get_available_video_gen_models

        models = get_available_video_gen_models()
        assert "grok/grok-imagine-video" in models

    @patch("prompture.infra.discovery.get_available_models", return_value=[])
    @patch.dict("os.environ", {"XAI_API_KEY": "xai-test"}, clear=True)
    def test_grok_video_model_discovered_with_xai_key_alias(self, mock_models):
        from prompture.infra.discovery import get_available_video_gen_models

        models = get_available_video_gen_models()
        assert "grok/grok-imagine-video" in models
