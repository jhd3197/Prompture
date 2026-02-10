"""Tests for image generation driver registry, base classes, cost mixin, and hooks."""

from __future__ import annotations

import asyncio

import pytest

from prompture.drivers.registry import (
    _ASYNC_IMG_GEN_REGISTRY,
    _IMG_GEN_REGISTRY,
    get_async_img_gen_driver_factory,
    get_img_gen_driver_factory,
    is_async_img_gen_driver_registered,
    is_img_gen_driver_registered,
    list_registered_async_img_gen_drivers,
    list_registered_img_gen_drivers,
    register_async_img_gen_driver,
    register_img_gen_driver,
    unregister_async_img_gen_driver,
    unregister_img_gen_driver,
)


class DummyImageGen:
    def __init__(self, model=None):
        self.model = model


@pytest.fixture(autouse=True)
def _clean_img_gen_registries():
    """Reset only image gen registries before each test, restore after."""
    saved = (
        dict(_IMG_GEN_REGISTRY),
        dict(_ASYNC_IMG_GEN_REGISTRY),
    )
    _IMG_GEN_REGISTRY.clear()
    _ASYNC_IMG_GEN_REGISTRY.clear()
    yield
    _IMG_GEN_REGISTRY.clear()
    _ASYNC_IMG_GEN_REGISTRY.clear()
    _IMG_GEN_REGISTRY.update(saved[0])
    _ASYNC_IMG_GEN_REGISTRY.update(saved[1])


# ── Sync Image Gen Registration ──────────────────────────────────────────


class TestImageGenRegistration:
    def test_register_and_lookup(self):
        register_img_gen_driver("test_ig", lambda model=None: DummyImageGen(model))
        assert is_img_gen_driver_registered("test_ig")
        factory = get_img_gen_driver_factory("test_ig")
        driver = factory("dall-e-3")
        assert isinstance(driver, DummyImageGen)
        assert driver.model == "dall-e-3"

    def test_case_insensitive(self):
        register_img_gen_driver("TestIG", lambda model=None: DummyImageGen(model))
        assert is_img_gen_driver_registered("testig")
        assert is_img_gen_driver_registered("TESTIG")

    def test_duplicate_raises(self):
        register_img_gen_driver("dup", lambda model=None: DummyImageGen())
        with pytest.raises(ValueError, match="already registered"):
            register_img_gen_driver("dup", lambda model=None: DummyImageGen())

    def test_overwrite(self):
        register_img_gen_driver("dup", lambda model=None: DummyImageGen("old"))
        register_img_gen_driver("dup", lambda model=None: DummyImageGen("new"), overwrite=True)
        driver = get_img_gen_driver_factory("dup")()
        assert driver.model == "new"

    def test_unregister(self):
        register_img_gen_driver("temp", lambda model=None: DummyImageGen())
        assert unregister_img_gen_driver("temp")
        assert not is_img_gen_driver_registered("temp")
        assert not unregister_img_gen_driver("temp")  # already gone

    def test_list(self):
        register_img_gen_driver("b_ig", lambda model=None: DummyImageGen())
        register_img_gen_driver("a_ig", lambda model=None: DummyImageGen())
        result = list_registered_img_gen_drivers()
        assert result == ["a_ig", "b_ig"]

    def test_lookup_missing_raises(self):
        with pytest.raises(ValueError, match="Unsupported image gen"):
            get_img_gen_driver_factory("nonexistent")


# ── Async Image Gen Registration ─────────────────────────────────────────


class TestAsyncImageGenRegistration:
    def test_register_and_lookup(self):
        register_async_img_gen_driver("test_async", lambda model=None: DummyImageGen(model))
        assert is_async_img_gen_driver_registered("test_async")
        factory = get_async_img_gen_driver_factory("test_async")
        driver = factory("dall-e-3")
        assert driver.model == "dall-e-3"

    def test_duplicate_raises(self):
        register_async_img_gen_driver("dup", lambda model=None: DummyImageGen())
        with pytest.raises(ValueError, match="already registered"):
            register_async_img_gen_driver("dup", lambda model=None: DummyImageGen())

    def test_unregister(self):
        register_async_img_gen_driver("temp", lambda model=None: DummyImageGen())
        assert unregister_async_img_gen_driver("temp")
        assert not is_async_img_gen_driver_registered("temp")

    def test_list(self):
        register_async_img_gen_driver("z_async", lambda model=None: DummyImageGen())
        register_async_img_gen_driver("a_async", lambda model=None: DummyImageGen())
        result = list_registered_async_img_gen_drivers()
        assert result == ["a_async", "z_async"]


# ── ImageCostMixin ───────────────────────────────────────────────────────


class TestImageCostMixin:
    def test_size_quality_lookup(self):
        from prompture.infra.cost_mixin import ImageCostMixin

        class TestDriver(ImageCostMixin):
            IMAGE_PRICING = {
                "dall-e-3": {
                    "1024x1024/standard": 0.04,
                    "1024x1024/hd": 0.08,
                },
            }

        driver = TestDriver()
        cost = driver._calculate_image_cost("openai", "dall-e-3", size="1024x1024", quality="standard", n=1)
        assert cost == pytest.approx(0.04, abs=1e-6)

    def test_hd_quality(self):
        from prompture.infra.cost_mixin import ImageCostMixin

        class TestDriver(ImageCostMixin):
            IMAGE_PRICING = {
                "dall-e-3": {
                    "1024x1024/standard": 0.04,
                    "1024x1024/hd": 0.08,
                },
            }

        driver = TestDriver()
        cost = driver._calculate_image_cost("openai", "dall-e-3", size="1024x1024", quality="hd", n=1)
        assert cost == pytest.approx(0.08, abs=1e-6)

    def test_multiple_images(self):
        from prompture.infra.cost_mixin import ImageCostMixin

        class TestDriver(ImageCostMixin):
            IMAGE_PRICING = {"dall-e-2": {"1024x1024": 0.020}}

        driver = TestDriver()
        cost = driver._calculate_image_cost("openai", "dall-e-2", size="1024x1024", n=5)
        assert cost == pytest.approx(0.10, abs=1e-6)

    def test_default_fallback(self):
        from prompture.infra.cost_mixin import ImageCostMixin

        class TestDriver(ImageCostMixin):
            IMAGE_PRICING = {"grok-2-image": {"default": 0.07}}

        driver = TestDriver()
        cost = driver._calculate_image_cost("grok", "grok-2-image", n=2)
        assert cost == pytest.approx(0.14, abs=1e-6)

    def test_unknown_model_zero_cost(self):
        from prompture.infra.cost_mixin import ImageCostMixin

        class TestDriver(ImageCostMixin):
            IMAGE_PRICING = {}

        driver = TestDriver()
        cost = driver._calculate_image_cost("openai", "unknown-model", n=3)
        assert cost == 0.0

    def test_size_only_fallback(self):
        from prompture.infra.cost_mixin import ImageCostMixin

        class TestDriver(ImageCostMixin):
            IMAGE_PRICING = {"dall-e-2": {"512x512": 0.018}}

        driver = TestDriver()
        # quality "standard" not in pricing, falls back to size-only key
        cost = driver._calculate_image_cost("openai", "dall-e-2", size="512x512", quality="standard", n=1)
        assert cost == pytest.approx(0.018, abs=1e-6)


# ── Base Class Contracts ─────────────────────────────────────────────────


class TestImageGenBaseContract:
    def test_sync_generate_image_raises(self):
        from prompture.drivers.img_gen_base import ImageGenDriver

        driver = ImageGenDriver()
        with pytest.raises(NotImplementedError):
            driver.generate_image("a cat", {})

    def test_async_generate_image_raises(self):
        from prompture.drivers.async_img_gen_base import AsyncImageGenDriver

        driver = AsyncImageGenDriver()
        with pytest.raises(NotImplementedError):
            asyncio.run(driver.generate_image("a cat", {}))

    def test_sync_class_attributes(self):
        from prompture.drivers.img_gen_base import ImageGenDriver

        driver = ImageGenDriver()
        assert driver.supports_multiple is True
        assert driver.supports_size_variants is True
        assert driver.max_images == 10
        assert driver.callbacks is None

    def test_async_class_attributes(self):
        from prompture.drivers.async_img_gen_base import AsyncImageGenDriver

        driver = AsyncImageGenDriver()
        assert driver.supports_multiple is True
        assert driver.supports_size_variants is True
        assert driver.max_images == 10
        assert driver.callbacks is None


# ── Hook Wrappers ────────────────────────────────────────────────────────


class TestImageGenHooks:
    def test_sync_generate_with_hooks_fires_callbacks(self):
        from prompture.drivers.img_gen_base import ImageGenDriver
        from prompture.infra.callbacks import DriverCallbacks
        from prompture.media.image import image_from_base64

        calls = []

        class TestIG(ImageGenDriver):
            def generate_image(self, prompt, options):
                return {
                    "images": [image_from_base64("AAAA", "image/png")],
                    "meta": {"image_count": 1, "cost": 0.04},
                }

        cb = DriverCallbacks(
            on_request=lambda p: calls.append(("request", p)),
            on_response=lambda p: calls.append(("response", p)),
        )
        driver = TestIG()
        driver.callbacks = cb

        result = driver.generate_image_with_hooks("a cat", {})
        assert result["meta"]["image_count"] == 1
        assert len(calls) == 2
        assert calls[0][0] == "request"
        assert calls[0][1]["prompt_length"] == 5
        assert calls[1][0] == "response"
        assert calls[1][1]["image_count"] == 1

    def test_sync_generate_with_hooks_fires_on_error(self):
        from prompture.drivers.img_gen_base import ImageGenDriver
        from prompture.infra.callbacks import DriverCallbacks

        calls = []

        class FailingIG(ImageGenDriver):
            def generate_image(self, prompt, options):
                raise RuntimeError("API error")

        cb = DriverCallbacks(
            on_request=lambda p: calls.append(("request", p)),
            on_error=lambda p: calls.append(("error", p)),
        )
        driver = FailingIG()
        driver.callbacks = cb

        with pytest.raises(RuntimeError, match="API error"):
            driver.generate_image_with_hooks("a cat", {})
        assert len(calls) == 2
        assert calls[0][0] == "request"
        assert calls[1][0] == "error"

    def test_async_generate_with_hooks_fires_callbacks(self):
        from prompture.drivers.async_img_gen_base import AsyncImageGenDriver
        from prompture.infra.callbacks import DriverCallbacks
        from prompture.media.image import image_from_base64

        calls = []

        class TestAsyncIG(AsyncImageGenDriver):
            async def generate_image(self, prompt, options):
                return {
                    "images": [image_from_base64("AAAA", "image/png")],
                    "meta": {"image_count": 1, "cost": 0.04},
                }

        cb = DriverCallbacks(
            on_request=lambda p: calls.append(("request", p)),
            on_response=lambda p: calls.append(("response", p)),
        )
        driver = TestAsyncIG()
        driver.callbacks = cb

        result = asyncio.run(driver.generate_image_with_hooks("a cat", {}))
        assert result["meta"]["image_count"] == 1
        assert len(calls) == 2
        assert calls[0][0] == "request"
        assert calls[1][0] == "response"


# ── Concrete Driver Attributes ───────────────────────────────────────────


class TestConcreteDriverAttributes:
    def test_openai_pricing_keys(self):
        from prompture.drivers.openai_img_gen_driver import OpenAIImageGenDriver

        assert "dall-e-3" in OpenAIImageGenDriver.IMAGE_PRICING
        assert "dall-e-2" in OpenAIImageGenDriver.IMAGE_PRICING
        assert "1024x1024/standard" in OpenAIImageGenDriver.IMAGE_PRICING["dall-e-3"]
        assert "1024x1024/hd" in OpenAIImageGenDriver.IMAGE_PRICING["dall-e-3"]

    def test_google_pricing_keys(self):
        from prompture.drivers.google_img_gen_driver import GoogleImageGenDriver

        assert "imagen-3.0-generate-002" in GoogleImageGenDriver.IMAGE_PRICING
        assert "imagen-3.0-fast-generate-001" in GoogleImageGenDriver.IMAGE_PRICING

    def test_stability_pricing_keys(self):
        from prompture.drivers.stability_img_gen_driver import StabilityImageGenDriver

        assert "stable-image-core" in StabilityImageGenDriver.IMAGE_PRICING
        assert "sd3.5-large" in StabilityImageGenDriver.IMAGE_PRICING

    def test_grok_pricing_keys(self):
        from prompture.drivers.grok_img_gen_driver import GrokImageGenDriver

        assert "grok-2-image" in GrokImageGenDriver.IMAGE_PRICING

    def test_async_drivers_share_pricing(self):
        from prompture.drivers.async_openai_img_gen_driver import AsyncOpenAIImageGenDriver
        from prompture.drivers.openai_img_gen_driver import OpenAIImageGenDriver

        assert AsyncOpenAIImageGenDriver.IMAGE_PRICING is OpenAIImageGenDriver.IMAGE_PRICING

    def test_openai_supported_sizes(self):
        from prompture.drivers.openai_img_gen_driver import OpenAIImageGenDriver

        assert "1024x1024" in OpenAIImageGenDriver.supported_sizes
        assert "1792x1024" in OpenAIImageGenDriver.supported_sizes

    def test_stability_aspect_ratios(self):
        from prompture.drivers.stability_img_gen_driver import StabilityImageGenDriver

        assert "1:1" in StabilityImageGenDriver.supported_sizes
        assert "16:9" in StabilityImageGenDriver.supported_sizes
        assert StabilityImageGenDriver.supports_multiple is False
