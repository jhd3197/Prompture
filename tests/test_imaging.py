"""Tests for prompture.imaging — provider-agnostic image prompt helpers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver
from prompture.imaging import (
    DEFAULT_STYLE_PRESETS,
    ImageSetPlan,
    ImageSpec,
    StylePreset,
    aenhance_image_prompt,
    aplan_image_set,
    compose_negative_prompt,
    enhance_image_prompt,
    get_style_preset,
    list_style_presets,
    model_supports_negative_prompt,
    plan_image_set,
    register_style_preset,
)


class _JsonDriver(Driver):
    """Sync driver that always returns a fixed JSON string."""

    supports_json_mode = True
    supports_json_schema = False
    supports_messages = True

    def __init__(self, text: str):
        self._text = text
        self.model = "mock-model"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._resp()

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._resp()

    def _resp(self) -> dict[str, Any]:
        return {"text": self._text, "meta": {"cost": 0.0, "model_name": "mock-model", "raw_response": {}}}


class _AsyncJsonDriver(AsyncDriver):
    """Async driver that always returns a fixed JSON string."""

    supports_json_mode = True
    supports_json_schema = False
    supports_messages = True

    def __init__(self, text: str):
        self._text = text
        self.model = "mock-async-model"

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._resp()

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._resp()

    def _resp(self) -> dict[str, Any]:
        return {"text": self._text, "meta": {"cost": 0.0, "model_name": "mock-async-model", "raw_response": {}}}


class _RaisingDriver(Driver):
    supports_json_mode = True
    supports_json_schema = False
    supports_messages = True

    def __init__(self):
        self.model = "boom"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("driver down")

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("driver down")


# --------------------------------------------------------------------------
# StylePreset
# --------------------------------------------------------------------------


class TestStylePreset:
    def test_apply_wraps_and_collapses_whitespace(self):
        p = StylePreset(id="t", label="T", prompt_prefix="A photo of", prompt_suffix="in bright light")
        prompt, opts = p.apply("  a   cat ")
        assert prompt == "A photo of a cat in bright light"
        assert opts == {}

    def test_apply_merges_params_over_options(self):
        p = StylePreset(id="t2", label="T2", params={"quality": "hd"})
        prompt, opts = p.apply("sunset", {"quality": "standard", "n": 2})
        assert prompt == "sunset"
        assert opts == {"quality": "hd", "n": 2}  # preset wins on collision, others preserved

    def test_apply_does_not_mutate_input_options(self):
        p = StylePreset(id="t3", label="T3", params={"quality": "hd"})
        original = {"n": 1}
        p.apply("x", original)
        assert original == {"n": 1}

    def test_default_library_registered_and_vendor_neutral(self):
        ids = {p.id for p in DEFAULT_STYLE_PRESETS}
        assert {"photoreal", "cinematic", "flat_vector"} <= ids
        for p in DEFAULT_STYLE_PRESETS:
            assert get_style_preset(p.id) is p
        assert set(ids) <= {p.id for p in list_style_presets()}

    def test_register_overwrite_guard(self):
        register_style_preset(StylePreset(id="custom_x", label="X"))
        with pytest.raises(ValueError):
            register_style_preset(StylePreset(id="custom_x", label="X2"))
        register_style_preset(StylePreset(id="custom_x", label="X2"), overwrite=True)
        assert get_style_preset("custom_x").label == "X2"


# --------------------------------------------------------------------------
# Negative-prompt composition
# --------------------------------------------------------------------------


class TestNegativePrompt:
    def test_capability_reads_driver_class(self):
        assert model_supports_negative_prompt("stability/stable-image-core") is True
        assert model_supports_negative_prompt("ideogram/ideogram-v3") is True
        assert model_supports_negative_prompt("fal/fal-ai/flux/dev") is True
        assert model_supports_negative_prompt("openai/gpt-image-1") is False
        assert model_supports_negative_prompt("nonsense/whatever") is False
        assert model_supports_negative_prompt("") is False

    def test_native_route_sets_option(self):
        prompt, opts = compose_negative_prompt("a cat", "blurry, text", native_supported=True)
        assert prompt == "a cat"
        assert opts["negative_prompt"] == "blurry, text"

    def test_native_route_merges_existing(self):
        _prompt, opts = compose_negative_prompt(
            "a cat", "text", {"negative_prompt": "blurry"}, native_supported=True
        )
        assert opts["negative_prompt"] == "blurry, text"

    def test_fold_route_appends_avoid_clause(self):
        prompt, opts = compose_negative_prompt("a cat", "blurry, text", native_supported=False)
        assert prompt == "a cat. Avoid: blurry, text."
        assert "negative_prompt" not in opts

    def test_model_based_routing(self):
        # stability -> native param
        _, opts = compose_negative_prompt("x", "blurry", model="stability/stable-image-core")
        assert opts.get("negative_prompt") == "blurry"
        # openai -> fold into prompt
        prompt, opts = compose_negative_prompt("x", "blurry", model="openai/gpt-image-1")
        assert prompt == "x. Avoid: blurry."
        assert "negative_prompt" not in opts

    def test_empty_negative_is_noop(self):
        prompt, opts = compose_negative_prompt("a cat", "  ", {"n": 1})
        assert prompt == "a cat"
        assert opts == {"n": 1}


# --------------------------------------------------------------------------
# Set planner
# --------------------------------------------------------------------------

_PLAN_JSON = json.dumps(
    {
        "images": [
            {"name": "hero", "prompt": "a wide hero banner", "aspect": "landscape"},
            {"prompt": "a square icon"},  # no name, no aspect -> defaults
            {"prompt": "", "aspect": "portrait"},  # dropped (no prompt)
            {"prompt": "extra", "aspect": "weird"},  # bad aspect -> square
            {"prompt": "overflow"},  # dropped by max_images=3
        ]
    }
)


class TestSetPlanner:
    def test_plan_coerces_clamps_and_defaults(self):
        plan = plan_image_set("a launch set", driver=_JsonDriver(_PLAN_JSON), max_images=3)
        assert isinstance(plan, ImageSetPlan)
        assert plan.brief == "a launch set"
        assert len(plan) == 3
        assert plan.images[0] == ImageSpec(prompt="a wide hero banner", name="hero", aspect="landscape")
        assert plan.images[1].aspect == "square" and plan.images[1].name is None
        assert plan.images[2].aspect == "square"  # bad token normalized
        assert [s.prompt for s in plan] == ["a wide hero banner", "a square icon", "extra"]

    @pytest.mark.asyncio
    async def test_aplan_matches_sync(self):
        plan = await aplan_image_set("set", driver=_AsyncJsonDriver(_PLAN_JSON), max_images=2)
        assert len(plan) == 2
        assert plan.images[0].prompt == "a wide hero banner"


# --------------------------------------------------------------------------
# Prompt enhancer
# --------------------------------------------------------------------------


class TestEnhancer:
    def test_enhance_returns_rewritten(self):
        driver = _JsonDriver(json.dumps({"enhanced_prompt": "a richly detailed cat at golden hour"}))
        out = enhance_image_prompt("a cat", driver=driver)
        assert out == "a richly detailed cat at golden hour"

    def test_enhance_accepts_style_preset(self):
        driver = _JsonDriver(json.dumps({"enhanced_prompt": "styled cat"}))
        out = enhance_image_prompt("a cat", driver=driver, style="cinematic")
        assert out == "styled cat"

    def test_enhance_empty_prompt_is_noop(self):
        out = enhance_image_prompt("   ", driver=_JsonDriver("{}"))
        assert out == "   "

    def test_enhance_returns_original_on_failure(self):
        out = enhance_image_prompt("a cat", driver=_RaisingDriver())
        assert out == "a cat"

    def test_enhance_returns_original_on_empty_result(self):
        out = enhance_image_prompt("a cat", driver=_JsonDriver(json.dumps({"enhanced_prompt": ""})))
        assert out == "a cat"

    @pytest.mark.asyncio
    async def test_aenhance_returns_rewritten(self):
        driver = _AsyncJsonDriver(json.dumps({"enhanced_prompt": "vivid cat"}))
        out = await aenhance_image_prompt("a cat", driver=driver)
        assert out == "vivid cat"
