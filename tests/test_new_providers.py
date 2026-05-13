"""Unit tests for Kling, MiniMax/Hailuo, and Fal.ai drivers.

These verify driver construction, header/auth shape, body builders, and
response parsing — all without making live network calls. Integration tests
against the real APIs would require credentials and are not included.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from prompture.drivers.fal_img_gen_driver import (
    AsyncFalImageGenDriver,
    FalImageGenDriver,
    _build_img_input,
    _extract_image_urls,
)
from prompture.drivers.fal_video_gen_driver import (
    AsyncFalVideoGenDriver,
    FalVideoGenDriver,
    _build_video_input,
)
from prompture.drivers.async_kling_img_gen_driver import AsyncKlingImageGenDriver
from prompture.drivers.kling_img_gen_driver import (
    KlingImageGenDriver,
    generate_kling_jwt,
)
from prompture.drivers.kling_video_gen_driver import (
    AsyncKlingVideoGenDriver,
    KlingVideoGenDriver,
    _build_video_body,
)
from prompture.drivers.minimax_driver import MiniMaxDriver
from prompture.drivers.minimax_video_gen_driver import (
    AsyncMiniMaxVideoGenDriver,
    MiniMaxVideoGenDriver,
    _build_body,
    _duration,
    _resolution,
    _select_model,
)


# ───────────────────────── Kling ────────────────────────────


class TestKlingJWT:
    def test_jwt_three_segments(self):
        token = generate_kling_jwt("ak", "sk", ttl_seconds=60)
        assert token.count(".") == 2
        h, p, s = token.split(".")
        assert all(part for part in (h, p, s))

    def test_jwt_requires_credentials(self):
        with pytest.raises(RuntimeError):
            generate_kling_jwt("", "sk")


class TestKlingImageGen:
    def test_construct(self):
        d = KlingImageGenDriver(access_key="ak", secret_key="sk")
        assert d.access_key == "ak"
        assert d.model == "kling-v2-1"
        assert d.endpoint.startswith("https://api-singapore.klingai.com")

    def test_build_body_text_to_image(self):
        d = KlingImageGenDriver(access_key="ak", secret_key="sk")
        path, body = d._build_body("a cat", {"aspect_ratio": "16:9", "resolution": "1k"})
        assert path == "/v1/images/generations"
        assert body["model_name"] == "kling-v2-1"
        assert body["prompt"] == "a cat"
        assert body["aspect_ratio"] == "16:9"
        assert body["image_size"] == "1k"

    def test_build_body_v1_5_subject_mode(self):
        d = KlingImageGenDriver(access_key="ak", secret_key="sk", model="kling-v1-5")
        path, body = d._build_body(
            "a cat",
            {"image": "https://example.com/x.png", "reference_mode": "subject"},
        )
        assert body["image"] == "https://example.com/x.png"
        assert body["image_reference"] == "subject"
        assert 0 < body["face_reference_intensity"] <= 1
        assert 0 < body["subject_reference_intensity"] <= 1

    def test_build_body_multi_image(self):
        d = KlingImageGenDriver(access_key="ak", secret_key="sk")
        path, body = d._build_body(
            "compose",
            {"subject_images": ["data:image/png;base64,AAAA", "https://x/y.jpg"]},
        )
        assert path == "/v1/images/multi-image2image"
        assert len(body["subject_image_list"]) == 2
        # data URL prefix should be stripped
        assert body["subject_image_list"][0]["subject_image"] == "AAAA"

    def test_invalid_aspect_ratio(self):
        d = KlingImageGenDriver(access_key="ak", secret_key="sk")
        with pytest.raises(ValueError):
            d._build_body("x", {"aspect_ratio": "999:1"})

    def test_missing_creds_raises_on_call(self):
        d = KlingImageGenDriver(access_key=None, secret_key=None)
        with pytest.raises(RuntimeError, match="KLING_ACCESS_KEY"):
            d.generate_image("hello", {})


class TestKlingVideoGen:
    def test_build_image_to_video(self):
        path, body = _build_video_body(
            "wave", {"image": "https://x/y.png", "duration": 5}, "kling-v2-1"
        )
        assert path == "/v1/videos/image2video"
        assert body["mode"] == "std"
        assert body["duration"] == "5"

    def test_pro_mode_for_last_frame(self):
        path, body = _build_video_body(
            "wave",
            {"image": "https://x/y.png", "last_frame": "https://x/z.png", "duration": 5},
            "kling-v2-1",
        )
        assert body["mode"] == "pro"
        assert "image_tail" in body

    def test_pro_mode_for_v2_6(self):
        path, body = _build_video_body("x", {"duration": 5}, "kling-v2-6")
        assert body["mode"] == "pro"

    def test_multi_image_video(self):
        path, body = _build_video_body(
            "x", {"image_list": ["https://x/1.png", "https://x/2.png"], "duration": 10}, "kling-v2-1"
        )
        assert path == "/v1/videos/multi-image2video"
        assert body["model_name"] == "kling-v1-6"
        assert len(body["image_list"]) == 2

    def test_bad_duration(self):
        with pytest.raises(ValueError):
            _build_video_body("x", {"duration": 7}, "kling-v2-1")

    def test_known_models(self):
        assert "kling-v2-6" in KlingVideoGenDriver.KNOWN_MODELS

    def test_async_class_inherits_metadata(self):
        assert AsyncKlingVideoGenDriver.max_seconds == KlingVideoGenDriver.max_seconds


# ───────────────────────── MiniMax ──────────────────────────


class TestMiniMaxHelpers:
    def test_duration_clamps_to_6_or_10(self):
        assert _duration({"duration": 5}) == 6
        assert _duration({"duration": 8}) == 6
        assert _duration({"duration": 9}) == 10
        assert _duration({}) == 6

    def test_resolution_fallback(self):
        assert _resolution({"resolution": "1080p"}) == "1080P"
        assert _resolution({"resolution": "weird"}) == "768P"

    def test_select_model_fl2v_forces_hailuo_02(self):
        assert _select_model("MiniMax-Hailuo-2.3", True, True) == "MiniMax-Hailuo-02"

    def test_build_body_s2v(self):
        body = _build_body("x", {"subject_image": "https://x/c.png"}, "MiniMax-Hailuo-2.3")
        assert body["model"] == "S2V-01"
        assert body["subject_reference"][0]["type"] == "character"

    def test_build_body_i2v(self):
        body = _build_body(
            "x", {"image": "https://x/img.png", "aspect_ratio": "9:16"}, "MiniMax-Hailuo-2.3"
        )
        assert body["first_frame_image"] == "https://x/img.png"
        assert body["aspect_ratio"] == "9:16"

    def test_build_body_fl2v(self):
        body = _build_body(
            "x",
            {"image": "https://x/1.png", "last_frame": "https://x/2.png"},
            "MiniMax-Hailuo-2.3",
        )
        assert body["model"] == "MiniMax-Hailuo-02"
        assert "last_frame_image" in body

    def test_base64_normalizes_to_data_url(self):
        body = _build_body("x", {"image": "AAAA="}, "MiniMax-Hailuo-2.3")
        assert body["first_frame_image"].startswith("data:image/jpeg;base64,")


class TestMiniMaxVideoDriver:
    def test_construct(self):
        d = MiniMaxVideoGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.endpoint == "https://api.minimax.io/v1"

    def test_missing_key_raises(self):
        d = MiniMaxVideoGenDriver(api_key=None)
        with pytest.raises(RuntimeError):
            d.generate_video("x", {})

    def test_async_construct(self):
        d = AsyncMiniMaxVideoGenDriver(api_key="k")
        assert d.api_key == "k"

    def test_known_models(self):
        assert "MiniMax-Hailuo-02" in MiniMaxVideoGenDriver.KNOWN_MODELS


class TestMiniMaxLLMDriver:
    def test_construct_raises_without_key(self):
        with pytest.raises(ValueError):
            MiniMaxDriver(api_key=None)

    def test_construct(self):
        d = MiniMaxDriver(api_key="k")
        assert d.model == "MiniMax-Text-01"
        assert d.headers["Authorization"] == "Bearer k"
        assert d.base_url == "https://api.minimax.io/v1"


# ───────────────────────── Fal.ai ───────────────────────────


class TestFalHelpers:
    def test_build_video_input_prompt_only(self):
        payload = _build_video_input("a cat", {})
        assert payload == {"prompt": "a cat"}

    def test_build_video_input_image(self):
        payload = _build_video_input("a", {"image": "https://x/y.png", "duration": 5})
        assert payload["image_url"] == "https://x/y.png"
        assert payload["duration"] == "5"

    def test_build_img_input_size_alias(self):
        payload = _build_img_input("a", {"size": "square_hd"})
        assert payload["image_size"] == "square_hd"

    def test_extract_image_urls_list(self):
        urls = _extract_image_urls({"images": [{"url": "u1"}, {"url": "u2"}]})
        assert urls == ["u1", "u2"]

    def test_extract_image_urls_single(self):
        urls = _extract_image_urls({"image": {"url": "u1"}})
        assert urls == ["u1"]

    def test_extract_image_urls_empty(self):
        assert _extract_image_urls({}) == []


class TestFalDrivers:
    def test_fal_video_construct(self):
        d = FalVideoGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.model == "fal-ai/kling-video/v2.6/pro/image-to-video"
        assert d.endpoint == "https://queue.fal.run"

    def test_fal_image_construct(self):
        d = FalImageGenDriver(api_key="k")
        assert d.model == "fal-ai/flux/dev"

    def test_fal_async_construct(self):
        d_v = AsyncFalVideoGenDriver(api_key="k")
        d_i = AsyncFalImageGenDriver(api_key="k")
        assert d_v.api_key == "k"
        assert d_i.api_key == "k"

    def test_missing_key_raises(self):
        d = FalVideoGenDriver(api_key=None)
        with pytest.raises(RuntimeError):
            d.generate_video("x", {})

    def test_headers_use_key_scheme(self):
        d = FalVideoGenDriver(api_key="abc")
        assert d._headers()["Authorization"] == "Key abc"


# ───────────────── Registry integration smoke ───────────────


class TestProviderDescriptors:
    def test_all_three_registered(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        assert "kling" in PROVIDER_DESCRIPTOR_MAP
        assert "minimax" in PROVIDER_DESCRIPTOR_MAP
        assert "fal" in PROVIDER_DESCRIPTOR_MAP
        assert "hailuo" in PROVIDER_DESCRIPTOR_MAP

    def test_kling_has_image_and_video(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        d = PROVIDER_DESCRIPTOR_MAP["kling"]
        assert d.img_gen_sync is not None
        assert d.img_gen_async is not None
        assert d.video_gen_sync is not None
        assert d.video_gen_async is not None

    def test_minimax_has_llm_and_video(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        d = PROVIDER_DESCRIPTOR_MAP["minimax"]
        assert d.llm_sync is not None
        assert d.llm_async is not None
        assert d.video_gen_sync is not None

    def test_hailuo_alias_inherits_video(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        d = PROVIDER_DESCRIPTOR_MAP["hailuo"]
        assert d.alias_for == "minimax"
        assert d.video_gen_sync is not None
        assert d.llm_sync is None  # alias only covers video_gen
