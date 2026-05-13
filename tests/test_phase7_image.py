"""Phase 7: Ideogram and Black Forest Labs (BFL / Flux) image generation tests.

All HTTP calls are mocked. Integration tests against the real APIs would
require credentials and are not included.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from prompture.drivers.async_bfl_img_gen_driver import AsyncBFLImageGenDriver
from prompture.drivers.async_ideogram_img_gen_driver import AsyncIdeogramImageGenDriver
from prompture.drivers.bfl_img_gen_driver import (
    BFLImageGenDriver,
)
from prompture.drivers.bfl_img_gen_driver import (
    _build_body as bfl_build_body,
)
from prompture.drivers.ideogram_img_gen_driver import (
    IdeogramImageGenDriver,
    _normalize_aspect_ratio,
)
from prompture.drivers.img_gen_registry import (
    get_async_img_gen_driver_for_model,
    get_img_gen_driver_for_model,
)

# ── helpers ─────────────────────────────────────────────────────────


def _mock_response(*, status_code: int = 200, json_data: Any = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json = MagicMock(return_value=json_data if json_data is not None else {})
    return resp


class _FakeSyncClient:
    def __init__(self, *, post_responses=None, get_responses=None):
        self._post_responses = list(post_responses or [])
        self._get_responses = list(get_responses or [])
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, **kwargs):
        self.post_calls.append({"url": url, **kwargs})
        if not self._post_responses:
            raise AssertionError("No more post responses queued")
        return self._post_responses.pop(0)

    def get(self, url, **kwargs):
        self.get_calls.append({"url": url, **kwargs})
        if not self._get_responses:
            raise AssertionError("No more get responses queued")
        return self._get_responses.pop(0)


class _FakeAsyncClient:
    def __init__(self, *, post_responses=None, get_responses=None):
        self._post_responses = list(post_responses or [])
        self._get_responses = list(get_responses or [])
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.post_calls.append({"url": url, **kwargs})
        if not self._post_responses:
            raise AssertionError("No more post responses queued")
        return self._post_responses.pop(0)

    async def get(self, url, **kwargs):
        self.get_calls.append({"url": url, **kwargs})
        if not self._get_responses:
            raise AssertionError("No more get responses queued")
        return self._get_responses.pop(0)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make polling instant in tests."""
    monkeypatch.setattr("prompture.drivers.bfl_img_gen_driver.time.sleep", lambda *_: None)

    async def _async_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr("prompture.drivers.async_bfl_img_gen_driver.asyncio.sleep", _async_sleep)


# ── Ideogram helpers ────────────────────────────────────────────────


class TestIdeogramHelpers:
    def test_normalize_aspect_ratio_colon_to_x(self):
        assert _normalize_aspect_ratio("16:9") == "16x9"
        assert _normalize_aspect_ratio("1x1") == "1x1"

    def test_build_form_basic(self):
        d = IdeogramImageGenDriver(api_key="k")
        form, path = d._build_form("a cat", {}, "ideogram-v3")
        assert form["prompt"] == "a cat"
        assert form["aspect_ratio"] == "1x1"
        assert form["rendering_speed"] == "DEFAULT"
        assert form["num_images"] == "1"
        assert path == "/v1/ideogram-v3/generate"

    def test_build_form_aspect_ratio_propagates(self):
        d = IdeogramImageGenDriver(api_key="k")
        form, _ = d._build_form("hi", {"aspect_ratio": "16x9"}, "ideogram-v3")
        assert form["aspect_ratio"] == "16x9"

    def test_build_form_invalid_ratio(self):
        d = IdeogramImageGenDriver(api_key="k")
        with pytest.raises(ValueError, match="aspect_ratio"):
            d._build_form("hi", {"aspect_ratio": "7x11"}, "ideogram-v3")

    def test_build_form_invalid_speed(self):
        d = IdeogramImageGenDriver(api_key="k")
        with pytest.raises(ValueError, match="rendering_speed"):
            d._build_form("hi", {"rendering_speed": "MAGIC"}, "ideogram-v3")

    def test_legacy_model_raises_not_implemented(self):
        d = IdeogramImageGenDriver(api_key="k")
        with pytest.raises(NotImplementedError, match="legacy"):
            d._build_form("hi", {}, "ideogram-v2")


# ── Ideogram sync driver ────────────────────────────────────────────


class TestIdeogramSyncDriver:
    def test_construct_defaults(self):
        d = IdeogramImageGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.model == "ideogram-v3"
        assert d.endpoint == "https://api.ideogram.ai"

    def test_known_models(self):
        for m in ("ideogram-v3", "ideogram-v2", "ideogram-v2-turbo"):
            assert m in IdeogramImageGenDriver.KNOWN_MODELS

    def test_missing_key_raises(self):
        d = IdeogramImageGenDriver(api_key=None)
        with pytest.raises(RuntimeError, match="IDEOGRAM_API_KEY"):
            d.generate_image("x", {})

    def test_generate_returns_image_url(self):
        client = _FakeSyncClient(
            post_responses=[
                _mock_response(
                    json_data={
                        "created": "2026-05-13T12:00:00Z",
                        "data": [
                            {
                                "url": "https://cdn.ideogram.ai/a.png",
                                "is_image_safe": True,
                                "prompt": "a cat sitting on a wall",
                            }
                        ],
                    }
                )
            ],
        )
        with patch("prompture.drivers.ideogram_img_gen_driver.httpx.Client", return_value=client):
            d = IdeogramImageGenDriver(api_key="k")
            result = d.generate_image("a cat", {})

        assert result["meta"]["image_count"] == 1
        assert result["meta"]["model_name"] == "ideogram/ideogram-v3"
        assert result["meta"]["revised_prompt"] == "a cat sitting on a wall"
        assert len(result["images"]) == 1
        assert result["images"][0].url == "https://cdn.ideogram.ai/a.png"

        post = client.post_calls[0]
        assert post["url"].endswith("/v1/ideogram-v3/generate")
        assert post["data"]["prompt"] == "a cat"
        assert post["data"]["aspect_ratio"] == "1x1"
        assert post["headers"]["Api-Key"] == "k"

    def test_aspect_ratio_propagates_to_request(self):
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"created": "now", "data": [{"url": "https://x/a.png"}]})]
        )
        with patch("prompture.drivers.ideogram_img_gen_driver.httpx.Client", return_value=client):
            d = IdeogramImageGenDriver(api_key="k")
            d.generate_image("hi", {"aspect_ratio": "16x9", "rendering_speed": "TURBO", "num_images": 2})

        sent = client.post_calls[0]["data"]
        assert sent["aspect_ratio"] == "16x9"
        assert sent["rendering_speed"] == "TURBO"
        assert sent["num_images"] == "2"

    def test_http_error_raises_with_body(self):
        client = _FakeSyncClient(post_responses=[_mock_response(status_code=401, text="unauthorized", json_data=None)])
        with patch("prompture.drivers.ideogram_img_gen_driver.httpx.Client", return_value=client):
            d = IdeogramImageGenDriver(api_key="bad")
            with pytest.raises(RuntimeError, match="unauthorized"):
                d.generate_image("x", {})


# ── Ideogram async smoke ────────────────────────────────────────────


class TestIdeogramAsyncDriver:
    def test_construct(self):
        d = AsyncIdeogramImageGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.model == "ideogram-v3"
        assert d.KNOWN_MODELS == IdeogramImageGenDriver.KNOWN_MODELS

    def test_async_generate(self):
        client = _FakeAsyncClient(
            post_responses=[_mock_response(json_data={"created": "now", "data": [{"url": "https://x/b.png"}]})]
        )
        with patch("prompture.drivers.async_ideogram_img_gen_driver.httpx.AsyncClient", return_value=client):
            d = AsyncIdeogramImageGenDriver(api_key="k")
            result = asyncio.run(d.generate_image("hello", {"aspect_ratio": "16x9"}))

        assert result["meta"]["model_name"] == "ideogram/ideogram-v3"
        assert result["images"][0].url == "https://x/b.png"
        sent = client.post_calls[0]["data"]
        assert sent["aspect_ratio"] == "16x9"


# ── BFL helpers ─────────────────────────────────────────────────────


class TestBFLBuildBody:
    def test_basic_flux_pro(self):
        body = bfl_build_body("a cat", {}, "flux-pro-1.1")
        assert body["prompt"] == "a cat"
        assert body["width"] == 1024
        assert body["height"] == 1024
        assert body["prompt_upsampling"] is False
        assert body["safety_tolerance"] == 2
        assert body["output_format"] == "jpeg"

    def test_ultra_supports_aspect_ratio(self):
        body = bfl_build_body("x", {"aspect_ratio": "16:9", "raw": True}, "flux-pro-1.1-ultra")
        assert body["aspect_ratio"] == "16:9"
        assert body["raw"] is True

    def test_kontext_with_input_image(self):
        body = bfl_build_body(
            "make it blue", {"input_image": "iVBORw0KGgo=", "aspect_ratio": "1:1"}, "flux-kontext-pro"
        )
        assert body["prompt"] == "make it blue"
        assert body["input_image"] == "iVBORw0KGgo="
        assert body["aspect_ratio"] == "1:1"
        # Kontext does NOT carry width/height.
        assert "width" not in body
        assert "height" not in body


# ── BFL sync driver ─────────────────────────────────────────────────


class TestBFLSyncDriver:
    def test_construct_defaults(self):
        d = BFLImageGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.model == "flux-pro-1.1"
        assert d.endpoint == "https://api.bfl.ai"

    def test_known_models(self):
        for m in (
            "flux-pro-1.1",
            "flux-pro-1.1-ultra",
            "flux-pro",
            "flux-dev",
            "flux-schnell",
            "flux-kontext-pro",
            "flux-kontext-max",
        ):
            assert m in BFLImageGenDriver.KNOWN_MODELS

    def test_missing_key_raises(self):
        d = BFLImageGenDriver(api_key=None)
        with pytest.raises(RuntimeError, match="BFL_API_KEY"):
            d.generate_image("x", {})

    def test_full_flow_polls_until_ready(self):
        polling_url = "https://api.bfl.ai/v1/get_result?id=req-1"
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"id": "req-1", "polling_url": polling_url})],
            get_responses=[
                _mock_response(json_data={"status": "Pending"}),
                _mock_response(json_data={"status": "Pending"}),
                _mock_response(
                    json_data={
                        "status": "Ready",
                        "result": {"sample": "https://cdn.bfl.ai/sample.jpg"},
                    }
                ),
            ],
        )
        with patch("prompture.drivers.bfl_img_gen_driver.httpx.Client", return_value=client):
            d = BFLImageGenDriver(api_key="k")
            result = d.generate_image("a dragon", {"poll_interval": 0.0, "max_retries": 5})

        assert result["meta"]["model_name"] == "bfl/flux-pro-1.1"
        assert result["meta"]["request_id"] == "req-1"
        assert result["meta"]["image_count"] == 1
        assert result["images"][0].url == "https://cdn.bfl.ai/sample.jpg"
        # Submit went to /v1/<model>
        post = client.post_calls[0]
        assert post["url"].endswith("/v1/flux-pro-1.1")
        assert post["headers"]["x-key"] == "k"
        # The polling URL was reused as-is.
        assert all(c["url"] == polling_url for c in client.get_calls)

    def test_ultra_uses_correct_endpoint(self):
        polling_url = "https://api.bfl.ai/v1/get_result?id=ultra-1"
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"id": "ultra-1", "polling_url": polling_url})],
            get_responses=[
                _mock_response(json_data={"status": "Ready", "result": {"sample": "https://cdn.bfl.ai/u.jpg"}}),
            ],
        )
        with patch("prompture.drivers.bfl_img_gen_driver.httpx.Client", return_value=client):
            d = BFLImageGenDriver(api_key="k", model="flux-pro-1.1-ultra")
            d.generate_image("x", {"poll_interval": 0.0, "aspect_ratio": "21:9"})

        post = client.post_calls[0]
        assert post["url"].endswith("/v1/flux-pro-1.1-ultra")
        assert post["json"]["aspect_ratio"] == "21:9"

    def test_kontext_input_image_propagates(self):
        polling_url = "https://api.bfl.ai/v1/get_result?id=k-1"
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"id": "k-1", "polling_url": polling_url})],
            get_responses=[
                _mock_response(json_data={"status": "Ready", "result": {"sample": "https://cdn.bfl.ai/k.jpg"}}),
            ],
        )
        with patch("prompture.drivers.bfl_img_gen_driver.httpx.Client", return_value=client):
            d = BFLImageGenDriver(api_key="k", model="flux-kontext-pro")
            d.generate_image(
                "make it neon",
                {"input_image": "iVBORw0KGgoAAAANSUhEUgAAA==", "poll_interval": 0.0},
            )

        post = client.post_calls[0]
        assert post["url"].endswith("/v1/flux-kontext-pro")
        body = post["json"]
        assert body["input_image"].startswith("iVBOR")
        assert "width" not in body

    def test_polling_timeout_raises(self):
        polling_url = "https://api.bfl.ai/v1/get_result?id=slow"
        polls = [_mock_response(json_data={"status": "Pending"}) for _ in range(5)]
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"id": "slow", "polling_url": polling_url})],
            get_responses=polls,
        )
        with patch("prompture.drivers.bfl_img_gen_driver.httpx.Client", return_value=client):
            d = BFLImageGenDriver(api_key="k")
            with pytest.raises(TimeoutError, match="timed out"):
                d.generate_image("x", {"poll_interval": 0.0, "max_retries": 5})

    def test_error_status_raises(self):
        polling_url = "https://api.bfl.ai/v1/get_result?id=err"
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"id": "err", "polling_url": polling_url})],
            get_responses=[_mock_response(json_data={"status": "Content Moderated", "result": None})],
        )
        with patch("prompture.drivers.bfl_img_gen_driver.httpx.Client", return_value=client):
            d = BFLImageGenDriver(api_key="k")
            with pytest.raises(RuntimeError, match="Content Moderated"):
                d.generate_image("x", {"poll_interval": 0.0})

    def test_submit_http_error_includes_body(self):
        client = _FakeSyncClient(post_responses=[_mock_response(status_code=400, text="bad prompt", json_data=None)])
        with patch("prompture.drivers.bfl_img_gen_driver.httpx.Client", return_value=client):
            d = BFLImageGenDriver(api_key="k")
            with pytest.raises(RuntimeError, match="bad prompt"):
                d.generate_image("x", {})


# ── BFL async smoke ─────────────────────────────────────────────────


class TestBFLAsyncDriver:
    def test_construct(self):
        d = AsyncBFLImageGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.model == "flux-pro-1.1"
        assert d.KNOWN_MODELS == BFLImageGenDriver.KNOWN_MODELS

    def test_async_full_flow(self):
        polling_url = "https://api.bfl.ai/v1/get_result?id=A"
        client = _FakeAsyncClient(
            post_responses=[_mock_response(json_data={"id": "A", "polling_url": polling_url})],
            get_responses=[
                _mock_response(json_data={"status": "Pending"}),
                _mock_response(json_data={"status": "Ready", "result": {"sample": "https://cdn.bfl.ai/A.jpg"}}),
            ],
        )
        with patch("prompture.drivers.async_bfl_img_gen_driver.httpx.AsyncClient", return_value=client):
            d = AsyncBFLImageGenDriver(api_key="k")
            result = asyncio.run(d.generate_image("hi", {"poll_interval": 0.0, "max_retries": 5}))

        assert result["meta"]["model_name"] == "bfl/flux-pro-1.1"
        assert result["images"][0].url == "https://cdn.bfl.ai/A.jpg"


# ── Registry routing ────────────────────────────────────────────────


class TestRegistryRouting:
    def test_ideogram_routes_to_driver(self, monkeypatch):
        monkeypatch.setenv("IDEOGRAM_API_KEY", "test")
        d = get_img_gen_driver_for_model("ideogram/ideogram-v3")
        assert isinstance(d, IdeogramImageGenDriver)

    def test_bfl_routes_to_driver(self, monkeypatch):
        monkeypatch.setenv("BFL_API_KEY", "test")
        d = get_img_gen_driver_for_model("bfl/flux-pro-1.1")
        assert isinstance(d, BFLImageGenDriver)

    def test_flux_alias_routes_to_bfl(self, monkeypatch):
        monkeypatch.setenv("BFL_API_KEY", "test")
        d = get_img_gen_driver_for_model("flux/flux-dev")
        assert isinstance(d, BFLImageGenDriver)

    def test_ideogram_async_routes_to_driver(self, monkeypatch):
        monkeypatch.setenv("IDEOGRAM_API_KEY", "test")
        d = get_async_img_gen_driver_for_model("ideogram/ideogram-v3")
        assert isinstance(d, AsyncIdeogramImageGenDriver)

    def test_bfl_async_routes_to_driver(self, monkeypatch):
        monkeypatch.setenv("BFL_API_KEY", "test")
        d = get_async_img_gen_driver_for_model("bfl/flux-pro-1.1")
        assert isinstance(d, AsyncBFLImageGenDriver)
