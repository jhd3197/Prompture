"""Phase 6: Luma AI and Pika Labs video generation driver tests.

All HTTP calls are mocked. Integration tests against the real APIs would
require credentials and are not included.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from prompture.drivers.async_luma_video_gen_driver import AsyncLumaVideoGenDriver
from prompture.drivers.async_pika_video_gen_driver import AsyncPikaVideoGenDriver
from prompture.drivers.luma_video_gen_driver import (
    LumaVideoGenDriver,
)
from prompture.drivers.luma_video_gen_driver import (
    _build_body as luma_build_body,
)
from prompture.drivers.pika_video_gen_driver import (
    PikaVideoGenDriver,
)
from prompture.drivers.pika_video_gen_driver import (
    _build_body as pika_build_body,
)
from prompture.drivers.video_gen_registry import (
    get_async_video_gen_driver_for_model,
    get_video_gen_driver_for_model,
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
    monkeypatch.setattr("prompture.drivers.luma_video_gen_driver.time.sleep", lambda *_: None)
    monkeypatch.setattr("prompture.drivers.pika_video_gen_driver.time.sleep", lambda *_: None)

    async def _async_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr("prompture.drivers.async_luma_video_gen_driver.asyncio.sleep", _async_sleep)
    monkeypatch.setattr("prompture.drivers.async_pika_video_gen_driver.asyncio.sleep", _async_sleep)


# ── Luma helpers ────────────────────────────────────────────────────


class TestLumaBuildBody:
    def test_basic_text_to_video(self):
        body = luma_build_body("a cat", {}, "ray-2")
        assert body["prompt"] == "a cat"
        assert body["model"] == "ray-2"
        assert body["aspect_ratio"] == "16:9"
        assert body["loop"] is False
        assert "keyframes" not in body

    def test_image_to_video_keyframes(self):
        body = luma_build_body("hi", {"image": "https://x/y.png"}, "ray-2")
        assert body["keyframes"] == {"frame0": {"type": "image", "url": "https://x/y.png"}}

    def test_image_to_video_with_end(self):
        body = luma_build_body("hi", {"image": "https://x/a.png", "end_image": "https://x/b.png"}, "ray-2")
        assert body["keyframes"]["frame0"]["url"] == "https://x/a.png"
        assert body["keyframes"]["frame1"]["url"] == "https://x/b.png"

    def test_invalid_aspect_ratio_raises(self):
        with pytest.raises(ValueError):
            luma_build_body("x", {"aspect_ratio": "7:11"}, "ray-2")

    def test_unknown_model_warns_but_returns(self):
        body = luma_build_body("x", {}, "ray-xyz")
        assert body["model"] == "ray-xyz"


# ── Luma sync driver ────────────────────────────────────────────────


class TestLumaVideoDriver:
    def test_construct(self):
        d = LumaVideoGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.model == "ray-2"
        assert d.endpoint == "https://api.lumalabs.ai"

    def test_missing_key_raises(self):
        d = LumaVideoGenDriver(api_key=None)
        with pytest.raises(RuntimeError, match="LUMA_API_KEY"):
            d.generate_video("x", {})

    def test_known_models(self):
        for m in ("ray-2", "ray-flash-2", "ray-1-6"):
            assert m in LumaVideoGenDriver.KNOWN_MODELS

    def test_text_to_video_full_flow(self):
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"id": "task-1", "state": "queued", "assets": None})],
            get_responses=[
                _mock_response(json_data={"id": "task-1", "state": "queued", "assets": None}),
                _mock_response(json_data={"id": "task-1", "state": "processing", "assets": None}),
                _mock_response(
                    json_data={
                        "id": "task-1",
                        "state": "completed",
                        "assets": {"video": "https://cdn.luma.ai/result.mp4"},
                    }
                ),
            ],
        )
        with patch("prompture.drivers.luma_video_gen_driver.httpx.Client", return_value=client):
            d = LumaVideoGenDriver(api_key="k")
            result = d.generate_video("a dragon", {"poll_interval": 0.0})

        assert result["meta"]["task_id"] == "task-1"
        assert result["meta"]["status"] == "completed"
        assert result["meta"]["model_name"] == "luma/ray-2"
        assert len(result["videos"]) == 1
        assert result["videos"][0].url == "https://cdn.luma.ai/result.mp4"
        # POST body matches the documented Luma shape
        post = client.post_calls[0]
        assert post["url"].endswith("/dream-machine/v1/generations")
        assert post["json"]["prompt"] == "a dragon"
        assert post["json"]["model"] == "ray-2"
        assert post["json"]["aspect_ratio"] == "16:9"
        assert post["json"]["loop"] is False
        # Bearer auth
        assert post["headers"]["Authorization"] == "Bearer k"

    def test_image_to_video_sends_keyframes(self):
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"id": "task-i", "state": "queued"})],
            get_responses=[
                _mock_response(
                    json_data={
                        "id": "task-i",
                        "state": "completed",
                        "assets": {"video": "https://cdn.luma.ai/img2vid.mp4"},
                    }
                ),
            ],
        )
        with patch("prompture.drivers.luma_video_gen_driver.httpx.Client", return_value=client):
            d = LumaVideoGenDriver(api_key="k")
            d.generate_video("dance", {"image": "https://x/y.png", "poll_interval": 0.0})

        sent = client.post_calls[0]["json"]
        assert sent["keyframes"] == {"frame0": {"type": "image", "url": "https://x/y.png"}}

    def test_http_error_raises_with_body(self):
        client = _FakeSyncClient(post_responses=[_mock_response(status_code=403, text="forbidden", json_data=None)])
        with patch("prompture.drivers.luma_video_gen_driver.httpx.Client", return_value=client):
            d = LumaVideoGenDriver(api_key="k")
            with pytest.raises(RuntimeError, match="Luma submit failed"):
                d.generate_video("x", {})

    def test_polling_timeout_after_max_retries(self):
        # All polls return "queued" — never terminal.
        polls = [_mock_response(json_data={"id": "t", "state": "queued"}) for _ in range(5)]
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"id": "t", "state": "queued"})],
            get_responses=polls,
        )
        with patch("prompture.drivers.luma_video_gen_driver.httpx.Client", return_value=client):
            d = LumaVideoGenDriver(api_key="k")
            with pytest.raises(TimeoutError, match="timed out"):
                d.generate_video("x", {"max_retries": 5, "poll_interval": 0.0})

    def test_failed_state_raises(self):
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"id": "t", "state": "queued"})],
            get_responses=[
                _mock_response(json_data={"id": "t", "state": "failed", "failure_reason": "nsfw"}),
            ],
        )
        with patch("prompture.drivers.luma_video_gen_driver.httpx.Client", return_value=client):
            d = LumaVideoGenDriver(api_key="k")
            with pytest.raises(RuntimeError, match="failed"):
                d.generate_video("x", {"poll_interval": 0.0})


# ── Luma async smoke ────────────────────────────────────────────────


class TestAsyncLumaVideoDriver:
    def test_async_construct(self):
        d = AsyncLumaVideoGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.model == "ray-2"
        assert d.KNOWN_MODELS == LumaVideoGenDriver.KNOWN_MODELS

    def test_async_text_to_video(self):
        client = _FakeAsyncClient(
            post_responses=[_mock_response(json_data={"id": "tA", "state": "queued"})],
            get_responses=[
                _mock_response(
                    json_data={
                        "id": "tA",
                        "state": "completed",
                        "assets": {"video": "https://cdn.luma.ai/x.mp4"},
                    }
                ),
            ],
        )
        with patch("prompture.drivers.async_luma_video_gen_driver.httpx.AsyncClient", return_value=client):
            d = AsyncLumaVideoGenDriver(api_key="k")
            result = asyncio.run(d.generate_video("hi", {"poll_interval": 0.0}))

        assert result["meta"]["task_id"] == "tA"
        assert result["videos"][0].url == "https://cdn.luma.ai/x.mp4"


# ── Pika helpers ────────────────────────────────────────────────────


class TestPikaBuildBody:
    def test_basic_t2v(self):
        op, body = pika_build_body("a cat", {})
        assert op == "t2v"
        assert body["promptText"] == "a cat"
        assert body["options"]["aspectRatio"] == "16:9"
        assert body["options"]["frameRate"] == 24
        assert body["options"]["duration"] == 5

    def test_i2v_with_image(self):
        op, body = pika_build_body("dance", {"image": "https://x/y.png", "duration": 8})
        assert op == "i2v"
        assert body["image"] == "https://x/y.png"
        assert body["options"]["duration"] == 8

    def test_bad_aspect_raises(self):
        with pytest.raises(ValueError):
            pika_build_body("x", {"aspect_ratio": "11:7"})


# ── Pika sync driver ────────────────────────────────────────────────


class TestPikaVideoDriver:
    def test_construct(self):
        d = PikaVideoGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.model == "pika-2.2"
        assert d.endpoint == "https://api.dev.pikalabs.org"

    def test_missing_key_raises(self):
        d = PikaVideoGenDriver(api_key=None)
        with pytest.raises(RuntimeError, match="PIKA_API_KEY"):
            d.generate_video("x", {})

    def test_known_models(self):
        for m in ("pika-2.2", "pika-2.1", "pika-1.5"):
            assert m in PikaVideoGenDriver.KNOWN_MODELS

    def test_text_to_video_full_flow(self):
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"video_id": "vid-1", "status": "pending"})],
            get_responses=[
                _mock_response(json_data={"video_id": "vid-1", "status": "pending"}),
                _mock_response(
                    json_data={
                        "video_id": "vid-1",
                        "status": "finished",
                        "result_url": "https://cdn.pika.art/v.mp4",
                    }
                ),
            ],
        )
        with patch("prompture.drivers.pika_video_gen_driver.httpx.Client", return_value=client):
            d = PikaVideoGenDriver(api_key="k")
            result = d.generate_video("a cat", {"poll_interval": 0.0})

        assert result["meta"]["task_id"] == "vid-1"
        assert result["meta"]["status"] == "finished"
        assert result["meta"]["model_name"] == "pika/pika-2.2"
        assert result["videos"][0].url == "https://cdn.pika.art/v.mp4"
        post = client.post_calls[0]
        assert "/generate/2.2/t2v" in post["url"]
        assert post["json"]["promptText"] == "a cat"
        assert post["headers"]["Authorization"] == "Bearer k"

    def test_image_to_video_uses_i2v(self):
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"video_id": "vi", "status": "pending"})],
            get_responses=[
                _mock_response(
                    json_data={
                        "video_id": "vi",
                        "status": "finished",
                        "result_url": "https://cdn.pika.art/i.mp4",
                    }
                ),
            ],
        )
        with patch("prompture.drivers.pika_video_gen_driver.httpx.Client", return_value=client):
            d = PikaVideoGenDriver(api_key="k")
            d.generate_video("c", {"image": "https://x/im.png", "poll_interval": 0.0})

        assert "/generate/2.2/i2v" in client.post_calls[0]["url"]
        assert client.post_calls[0]["json"]["image"] == "https://x/im.png"

    def test_http_error_raises_with_body(self):
        client = _FakeSyncClient(post_responses=[_mock_response(status_code=500, text="boom", json_data=None)])
        with patch("prompture.drivers.pika_video_gen_driver.httpx.Client", return_value=client):
            d = PikaVideoGenDriver(api_key="k")
            with pytest.raises(RuntimeError, match="Pika submit failed"):
                d.generate_video("x", {})

    def test_polling_timeout(self):
        polls = [_mock_response(json_data={"status": "pending"}) for _ in range(4)]
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"video_id": "v", "status": "pending"})],
            get_responses=polls,
        )
        with patch("prompture.drivers.pika_video_gen_driver.httpx.Client", return_value=client):
            d = PikaVideoGenDriver(api_key="k")
            with pytest.raises(TimeoutError, match="timed out"):
                d.generate_video("x", {"max_retries": 4, "poll_interval": 0.0})

    def test_failed_status_raises(self):
        client = _FakeSyncClient(
            post_responses=[_mock_response(json_data={"video_id": "v", "status": "pending"})],
            get_responses=[
                _mock_response(json_data={"video_id": "v", "status": "failed", "error": "bad"}),
            ],
        )
        with patch("prompture.drivers.pika_video_gen_driver.httpx.Client", return_value=client):
            d = PikaVideoGenDriver(api_key="k")
            with pytest.raises(RuntimeError, match="failed"):
                d.generate_video("x", {"poll_interval": 0.0})


# ── Pika async smoke ────────────────────────────────────────────────


class TestAsyncPikaVideoDriver:
    def test_async_construct(self):
        d = AsyncPikaVideoGenDriver(api_key="k")
        assert d.api_key == "k"
        assert d.model == "pika-2.2"
        assert d.KNOWN_MODELS == PikaVideoGenDriver.KNOWN_MODELS

    def test_async_text_to_video(self):
        client = _FakeAsyncClient(
            post_responses=[_mock_response(json_data={"video_id": "vA", "status": "pending"})],
            get_responses=[
                _mock_response(
                    json_data={
                        "video_id": "vA",
                        "status": "finished",
                        "result_url": "https://cdn.pika.art/a.mp4",
                    }
                ),
            ],
        )
        with patch("prompture.drivers.async_pika_video_gen_driver.httpx.AsyncClient", return_value=client):
            d = AsyncPikaVideoGenDriver(api_key="k")
            result = asyncio.run(d.generate_video("hi", {"poll_interval": 0.0}))

        assert result["meta"]["task_id"] == "vA"
        assert result["videos"][0].url == "https://cdn.pika.art/a.mp4"


# ── Registry integration ────────────────────────────────────────────


class TestRegistry:
    def test_luma_video_factory(self):
        drv = get_video_gen_driver_for_model("luma/ray-2")
        assert isinstance(drv, LumaVideoGenDriver)
        assert drv.model == "ray-2"

    def test_pika_video_factory(self):
        drv = get_video_gen_driver_for_model("pika/pika-2.2")
        assert isinstance(drv, PikaVideoGenDriver)
        assert drv.model == "pika-2.2"

    def test_async_luma_factory(self):
        drv = get_async_video_gen_driver_for_model("luma/ray-flash-2")
        assert isinstance(drv, AsyncLumaVideoGenDriver)

    def test_async_pika_factory(self):
        drv = get_async_video_gen_driver_for_model("pika/pika-2.1")
        assert isinstance(drv, AsyncPikaVideoGenDriver)

    def test_luma_aliases(self):
        for alias in ("dream-machine", "lumalabs"):
            drv = get_video_gen_driver_for_model(f"{alias}/ray-2")
            assert isinstance(drv, LumaVideoGenDriver)

    def test_descriptors_registered(self):
        from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP

        assert "luma" in PROVIDER_DESCRIPTOR_MAP
        assert "pika" in PROVIDER_DESCRIPTOR_MAP
        assert PROVIDER_DESCRIPTOR_MAP["luma"].display_name == "Luma AI"
        assert PROVIDER_DESCRIPTOR_MAP["pika"].display_name == "Pika Labs"
        assert PROVIDER_DESCRIPTOR_MAP["luma"].is_configured_check == "luma_api_key"
        assert PROVIDER_DESCRIPTOR_MAP["pika"].is_configured_check == "pika_api_key"
