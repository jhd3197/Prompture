"""Tests for the lipsync modality: base, Muapi driver, registry, discovery (G05)."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import httpx

from prompture.drivers.async_muapi_aggregator_driver import AsyncMuapiLipsyncDriver
from prompture.drivers.lipsync_base import LipsyncDriver
from prompture.drivers.lipsync_registry import (
    get_async_lipsync_driver_for_model,
    get_lipsync_driver_for_model,
)
from prompture.drivers.muapi_aggregator_driver import MuapiLipsyncDriver
from prompture.drivers.registry import is_lipsync_driver_registered
from prompture.jobs import JobHandle

_HTTP = "prompture.drivers.aggregator_base"


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class TestBaseAndWiring:
    def test_base_flags(self):
        assert LipsyncDriver.supports_image is True
        assert LipsyncDriver.supports_video is True

    def test_registered(self):
        assert is_lipsync_driver_registered("muapi")

    def test_registry_routing(self):
        d = get_lipsync_driver_for_model("muapi/infinite-talk")
        assert isinstance(d, MuapiLipsyncDriver)
        assert d.model == "infinite-talk"

    def test_async_registry_routing(self):
        d = get_async_lipsync_driver_for_model("muapi/infinite-talk")
        assert isinstance(d, AsyncMuapiLipsyncDriver)


class TestGenerateLipsync:
    def test_image_plus_audio(self):
        d = MuapiLipsyncDriver(api_key="k", model="infinite-talk")
        with (
            patch(f"{_HTTP}.httpx.post", return_value=_Response(200, {"request_id": "l1"})) as mp,
            patch(
                f"{_HTTP}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/talk.mp4"]}),
            ),
        ):
            res = d.generate_lipsync(
                None,
                {"audio_url": "https://in/a.mp3", "image_url": "https://in/p.png"},
            )
        assert res["videos"][0].url == "https://x/talk.mp4"
        assert res["meta"]["category"] == "image"
        payload = mp.call_args.kwargs["json"]
        assert payload["audio_url"] == "https://in/a.mp3"
        assert payload["image_url"] == "https://in/p.png"

    def test_video_plus_audio_category(self):
        d = MuapiLipsyncDriver(api_key="k", model="latentsync")
        with (
            patch(f"{_HTTP}.httpx.post", return_value=_Response(200, {"request_id": "l2"})),
            patch(
                f"{_HTTP}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/v.mp4"]}),
            ),
        ):
            res = d.generate_lipsync(None, {"audio_url": "https://in/a.mp3", "video_url": "https://in/src.mp4"})
        assert res["meta"]["category"] == "video"

    def test_bytes_audio_is_uploaded(self):
        d = MuapiLipsyncDriver(api_key="k", model="infinite-talk")
        with (
            patch.object(MuapiLipsyncDriver, "upload_file", return_value="https://host/a.mp3") as mu,
            patch(f"{_HTTP}.httpx.post", return_value=_Response(200, {"request_id": "l3"})) as mp,
            patch(
                f"{_HTTP}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/o.mp4"]}),
            ),
        ):
            d.generate_lipsync(b"rawaudio", {"image_url": "https://in/p.png"})
        mu.assert_called_once()
        assert mp.call_args.kwargs["json"]["audio_url"] == "https://host/a.mp3"

    def test_cost_from_media_rates(self):
        # infinite-talk = $0.04/sec; 5s → $0.20
        d = MuapiLipsyncDriver(api_key="k", model="infinite-talk")
        with (
            patch(f"{_HTTP}.httpx.post", return_value=_Response(200, {"request_id": "l4"})),
            patch(
                f"{_HTTP}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/o.mp4"]}),
            ),
        ):
            res = d.generate_lipsync(
                None, {"audio_url": "https://in/a.mp3", "image_url": "https://in/p.png", "duration": 5}
            )
        assert res["meta"]["cost"] == 0.2

    def test_poll_false_returns_handle(self):
        d = MuapiLipsyncDriver(api_key="k", model="infinite-talk")
        with patch(f"{_HTTP}.httpx.post", return_value=_Response(200, {"request_id": "l5"})):
            res = d.generate_lipsync(
                None, {"audio_url": "https://in/a.mp3", "image_url": "https://in/p.png", "poll": False}
            )
        handle = JobHandle.from_meta(res["meta"])
        assert handle.modality == "lipsync"
        assert handle.task_id == "l5"

    def test_resume_job(self):
        d = MuapiLipsyncDriver(api_key="k", model="infinite-talk")
        handle = JobHandle(task_id="l6", provider="muapi", model="infinite-talk", modality="lipsync")
        with patch(
            f"{_HTTP}.httpx.get", return_value=_Response(200, {"status": "completed", "outputs": ["https://x/r.mp4"]})
        ):
            result = d.resume_job(handle)
        assert result.done and result.first_url == "https://x/r.mp4"

    def test_missing_key_raises(self):
        d = MuapiLipsyncDriver(api_key=None, model="infinite-talk")
        try:
            d.generate_lipsync(None, {"audio_url": "https://in/a.mp3"})
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "MUAPI_API_KEY" in str(e)


def _mock_async_client(handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real(transport=httpx.MockTransport(handler))

    return factory


class TestAsyncLipsync:
    def test_async_generate(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path.endswith("/api/v1/infinite-talk"):
                return httpx.Response(200, json={"request_id": "al1"})
            if request.method == "GET" and request.url.path.endswith("/predictions/al1/result"):
                return httpx.Response(200, json={"status": "completed", "outputs": ["https://x/a.mp4"]})
            return httpx.Response(404, json={})

        d = AsyncMuapiLipsyncDriver(api_key="k", model="infinite-talk")
        with patch("prompture.drivers.async_aggregator_base.httpx.AsyncClient", _mock_async_client(handler)):
            res = asyncio.run(
                d.generate_lipsync(None, {"audio_url": "https://in/a.mp3", "image_url": "https://in/p.png"})
            )
        assert res["videos"][0].url == "https://x/a.mp4"
        assert res["meta"]["category"] == "image"


class TestDiscovery:
    def test_available_when_configured(self):
        from prompture.infra.discovery import get_available_lipsync_models

        os.environ["MUAPI_API_KEY"] = "test-key"
        try:
            models = get_available_lipsync_models()
        finally:
            del os.environ["MUAPI_API_KEY"]
        assert "muapi/infinite-talk" in models

    def test_empty_when_unconfigured(self):
        from prompture.infra.discovery import get_available_lipsync_models

        saved = os.environ.pop("MUAPI_API_KEY", None)
        try:
            with patch("prompture.infra.discovery._cfg_value", return_value=None):
                assert get_available_lipsync_models() == []
        finally:
            if saved is not None:
                os.environ["MUAPI_API_KEY"] = saved
