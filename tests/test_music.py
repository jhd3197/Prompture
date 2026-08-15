"""Tests for the music modality: base, Muapi driver, registry, discovery (G06)."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import httpx

from prompture.drivers.async_muapi_aggregator_driver import AsyncMuapiMusicGenDriver
from prompture.drivers.muapi_aggregator_driver import MuapiMusicGenDriver
from prompture.drivers.music_base import MusicGenDriver
from prompture.drivers.music_registry import (
    get_async_music_driver_for_model,
    get_music_driver_for_model,
)
from prompture.drivers.registry import is_music_driver_registered
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
        assert MusicGenDriver.supports_instrumental is True
        assert MusicGenDriver.supports_vocals is True

    def test_registered(self):
        assert is_music_driver_registered("muapi")

    def test_registry_routing(self):
        d = get_music_driver_for_model("muapi/suno-create-music")
        assert isinstance(d, MuapiMusicGenDriver)
        assert d.model == "suno-create-music"

    def test_async_registry_routing(self):
        assert isinstance(get_async_music_driver_for_model("muapi/suno-create-music"), AsyncMuapiMusicGenDriver)


class TestGenerateMusic:
    def test_create_returns_audio(self):
        d = MuapiMusicGenDriver(api_key="k", model="suno-create-music")
        with (
            patch(f"{_HTTP}.httpx.post", return_value=_Response(200, {"request_id": "m1"})) as mp,
            patch(
                f"{_HTTP}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/song.mp3"]}),
            ),
        ):
            res = d.generate_music("an upbeat synthwave track", {"instrumental": True})
        assert res["audio"][0].url == "https://x/song.mp3"
        assert res["meta"]["audio_count"] == 1
        assert res["meta"]["operation"] == "create"
        assert mp.call_args.kwargs["json"]["instrumental"] is True
        assert mp.call_args.kwargs["json"]["prompt"] == "an upbeat synthwave track"

    def test_operation_and_audio_url_forwarded(self):
        d = MuapiMusicGenDriver(api_key="k", model="suno-extend-music")
        with (
            patch(f"{_HTTP}.httpx.post", return_value=_Response(200, {"request_id": "m2"})) as mp,
            patch(
                f"{_HTTP}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/ext.mp3"]}),
            ),
        ):
            res = d.generate_music("continue it", {"operation": "extend", "audio_url": "https://in/src.mp3"})
        assert res["meta"]["operation"] == "extend"
        payload = mp.call_args.kwargs["json"]
        assert payload["operation"] == "extend"
        assert payload["audio_url"] == "https://in/src.mp3"

    def test_cost_from_media_rates(self):
        # suno-create-music = $0.10 per job
        d = MuapiMusicGenDriver(api_key="k", model="suno-create-music")
        with (
            patch(f"{_HTTP}.httpx.post", return_value=_Response(200, {"request_id": "m3"})),
            patch(
                f"{_HTTP}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/s.mp3"]}),
            ),
        ):
            res = d.generate_music("x", {})
        assert res["meta"]["cost"] == 0.10

    def test_poll_false_returns_handle(self):
        d = MuapiMusicGenDriver(api_key="k", model="suno-create-music")
        with patch(f"{_HTTP}.httpx.post", return_value=_Response(200, {"request_id": "m4"})):
            res = d.generate_music("x", {"poll": False})
        handle = JobHandle.from_meta(res["meta"])
        assert handle.modality == "music"
        assert handle.task_id == "m4"

    def test_resume_job_audio_asset(self):
        d = MuapiMusicGenDriver(api_key="k", model="suno-create-music")
        handle = JobHandle(task_id="m5", provider="muapi", model="suno-create-music", modality="music")
        with patch(
            f"{_HTTP}.httpx.get", return_value=_Response(200, {"status": "completed", "outputs": ["https://x/r.mp3"]})
        ):
            result = d.resume_job(handle)
        assert result.done
        assert result.assets[0].kind == "audio"
        assert result.first_url == "https://x/r.mp3"

    def test_missing_key_raises(self):
        d = MuapiMusicGenDriver(api_key=None, model="suno-create-music")
        try:
            d.generate_music("x", {})
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "MUAPI_API_KEY" in str(e)


def _mock_async_client(handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real(transport=httpx.MockTransport(handler))

    return factory


class TestAsyncMusic:
    def test_async_generate(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path.endswith("/api/v1/suno-create-music"):
                return httpx.Response(200, json={"request_id": "am1"})
            if request.method == "GET" and request.url.path.endswith("/predictions/am1/result"):
                return httpx.Response(200, json={"status": "completed", "outputs": ["https://x/a.mp3"]})
            return httpx.Response(404, json={})

        d = AsyncMuapiMusicGenDriver(api_key="k", model="suno-create-music")
        with patch("prompture.drivers.async_aggregator_base.httpx.AsyncClient", _mock_async_client(handler)):
            res = asyncio.run(d.generate_music("a jazzy tune", {}))
        assert res["audio"][0].url == "https://x/a.mp3"


class TestDiscovery:
    def test_available_when_configured(self):
        from prompture.infra.discovery import get_available_music_models

        os.environ["MUAPI_API_KEY"] = "test-key"
        try:
            models = get_available_music_models()
        finally:
            del os.environ["MUAPI_API_KEY"]
        assert "muapi/suno-create-music" in models
