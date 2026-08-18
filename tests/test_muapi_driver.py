"""Tests for the Muapi aggregator driver (sync + async) and its job/resume API."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx
import pytest

from prompture.drivers.async_muapi_aggregator_driver import AsyncMuapiVideoGenDriver
from prompture.drivers.muapi_aggregator_driver import (
    MuapiImageGenDriver,
    MuapiVideoGenDriver,
    _extract_urls,
)
from prompture.jobs import JobHandle

# HTTP now lives in the shared aggregator base; patch there.
_MOD = "prompture.drivers.aggregator_base"


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class TestExtractUrls:
    def test_outputs_list_of_strings(self):
        assert _extract_urls({"outputs": ["a", "b"]}) == ["a", "b"]

    def test_outputs_list_of_dicts(self):
        assert _extract_urls({"outputs": [{"url": "a"}]}) == ["a"]

    def test_top_level_url(self):
        assert _extract_urls({"url": "z"}) == ["z"]

    def test_output_object(self):
        assert _extract_urls({"output": {"url": "q"}}) == ["q"]

    def test_empty(self):
        assert _extract_urls({"status": "completed"}) == []


class TestMuapiImageGen:
    def test_generate_image_submits_and_polls(self):
        d = MuapiImageGenDriver(api_key="k", model="nano-banana")
        with (
            patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "req1"})) as mp,
            patch(
                f"{_MOD}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/out.png"]}),
            ) as mg,
        ):
            res = d.generate_image("a cat", {"aspect_ratio": "1:1"})
        assert res["meta"]["image_count"] == 1
        assert res["images"][0].url == "https://x/out.png"
        assert res["meta"]["request_id"] == "req1"
        assert res["meta"]["model_name"] == "muapi/nano-banana"
        # submit hit the slug endpoint with the prompt + option
        assert mp.call_args.args[0].endswith("/api/v1/nano-banana")
        assert mp.call_args.kwargs["json"] == {"prompt": "a cat", "aspect_ratio": "1:1"}
        mg.assert_called_once()

    def test_poll_false_returns_job_handle(self):
        d = MuapiImageGenDriver(api_key="k", model="nano-banana")
        with (
            patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "req2"})),
            patch(f"{_MOD}.httpx.get") as mg,
        ):
            res = d.generate_image("x", {"poll": False})
        assert res["meta"]["status"] == "pending"
        handle = JobHandle.from_meta(res["meta"])
        assert handle.task_id == "req2"
        assert handle.provider == "muapi"
        assert handle.modality == "image"
        mg.assert_not_called()

    def test_edit_image_hosts_then_submits(self):
        d = MuapiImageGenDriver(api_key="k", model="seedream-edit")
        with (
            patch.object(MuapiImageGenDriver, "upload_file", return_value="https://host/in.png") as mu,
            patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "e1"})) as mp,
            patch(
                f"{_MOD}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/e.png"]}),
            ),
        ):
            res = d.edit_image(b"rawpng", "make it blue", {})
        mu.assert_called_once()
        assert mp.call_args.kwargs["json"]["image_url"] == "https://host/in.png"
        assert res["images"][0].url == "https://x/e.png"

    def test_edit_image_uses_provided_url_without_upload(self):
        d = MuapiImageGenDriver(api_key="k", model="seedream-edit")
        with (
            patch.object(MuapiImageGenDriver, "upload_file") as mu,
            patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "e2"})) as mp,
            patch(f"{_MOD}.httpx.get", return_value=_Response(200, {"status": "completed", "outputs": ["u"]})),
        ):
            d.edit_image(b"x", "p", {"image_url": "https://given/img.png"})
        mu.assert_not_called()
        assert mp.call_args.kwargs["json"]["image_url"] == "https://given/img.png"

    def test_missing_key_raises(self):
        d = MuapiImageGenDriver(api_key=None, model="nano-banana")
        with pytest.raises(RuntimeError, match="MUAPI_API_KEY"):
            d.generate_image("x", {})

    def test_supports_edit_flag(self):
        assert MuapiImageGenDriver.supports_edit is True

    def test_cost_from_media_rates(self):
        # G16: generate cost is quoted from the media rate table (nano-banana = $0.039).
        d = MuapiImageGenDriver(api_key="k", model="nano-banana")
        with (
            patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "c1"})),
            patch(
                f"{_MOD}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/o.png"]}),
            ),
        ):
            res = d.generate_image("x", {})
        assert res["meta"]["cost"] == 0.039


class TestMuapiResume:
    def test_resume_job_polls_to_completion(self):
        d = MuapiImageGenDriver(api_key="k", model="nano-banana")
        handle = JobHandle(task_id="req3", provider="muapi", model="nano-banana", modality="image")
        with patch(
            f"{_MOD}.httpx.get",
            return_value=_Response(200, {"status": "completed", "outputs": ["https://x/r.png"]}),
        ):
            result = d.resume_job(handle)
        assert result.done is True
        assert result.first_url == "https://x/r.png"
        assert result.assets[0].provenance["request_id"] == "req3"

    def test_resume_job_no_poll_returns_current(self):
        d = MuapiVideoGenDriver(api_key="k", model="kling")
        handle = JobHandle(task_id="req4", provider="muapi", model="kling", modality="video")
        with patch(f"{_MOD}.httpx.get", return_value=_Response(200, {"status": "processing"})):
            result = d.resume_job(handle, poll=False)
        assert result.status == "running"
        assert result.assets == []

    def test_get_job_status(self):
        d = MuapiVideoGenDriver(api_key="k", model="kling")
        handle = JobHandle(task_id="req5", provider="muapi", model="kling", modality="video")
        with patch(f"{_MOD}.httpx.get", return_value=_Response(200, {"status": "IN_PROGRESS"})):
            assert d.get_job_status(handle) == "running"


class TestMuapiVideoGen:
    def test_generate_video_t2v(self):
        d = MuapiVideoGenDriver(api_key="k", model="kling-video")
        with (
            patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "v1"})),
            patch(
                f"{_MOD}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://x/v.mp4"]}),
            ),
        ):
            res = d.generate_video("a dog running", {"duration": 5, "aspect_ratio": "16:9"})
        assert res["meta"]["video_count"] == 1
        assert res["videos"][0].url == "https://x/v.mp4"
        assert res["meta"]["duration_seconds"] == 5


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _mock_async_client(handler):
    def factory(*args, **kwargs):
        kwargs.pop("timeout", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))

    return factory


class TestAsyncMuapiVideoGen:
    def test_async_generate_video(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path.endswith("/api/v1/kling"):
                return httpx.Response(200, json={"request_id": "areq"})
            if request.method == "GET" and request.url.path.endswith("/predictions/areq/result"):
                return httpx.Response(200, json={"status": "completed", "outputs": ["https://x/a.mp4"]})
            return httpx.Response(404, json={})

        d = AsyncMuapiVideoGenDriver(api_key="k", model="kling")
        with patch(
            "prompture.drivers.async_aggregator_base.httpx.AsyncClient",
            _mock_async_client(handler),
        ):
            res = asyncio.run(d.generate_video("x", {"duration": 3}))
        assert res["videos"][0].url == "https://x/a.mp4"
        assert res["meta"]["request_id"] == "areq"


class TestDescriptorWiring:
    def test_img_routing(self):
        from prompture.drivers.img_gen_registry import get_img_gen_driver_for_model

        d = get_img_gen_driver_for_model("muapi/some-slug")
        assert isinstance(d, MuapiImageGenDriver)
        assert d.model == "some-slug"

    def test_video_routing(self):
        from prompture.drivers.video_gen_registry import get_video_gen_driver_for_model

        d = get_video_gen_driver_for_model("muapi/some-video")
        assert isinstance(d, MuapiVideoGenDriver)
        assert d.model == "some-video"
