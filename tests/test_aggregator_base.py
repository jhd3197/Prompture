"""Tests for the generic AggregatorDriver base via a toy subclass.

Proves a new aggregator (different auth scheme + URL shapes) is a thin subclass,
and that estimate_cost (G16) is wired into generated responses.
"""

from __future__ import annotations

from unittest.mock import patch

from prompture.drivers.aggregator_base import (
    AggregatorImageDriver,
    AggregatorVideoDriver,
)
from prompture.infra.media_pricing import register_media_rate
from prompture.jobs import JobHandle

_MOD = "prompture.drivers.aggregator_base"


class _Response:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _ToyMixin:
    PROVIDER = "toy"
    DEFAULT_BASE = "https://toy.test"
    API_KEY_ENVS = ("TOY_KEY",)
    ENDPOINT_ENV = "TOY_ENDPOINT"
    AUTH_STYLE = "bearer"
    DEFAULT_IMAGE_MODEL = "m1"
    DEFAULT_VIDEO_MODEL = "v1"

    def submit_url(self, slug):
        return f"{self.endpoint}/run/{slug}"

    def result_url(self, request_id):
        return f"{self.endpoint}/jobs/{request_id}"

    def build_image_payload(self, prompt, options):
        return {"input": prompt}

    def build_video_payload(self, prompt, options):
        return {"input": prompt}


class ToyImageDriver(_ToyMixin, AggregatorImageDriver):
    pass


class ToyVideoDriver(_ToyMixin, AggregatorVideoDriver):
    pass


class TestAuthAndConfig:
    def test_bearer_auth_pair(self):
        d = ToyImageDriver(api_key="k", model="m1")
        assert d._auth_pair() == ("Authorization", "Bearer k")
        assert d._headers()["Authorization"] == "Bearer k"

    def test_endpoint_default(self):
        assert ToyImageDriver(api_key="k").endpoint == "https://toy.test"

    def test_default_models(self):
        assert ToyImageDriver(api_key="k").model == "m1"
        assert ToyVideoDriver(api_key="k").model == "v1"


class TestGenericGenerate:
    def test_generate_image_uses_provider_urls_and_auth(self):
        d = ToyImageDriver(api_key="k", model="m1")
        with (
            patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "t1"})) as mp,
            patch(
                f"{_MOD}.httpx.get",
                return_value=_Response(200, {"status": "succeeded", "outputs": [{"url": "https://toy/o.png"}]}),
            ),
        ):
            res = d.generate_image("hello", {})
        assert res["images"][0].url == "https://toy/o.png"
        assert res["meta"]["model_name"] == "toy/m1"
        assert mp.call_args.args[0] == "https://toy.test/run/m1"
        assert mp.call_args.kwargs["headers"]["Authorization"] == "Bearer k"
        assert mp.call_args.kwargs["json"] == {"input": "hello"}

    def test_cost_from_estimate(self):
        register_media_rate("toy/m1", {"per_image": 0.5})
        d = ToyImageDriver(api_key="k", model="m1")
        with (
            patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "t2"})),
            patch(
                f"{_MOD}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://toy/x.png"]}),
            ),
        ):
            res = d.generate_image("x", {})
        assert res["meta"]["cost"] == 0.5

    def test_poll_false_returns_handle(self):
        d = ToyImageDriver(api_key="k", model="m1")
        with patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "t3"})):
            res = d.generate_image("x", {"poll": False})
        handle = JobHandle.from_meta(res["meta"])
        assert handle.provider == "toy" and handle.task_id == "t3"

    def test_resume_job(self):
        d = ToyImageDriver(api_key="k", model="m1")
        handle = JobHandle(task_id="t4", provider="toy", model="m1", modality="image")
        with patch(
            f"{_MOD}.httpx.get", return_value=_Response(200, {"status": "completed", "outputs": ["https://toy/r.png"]})
        ):
            result = d.resume_job(handle)
        assert result.done and result.first_url == "https://toy/r.png"

    def test_video_generate(self):
        d = ToyVideoDriver(api_key="k", model="v1")
        with (
            patch(f"{_MOD}.httpx.post", return_value=_Response(200, {"request_id": "tv"})),
            patch(
                f"{_MOD}.httpx.get",
                return_value=_Response(200, {"status": "completed", "outputs": ["https://toy/v.mp4"]}),
            ),
        ):
            res = d.generate_video("move", {"duration": 4})
        assert res["videos"][0].url == "https://toy/v.mp4"

    def test_missing_key_raises(self):
        d = ToyImageDriver(api_key=None, model="m1")
        try:
            d.generate_image("x", {})
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "TOY_API_KEY" in str(e)
