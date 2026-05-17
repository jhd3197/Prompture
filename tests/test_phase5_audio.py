"""Phase 5: Cartesia, Deepgram, and AssemblyAI audio driver tests.

All HTTP calls are mocked via ``unittest.mock.patch`` against ``httpx``.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from prompture.drivers.assemblyai_stt_driver import AssemblyAISTTDriver
from prompture.drivers.async_assemblyai_stt_driver import AsyncAssemblyAISTTDriver
from prompture.drivers.async_cartesia_tts_driver import AsyncCartesiaTTSDriver
from prompture.drivers.async_deepgram_stt_driver import AsyncDeepgramSTTDriver
from prompture.drivers.async_deepgram_tts_driver import AsyncDeepgramTTSDriver
from prompture.drivers.audio_registry import (
    get_async_stt_driver_for_model,
    get_async_tts_driver_for_model,
    get_stt_driver_for_model,
    get_tts_driver_for_model,
)
from prompture.drivers.cartesia_tts_driver import CartesiaTTSDriver
from prompture.drivers.deepgram_stt_driver import DeepgramSTTDriver
from prompture.drivers.deepgram_tts_driver import DeepgramTTSDriver

# ── helpers ──────────────────────────────────────────────────────────


def _mock_response(*, status_code: int = 200, content: bytes = b"", json_data: Any = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = text
    if json_data is not None:
        resp.json = MagicMock(return_value=json_data)
    return resp


class _AsyncReturn:
    """Awaitable wrapper so AsyncMock-style patches work without AsyncMock."""

    def __init__(self, value: Any):
        self._value = value

    def __await__(self):
        async def _coro():
            return self._value

        return _coro().__await__()


class _FakeAsyncClient:
    """Minimal async context manager + post/get returning preset responses."""

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


# ── Cartesia TTS ─────────────────────────────────────────────────────


class TestCartesiaTTS:
    def test_synthesize_returns_bytes(self):
        drv = CartesiaTTSDriver(api_key="fake", voice_id="voice-123")
        fake_resp = _mock_response(status_code=200, content=b"WAVE_BYTES")
        with patch("prompture.drivers.cartesia_tts_driver.httpx") as h:
            h.post.return_value = fake_resp
            result = drv.synthesize("hello world", {})

        assert result["audio"] == b"WAVE_BYTES"
        assert result["meta"]["characters"] == len("hello world")
        assert result["meta"]["model_name"] == "cartesia/sonic-2"
        assert result["meta"]["pricing_unknown"] is True
        # Headers / body assertions
        call = h.post.call_args
        body = call.kwargs["json"]
        headers = call.kwargs["headers"]
        assert headers["X-API-Key"] == "fake"
        assert headers["Cartesia-Version"] == "2025-04-16"
        assert body["model_id"] == "sonic-2"
        assert body["voice"] == {"mode": "id", "id": "voice-123"}
        assert body["transcript"] == "hello world"

    def test_requires_voice_id(self):
        drv = CartesiaTTSDriver(api_key="fake")  # no voice_id set
        with patch("prompture.drivers.cartesia_tts_driver.httpx"):
            with pytest.raises(RuntimeError, match="voice_id"):
                drv.synthesize("hi", {})

    def test_http_error_raises_runtime(self):
        drv = CartesiaTTSDriver(api_key="fake", voice_id="v1")
        fake_resp = _mock_response(status_code=401, text="unauthorized")
        with patch("prompture.drivers.cartesia_tts_driver.httpx") as h:
            h.post.return_value = fake_resp
            with pytest.raises(RuntimeError, match="401.*unauthorized"):
                drv.synthesize("hi", {})

    def test_async_synthesize(self):
        drv = AsyncCartesiaTTSDriver(api_key="fake", voice_id="v1")
        fake_resp = _mock_response(status_code=200, content=b"ASYNC_WAV")
        client = _FakeAsyncClient(post_responses=[fake_resp])
        with patch("prompture.drivers.async_cartesia_tts_driver.httpx") as h:
            h.AsyncClient.return_value = client
            result = asyncio.run(drv.synthesize("hi", {}))
        assert result["audio"] == b"ASYNC_WAV"
        assert result["meta"]["voice_id"] == "v1"


# ── Deepgram STT ─────────────────────────────────────────────────────


_NOVA_RESPONSE = {
    "metadata": {"duration": 1.23},
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": "hello there",
                        "confidence": 0.98,
                        "words": [],
                    }
                ]
            }
        ]
    },
}


class TestDeepgramSTT:
    def test_transcribe_returns_text(self):
        drv = DeepgramSTTDriver(api_key="fake")
        fake_resp = _mock_response(status_code=200, json_data=_NOVA_RESPONSE)
        with patch("prompture.drivers.deepgram_stt_driver.httpx") as h:
            h.post.return_value = fake_resp
            result = drv.transcribe(b"AUDIO", {})

        assert result["text"] == "hello there"
        assert result["meta"]["model_name"] == "deepgram/nova-3"
        assert result["meta"]["duration_seconds"] == pytest.approx(1.23)
        call = h.post.call_args
        assert call.kwargs["headers"]["Authorization"] == "Token fake"
        assert call.kwargs["params"]["model"] == "nova-3"
        assert call.kwargs["params"]["smart_format"] == "true"

    def test_http_error_raises(self):
        drv = DeepgramSTTDriver(api_key="fake")
        fake_resp = _mock_response(status_code=400, text="bad audio")
        with patch("prompture.drivers.deepgram_stt_driver.httpx") as h:
            h.post.return_value = fake_resp
            with pytest.raises(RuntimeError, match="400.*bad audio"):
                drv.transcribe(b"AUDIO", {})

    def test_async_transcribe(self):
        drv = AsyncDeepgramSTTDriver(api_key="fake")
        fake_resp = _mock_response(status_code=200, json_data=_NOVA_RESPONSE)
        client = _FakeAsyncClient(post_responses=[fake_resp])
        with patch("prompture.drivers.async_deepgram_stt_driver.httpx") as h:
            h.AsyncClient.return_value = client
            result = asyncio.run(drv.transcribe(b"AUDIO", {}))
        assert result["text"] == "hello there"


# ── Deepgram TTS ─────────────────────────────────────────────────────


class TestDeepgramTTS:
    def test_synthesize_returns_bytes(self):
        drv = DeepgramTTSDriver(api_key="fake")
        fake_resp = _mock_response(status_code=200, content=b"MP3_BYTES")
        with patch("prompture.drivers.deepgram_tts_driver.httpx") as h:
            h.post.return_value = fake_resp
            result = drv.synthesize("hi", {})

        assert result["audio"] == b"MP3_BYTES"
        assert result["media_type"] == "audio/mpeg"
        assert result["meta"]["model_name"] == "deepgram/aura-2-thalia-en"
        call = h.post.call_args
        assert call.kwargs["headers"]["Authorization"] == "Token fake"
        assert call.kwargs["params"]["model"] == "aura-2-thalia-en"
        assert call.kwargs["params"]["encoding"] == "mp3"
        assert call.kwargs["json"] == {"text": "hi"}

    def test_http_error_raises(self):
        drv = DeepgramTTSDriver(api_key="fake")
        fake_resp = _mock_response(status_code=500, text="server boom")
        with patch("prompture.drivers.deepgram_tts_driver.httpx") as h:
            h.post.return_value = fake_resp
            with pytest.raises(RuntimeError, match="500.*server boom"):
                drv.synthesize("hi", {})

    def test_async_synthesize(self):
        drv = AsyncDeepgramTTSDriver(api_key="fake")
        fake_resp = _mock_response(status_code=200, content=b"AUDIO")
        client = _FakeAsyncClient(post_responses=[fake_resp])
        with patch("prompture.drivers.async_deepgram_tts_driver.httpx") as h:
            h.AsyncClient.return_value = client
            result = asyncio.run(drv.synthesize("hi", {}))
        assert result["audio"] == b"AUDIO"


# ── AssemblyAI STT ───────────────────────────────────────────────────


_UPLOAD_RESP = {"upload_url": "https://cdn.assemblyai.com/upload/abc"}
_SUBMIT_RESP = {"id": "tr_123", "status": "queued"}
_PROCESSING_RESP = {"id": "tr_123", "status": "processing"}
_COMPLETED_RESP = {
    "id": "tr_123",
    "status": "completed",
    "text": "this is the transcript",
    "language_code": "en",
    "audio_duration": 4.5,
    "words": [],
}


class TestAssemblyAISTT:
    def test_three_step_flow(self):
        drv = AssemblyAISTTDriver(api_key="fake")
        upload = _mock_response(status_code=200, json_data=_UPLOAD_RESP)
        submit = _mock_response(status_code=200, json_data=_SUBMIT_RESP)
        poll1 = _mock_response(status_code=200, json_data=_PROCESSING_RESP)
        poll2 = _mock_response(status_code=200, json_data=_COMPLETED_RESP)

        with (
            patch("prompture.drivers.assemblyai_stt_driver.httpx") as h,
            patch("prompture.drivers.assemblyai_stt_driver.time.sleep") as sleep_mock,
        ):
            h.post.side_effect = [upload, submit]
            h.get.side_effect = [poll1, poll2]
            result = drv.transcribe(b"AUDIO_BYTES", {})

        assert result["text"] == "this is the transcript"
        assert result["language"] == "en"
        assert result["meta"]["model_name"] == "assemblyai/universal"
        assert result["meta"]["duration_seconds"] == pytest.approx(4.5)
        # The submit POST should include speech_model.
        submit_call = h.post.call_args_list[1]
        assert submit_call.kwargs["json"]["speech_model"] == "universal"
        assert submit_call.kwargs["json"]["audio_url"] == _UPLOAD_RESP["upload_url"]
        # Auth header on every call.
        assert h.post.call_args_list[0].kwargs["headers"]["Authorization"] == "fake"
        sleep_mock.assert_called()  # waited between polls

    def test_polling_timeout_raises(self):
        drv = AssemblyAISTTDriver(api_key="fake")
        upload = _mock_response(status_code=200, json_data=_UPLOAD_RESP)
        submit = _mock_response(status_code=200, json_data=_SUBMIT_RESP)
        processing = _mock_response(status_code=200, json_data=_PROCESSING_RESP)

        with (
            patch("prompture.drivers.assemblyai_stt_driver.httpx") as h,
            patch("prompture.drivers.assemblyai_stt_driver.time.sleep"),
        ):
            h.post.side_effect = [upload, submit]
            # Always processing — never completes.
            h.get.side_effect = lambda *a, **kw: processing
            with pytest.raises(RuntimeError, match="timed out"):
                drv.transcribe(b"AUDIO", {"max_poll_attempts": 3, "poll_interval_seconds": 0})

    def test_error_status_raises(self):
        drv = AssemblyAISTTDriver(api_key="fake")
        upload = _mock_response(status_code=200, json_data=_UPLOAD_RESP)
        submit = _mock_response(status_code=200, json_data=_SUBMIT_RESP)
        error_poll = _mock_response(status_code=200, json_data={"status": "error", "error": "decoding failed"})
        with (
            patch("prompture.drivers.assemblyai_stt_driver.httpx") as h,
            patch("prompture.drivers.assemblyai_stt_driver.time.sleep"),
        ):
            h.post.side_effect = [upload, submit]
            h.get.return_value = error_poll
            with pytest.raises(RuntimeError, match="decoding failed"):
                drv.transcribe(b"AUDIO", {})

    def test_upload_http_error_raises(self):
        drv = AssemblyAISTTDriver(api_key="fake")
        upload = _mock_response(status_code=401, text="invalid key")
        with patch("prompture.drivers.assemblyai_stt_driver.httpx") as h:
            h.post.side_effect = [upload]
            with pytest.raises(RuntimeError, match="upload failed.*401.*invalid key"):
                drv.transcribe(b"AUDIO", {})

    def test_async_flow(self):
        drv = AsyncAssemblyAISTTDriver(api_key="fake")
        upload = _mock_response(status_code=200, json_data=_UPLOAD_RESP)
        submit = _mock_response(status_code=200, json_data=_SUBMIT_RESP)
        poll = _mock_response(status_code=200, json_data=_COMPLETED_RESP)
        client = _FakeAsyncClient(post_responses=[upload, submit], get_responses=[poll])

        async def _no_sleep(_):
            return None

        with (
            patch("prompture.drivers.async_assemblyai_stt_driver.httpx") as h,
            patch("prompture.drivers.async_assemblyai_stt_driver.asyncio.sleep", new=_no_sleep),
        ):
            h.AsyncClient.return_value = client
            result = asyncio.run(drv.transcribe(b"AUDIO", {}))

        assert result["text"] == "this is the transcript"


# ── Registry resolution ──────────────────────────────────────────────


class TestRegistryResolution:
    def test_cartesia_tts(self):
        drv = get_tts_driver_for_model("cartesia/sonic-2")
        assert isinstance(drv, CartesiaTTSDriver)
        assert drv.model == "sonic-2"

    def test_cartesia_tts_async(self):
        drv = get_async_tts_driver_for_model("cartesia/sonic-turbo")
        assert isinstance(drv, AsyncCartesiaTTSDriver)
        assert drv.model == "sonic-turbo"

    def test_deepgram_stt(self):
        drv = get_stt_driver_for_model("deepgram/nova-3")
        assert isinstance(drv, DeepgramSTTDriver)
        assert drv.model == "nova-3"

    def test_deepgram_stt_async(self):
        drv = get_async_stt_driver_for_model("deepgram/nova-3")
        assert isinstance(drv, AsyncDeepgramSTTDriver)

    def test_deepgram_tts(self):
        drv = get_tts_driver_for_model("deepgram/aura-2-thalia-en")
        assert isinstance(drv, DeepgramTTSDriver)
        assert drv.model == "aura-2-thalia-en"

    def test_deepgram_tts_async(self):
        drv = get_async_tts_driver_for_model("deepgram/aura-asteria-en")
        assert isinstance(drv, AsyncDeepgramTTSDriver)

    def test_assemblyai_stt(self):
        drv = get_stt_driver_for_model("assemblyai/universal")
        assert isinstance(drv, AssemblyAISTTDriver)
        assert drv.model == "universal"

    def test_assemblyai_stt_async(self):
        drv = get_async_stt_driver_for_model("assemblyai/best")
        assert isinstance(drv, AsyncAssemblyAISTTDriver)
        assert drv.model == "best"
