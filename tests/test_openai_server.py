"""Tests for the OpenAI-compatible endpoints exposed by ``prompture serve``.

Exercises ``/v1/chat/completions`` (sync + streaming), ``/v1/completions``,
``/v1/embeddings``, ``/v1/models``, bearer auth, and model allowlist.

All tests use a mocked async driver — no live network.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from prompture.cli.server import create_app
from prompture.drivers.async_base import AsyncDriver

# ---------------------------------------------------------------------------
# Mock drivers
# ---------------------------------------------------------------------------


class MockAsyncDriver(AsyncDriver):
    supports_messages = True

    def __init__(self, text: str = "Hello back!"):
        self.text = text
        self.last_options: dict[str, Any] | None = None
        self.last_messages: list[dict[str, Any]] | None = None

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.last_options = dict(options)
        return {
            "text": self.text,
            "meta": {
                "prompt_tokens": 12,
                "completion_tokens": 6,
                "total_tokens": 18,
                "cost": 0.001,
            },
        }

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        self.last_options = dict(options)
        self.last_messages = list(messages)
        return {
            "text": self.text,
            "meta": {
                "prompt_tokens": 12,
                "completion_tokens": 6,
                "total_tokens": 18,
                "cost": 0.001,
            },
        }


class MockAsyncEmbeddingDriver:
    """Minimal stand-in matching the async embedding contract."""

    async def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        return {
            "embeddings": [[0.1, 0.2, 0.3] for _ in texts],
            "meta": {
                "model_name": "mock/embed",
                "dimensions": 3,
                "total_tokens": len(texts) * 4,
                "cost": 0.0,
            },
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_driver():
    return MockAsyncDriver()


@pytest.fixture
def app():
    return create_app(model_name="mock/test", system_prompt="Test system.")


@pytest.fixture
def client(app, mock_driver):
    with patch(
        "prompture.agents.async_conversation.get_async_driver_for_model",
        return_value=mock_driver,
    ):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def authed_client(mock_driver):
    app = create_app(model_name="mock/test", api_key="secret-123")
    with patch(
        "prompture.agents.async_conversation.get_async_driver_for_model",
        return_value=mock_driver,
    ):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# /v1/chat/completions — non-streaming
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_basic_shape(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock/test",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "mock/test"
        assert data["id"].startswith("chatcmpl-")
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello back!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["total_tokens"] == 18

    def test_default_model_when_omitted(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["model"] == "mock/test"

    def test_system_message_is_extracted(self, client, mock_driver):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "Hi"},
                ],
            },
        )
        assert resp.status_code == 200
        # System prompt must appear as the first driver-visible message.
        assert mock_driver.last_messages is not None
        assert mock_driver.last_messages[0]["role"] == "system"
        assert mock_driver.last_messages[0]["content"] == "You are terse."

    def test_prior_assistant_messages_replayed(self, client, mock_driver):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4."},
                    {"role": "user", "content": "And 3+3?"},
                ],
            },
        )
        assert resp.status_code == 200
        # The full prior turn should be in the driver-visible message list.
        msgs = mock_driver.last_messages
        assert any(m["role"] == "assistant" and m["content"] == "4." for m in msgs)
        # Last user message is the most recent one.
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert user_msgs[-1]["content"] == "And 3+3?"

    def test_multipart_text_content_is_flattened(self, client, mock_driver):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Part one."},
                            {"type": "text", "text": "Part two."},
                        ],
                    }
                ],
            },
        )
        assert resp.status_code == 200
        user_msg = [m for m in mock_driver.last_messages if m["role"] == "user"][-1]
        assert "Part one." in user_msg["content"]
        assert "Part two." in user_msg["content"]

    def test_options_are_forwarded(self, client, mock_driver):
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 200,
            },
        )
        assert mock_driver.last_options is not None
        assert mock_driver.last_options.get("temperature") == 0.3
        assert mock_driver.last_options.get("top_p") == 0.9
        assert mock_driver.last_options.get("max_tokens") == 200

    def test_client_tools_forwarded_to_driver(self, client, mock_driver):
        tool_schema = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Weather?"}],
                "tools": [tool_schema],
                "tool_choice": "auto",
            },
        )
        assert mock_driver.last_options.get("tools") == [tool_schema]
        assert mock_driver.last_options.get("tool_choice") == "auto"

    def test_empty_messages_rejected(self, client):
        resp = client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code in (400, 422)

    def test_messages_without_user_role_rejected(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "system", "content": "hi"}]},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /v1/chat/completions — streaming
# ---------------------------------------------------------------------------


class TestChatCompletionsStreaming:
    def test_stream_produces_sse_chunks(self, client):
        # SSE streaming needs the `serve` extra; the endpoint answers 501
        # without it, which is correct behaviour rather than a failure.
        pytest.importorskip("sse_starlette")
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = resp.read().decode("utf-8")

        assert "[DONE]" in body
        # Each SSE event begins with "data:"; parse the first that carries a delta.
        chunks: list[dict] = []
        for line in body.splitlines():
            if line.startswith("data:") and "[DONE]" not in line:
                payload = line[len("data:") :].strip()
                if payload:
                    chunks.append(json.loads(payload))

        assert chunks  # got at least one delta
        assert chunks[0]["object"] == "chat.completion.chunk"
        # First chunk announces the role
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        # Some later chunk carries content
        assert any(c["choices"][0]["delta"].get("content") for c in chunks)
        # Final chunk has finish_reason
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


class TestCompletions:
    def test_basic_completion(self, client):
        resp = client.post(
            "/v1/completions",
            json={"model": "mock/test", "prompt": "Write a haiku."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "Hello back!"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_list_prompt_joined(self, client):
        resp = client.post(
            "/v1/completions",
            json={"prompt": ["line one", "line two"]},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_single_string(self, client):
        with patch(
            "prompture.drivers.embedding_registry.get_async_embedding_driver_for_model",
            return_value=MockAsyncEmbeddingDriver(),
        ):
            resp = client.post(
                "/v1/embeddings",
                json={"model": "mock/embed", "input": "hello world"},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert data["data"][0]["index"] == 0
        assert data["usage"]["total_tokens"] > 0

    def test_list_input(self, client):
        with patch(
            "prompture.drivers.embedding_registry.get_async_embedding_driver_for_model",
            return_value=MockAsyncEmbeddingDriver(),
        ):
            resp = client.post(
                "/v1/embeddings",
                json={"model": "mock/embed", "input": ["one", "two", "three"]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        assert [d["index"] for d in data["data"]] == [0, 1, 2]


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


class TestListModels:
    def test_shape(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        for m in data["data"]:
            assert m["object"] == "model"
            assert "id" in m
            assert "owned_by" in m

    def test_allowlist_filters_results(self, mock_driver):
        app = create_app(
            model_name="mock/test",
            allowed_models=["openai/gpt-4o-mini"],
        )
        with patch(
            "prompture.agents.async_conversation.get_async_driver_for_model",
            return_value=mock_driver,
        ):
            with TestClient(app) as c:
                resp = c.get("/v1/models")
                data = resp.json()
                names = [m["id"] for m in data["data"]]
                assert all(n == "openai/gpt-4o-mini" for n in names)


# ---------------------------------------------------------------------------
# Bearer auth
# ---------------------------------------------------------------------------


class TestBearerAuth:
    def test_missing_token_rejected(self, authed_client):
        resp = authed_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, authed_client):
        resp = authed_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    def test_correct_token_accepted(self, authed_client):
        resp = authed_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer secret-123"},
        )
        assert resp.status_code == 200

    def test_health_endpoint_is_public(self, authed_client):
        resp = authed_client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Model allowlist
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Vision content parts
# ---------------------------------------------------------------------------


class TestVisionContent:
    def test_image_url_part_forwarded_to_driver(self, client, mock_driver):
        # Patch ask() so we can capture how the conversation forwarded
        # the images argument.
        from prompture.agents.async_conversation import AsyncConversation

        captured: dict[str, Any] = {}

        original_ask = AsyncConversation.ask

        async def _capturing_ask(self, content, options=None, images=None, **kw):
            captured["content"] = content
            captured["images"] = images
            return await original_ask(self, content, options=options, images=images, **kw)

        with patch.object(AsyncConversation, "ask", _capturing_ask):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/cat.png"},
                                },
                            ],
                        }
                    ],
                },
            )
        assert resp.status_code == 200, resp.text
        assert captured["images"] == ["https://example.com/cat.png"]
        assert "What's in this image?" in captured["content"]

    def test_text_only_passes_no_images(self, client, mock_driver):
        from prompture.agents.async_conversation import AsyncConversation

        captured: dict[str, Any] = {}
        original_ask = AsyncConversation.ask

        async def _capturing_ask(self, content, options=None, images=None, **kw):
            captured["images"] = images
            return await original_ask(self, content, options=options, images=images, **kw)

        with patch.object(AsyncConversation, "ask", _capturing_ask):
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Plain text only."}]},
            )
        assert captured["images"] in (None, [])


# ---------------------------------------------------------------------------
# /v1/images/generations
# ---------------------------------------------------------------------------


class _MockAsyncImageGenDriver:
    """Stand-in async image-generation driver."""

    def __init__(self, with_url: bool = True):
        self.with_url = with_url

    async def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        from prompture.media.image import ImageContent

        img = ImageContent(
            data="ZmFrZS1iNjQ=",  # 'fake-b64'
            media_type="image/png",
            source_type="url" if self.with_url else "base64",
            url="https://example.com/generated.png" if self.with_url else None,
        )
        return {
            "images": [img for _ in range(options.get("n", 1))],
            "meta": {
                "image_count": options.get("n", 1),
                "model_name": "mock/img",
                "cost": 0.04,
                "revised_prompt": "revised: " + prompt,
            },
        }


class TestImagesGenerations:
    def test_url_response_format(self, client):
        with patch(
            "prompture.drivers.img_gen_registry.get_async_img_gen_driver_for_model",
            return_value=_MockAsyncImageGenDriver(with_url=True),
        ):
            resp = client.post(
                "/v1/images/generations",
                json={
                    "model": "mock/img",
                    "prompt": "A cat in a top hat",
                    "n": 2,
                    "response_format": "url",
                },
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["model"] == "mock/img"
        assert len(data["data"]) == 2
        assert data["data"][0]["url"].startswith("https://")
        assert data["data"][0]["revised_prompt"].startswith("revised:")

    def test_b64_response_format(self, client):
        with patch(
            "prompture.drivers.img_gen_registry.get_async_img_gen_driver_for_model",
            return_value=_MockAsyncImageGenDriver(with_url=True),
        ):
            resp = client.post(
                "/v1/images/generations",
                json={
                    "model": "mock/img",
                    "prompt": "A cat in a top hat",
                    "response_format": "b64_json",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["data"][0]["b64_json"] == "ZmFrZS1iNjQ="

    def test_data_uri_fallback_when_no_url(self, client):
        with patch(
            "prompture.drivers.img_gen_registry.get_async_img_gen_driver_for_model",
            return_value=_MockAsyncImageGenDriver(with_url=False),
        ):
            resp = client.post(
                "/v1/images/generations",
                json={"model": "mock/img", "prompt": "x"},
            )
        assert resp.status_code == 200
        assert resp.json()["data"][0]["url"].startswith("data:image/png;base64,")

    def test_unknown_model_returns_400(self, client):
        with patch(
            "prompture.drivers.img_gen_registry.get_async_img_gen_driver_for_model",
            side_effect=ValueError("no such provider"),
        ):
            resp = client.post(
                "/v1/images/generations",
                json={"model": "nope/x", "prompt": "y"},
            )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /v1/audio/speech
# ---------------------------------------------------------------------------


class _MockAsyncTTSDriver:
    async def synthesize(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "audio": b"FAKE-AUDIO-BYTES",
            "media_type": "audio/mpeg",
            "meta": {"cost": 0.0001, "model_name": "mock/tts"},
        }


class TestAudioSpeech:
    def test_returns_audio_bytes(self, client):
        with patch(
            "prompture.drivers.audio_registry.get_async_tts_driver_for_model",
            return_value=_MockAsyncTTSDriver(),
        ):
            resp = client.post(
                "/v1/audio/speech",
                json={"model": "mock/tts", "input": "Hello world.", "voice": "alloy"},
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("audio/")
        assert resp.content == b"FAKE-AUDIO-BYTES"

    def test_unknown_model_returns_400(self, client):
        with patch(
            "prompture.drivers.audio_registry.get_async_tts_driver_for_model",
            side_effect=ValueError("no tts here"),
        ):
            resp = client.post(
                "/v1/audio/speech",
                json={"model": "nope/x", "input": "hi"},
            )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /v1/audio/transcriptions
# ---------------------------------------------------------------------------


class _MockAsyncSTTDriver:
    async def transcribe(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": f"transcribed {len(audio)} bytes",
            "language": options.get("language", "en"),
            "segments": [],
            "meta": {"cost": 0.0001},
        }


class TestAudioTranscriptions:
    def test_json_response(self, client):
        with patch(
            "prompture.drivers.audio_registry.get_async_stt_driver_for_model",
            return_value=_MockAsyncSTTDriver(),
        ):
            resp = client.post(
                "/v1/audio/transcriptions",
                data={"model": "mock/stt", "language": "en"},
                files={"file": ("clip.wav", b"AUDIODATA", "audio/wav")},
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["text"].startswith("transcribed")
        assert data["language"] == "en"

    def test_text_response_format(self, client):
        with patch(
            "prompture.drivers.audio_registry.get_async_stt_driver_for_model",
            return_value=_MockAsyncSTTDriver(),
        ):
            resp = client.post(
                "/v1/audio/transcriptions",
                data={"model": "mock/stt", "response_format": "text"},
                files={"file": ("clip.wav", b"AUDIODATA", "audio/wav")},
            )
        assert resp.status_code == 200
        assert resp.text.startswith("transcribed")

    def test_missing_file_returns_422(self, client):
        # No 'file' field — FastAPI multipart validation rejects.
        resp = client.post(
            "/v1/audio/transcriptions",
            data={"model": "mock/stt"},
        )
        assert resp.status_code in (400, 422)


class TestModelAllowlist:
    def test_blocks_disallowed_model(self, mock_driver):
        app = create_app(
            model_name="mock/test",
            allowed_models=["mock/test"],
        )
        with patch(
            "prompture.agents.async_conversation.get_async_driver_for_model",
            return_value=mock_driver,
        ):
            with TestClient(app) as c:
                resp = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai/gpt-4o",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
                assert resp.status_code == 403

    def test_allows_listed_model(self, mock_driver):
        app = create_app(
            model_name="mock/test",
            allowed_models=["mock/test"],
        )
        with patch(
            "prompture.agents.async_conversation.get_async_driver_for_model",
            return_value=mock_driver,
        ):
            with TestClient(app) as c:
                resp = c.post(
                    "/v1/chat/completions",
                    json={
                        "model": "mock/test",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
                assert resp.status_code == 200
