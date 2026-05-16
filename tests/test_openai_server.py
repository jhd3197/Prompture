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
