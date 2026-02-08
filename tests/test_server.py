"""Tests for the built-in API server (prompture serve)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from prompture.drivers.async_base import AsyncDriver

# Skip entire module if fastapi is not installed
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from prompture.cli.server import create_app


# ---------------------------------------------------------------------------
# Mock async driver
# ---------------------------------------------------------------------------


class MockAsyncDriver(AsyncDriver):
    supports_messages = True

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": f"Response to: {prompt}",
            "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        }

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        last = messages[-1].get("content", "") if messages else ""
        return {
            "text": f"Response to: {last}",
            "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_driver():
    return MockAsyncDriver()


@pytest.fixture
def app(mock_driver):
    """Create a test app, patching the driver creation."""
    return create_app(model_name="mock/test", system_prompt="You are a test assistant.")


@pytest.fixture
def client(app, mock_driver):
    with patch("prompture.agents.async_conversation.get_async_driver_for_model", return_value=mock_driver):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    def test_basic_chat(self, client):
        resp = client.post("/v1/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "conversation_id" in data
        assert "usage" in data

    def test_chat_returns_conversation_id(self, client):
        resp = client.post("/v1/chat", json={"message": "Hello"})
        conv_id = resp.json()["conversation_id"]
        assert conv_id is not None and len(conv_id) > 0

    def test_chat_with_existing_conversation(self, client):
        r1 = client.post("/v1/chat", json={"message": "First"})
        conv_id = r1.json()["conversation_id"]

        r2 = client.post("/v1/chat", json={"message": "Second", "conversation_id": conv_id})
        assert r2.status_code == 200
        assert r2.json()["conversation_id"] == conv_id


class TestExtractEndpoint:
    def test_basic_extract(self, client, mock_driver):
        # Override to return valid JSON
        async def mock_gen(messages, options):
            return {
                "text": '{"name": "John", "age": 30}',
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
            }

        mock_driver.generate_messages = mock_gen

        resp = client.post("/v1/extract", json={
            "text": "John is 30 years old.",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "json_object" in data
        assert "conversation_id" in data


class TestConversationEndpoints:
    def test_get_conversation(self, client):
        r1 = client.post("/v1/chat", json={"message": "Hello"})
        conv_id = r1.json()["conversation_id"]

        r2 = client.get(f"/v1/conversations/{conv_id}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["conversation_id"] == conv_id
        assert "messages" in data

    def test_get_conversation_not_found(self, client):
        resp = client.get("/v1/conversations/nonexistent")
        assert resp.status_code == 404

    def test_delete_conversation(self, client):
        r1 = client.post("/v1/chat", json={"message": "Hello"})
        conv_id = r1.json()["conversation_id"]

        r2 = client.delete(f"/v1/conversations/{conv_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] == "deleted"

        # Should be gone now
        r3 = client.get(f"/v1/conversations/{conv_id}")
        assert r3.status_code == 404

    def test_delete_conversation_not_found(self, client):
        resp = client.delete("/v1/conversations/nonexistent")
        assert resp.status_code == 404


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
