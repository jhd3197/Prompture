"""Tests for the built-in API server (prompture serve)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from prompture.drivers.async_base import AsyncDriver

# Skip entire module if fastapi is not installed
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from prompture.cli.server import create_app
from prompture.infra.discovery import CodingAgentInfo

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

        resp = client.post(
            "/v1/extract",
            json={
                "text": "John is 30 years old.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            },
        )
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


class TestCodingAgentsEndpoint:
    def test_list_coding_agents(self, client):
        """The server exposes verified coding-agent discovery."""
        with patch(
            "prompture.infra.discovery.get_available_coding_agents",
            return_value=[
                CodingAgentInfo(
                    id="codex",
                    name="Codex CLI",
                    available=True,
                    binary="/usr/local/bin/codex",
                    verified=True,
                    healthy=True,
                )
            ],
        ) as mock_discovery:
            resp = client.get("/v1/coding-agents")

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert data["coding_agents"] == ["codex"]
        assert data["data"][0]["object"] == "coding_agent"
        assert data["data"][0]["id"] == "codex"
        mock_discovery.assert_called_once_with(verify=True)

    def test_list_coding_agents_can_skip_verification(self, client):
        """Apps can request cheap PATH-only discovery when needed."""
        with patch("prompture.infra.discovery.get_available_coding_agents", return_value=[]) as mock_discovery:
            resp = client.get("/v1/coding-agents?verify=false")

        assert resp.status_code == 200
        mock_discovery.assert_called_once_with(verify=False)


class TestCodingAgentRunEndpoint:
    def test_run_endpoint_returns_result_payload(self, client):
        """POST /v1/coding-agents/run returns the serialized CodingAgentRunResult."""
        from prompture.infra.coding_agents import CodingAgentRunResult

        fake_result = CodingAgentRunResult(
            agent="claude",
            command=["claude", "--print", "hi"],
            cwd="/tmp",
            returncode=0,
            output="hello",
            duration_seconds=1.5,
            cost_usd=0.002,
            input_tokens=10,
            output_tokens=5,
        )

        async def _fake_arun(*args, **kwargs):
            return fake_result

        with patch("prompture.infra.coding_agents.arun_coding_agent", side_effect=_fake_arun):
            resp = client.post(
                "/v1/coding-agents/run",
                json={"agent": "claude", "task": "say hi"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "coding_agent_run"
        assert data["agent"] == "claude"
        assert data["output"] == "hello"
        assert data["cost_usd"] == 0.002

    def test_run_endpoint_validation_error_returns_400(self, client):
        async def _raise(*args, **kwargs):
            raise ValueError("bad agent")

        with patch("prompture.infra.coding_agents.arun_coding_agent", side_effect=_raise):
            resp = client.post(
                "/v1/coding-agents/run",
                json={"agent": "nonesuch", "task": "x"},
            )

        assert resp.status_code == 400
        assert "bad agent" in resp.json()["detail"]

    def test_run_endpoint_missing_binary_returns_404(self, client):
        async def _raise(*args, **kwargs):
            raise RuntimeError("'claude' is not installed or not on PATH")

        with patch("prompture.infra.coding_agents.arun_coding_agent", side_effect=_raise):
            resp = client.post(
                "/v1/coding-agents/run",
                json={"agent": "claude", "task": "x"},
            )

        assert resp.status_code == 404
        assert "not installed" in resp.json()["detail"]

    def test_run_endpoint_streams_sse_events(self, client):
        """When stream=true, the endpoint emits SSE-framed events."""
        # SSE streaming needs the `serve` extra; the endpoint answers 501
        # without it, which is correct behaviour rather than a failure.
        pytest.importorskip("sse_starlette")

        from prompture.infra.coding_agent_events import CodingAgentEvent

        async def _fake_stream(*args, **kwargs):
            yield CodingAgentEvent(type="system", raw={"hi": "there"})
            yield CodingAgentEvent(type="message", text="hello")
            yield CodingAgentEvent(type="done", cost_usd=0.001, input_tokens=4, output_tokens=2)

        with patch("prompture.infra.coding_agents.astream_coding_agent", side_effect=_fake_stream):
            resp = client.post(
                "/v1/coding-agents/run",
                json={"agent": "claude", "task": "hi", "stream": True},
            )

        assert resp.status_code == 200
        body = resp.text
        assert "type" in body
        assert '"message"' in body
        assert '"done"' in body
        assert "stream_end" in body
