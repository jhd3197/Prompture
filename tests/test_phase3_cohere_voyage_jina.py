"""Unit tests for Phase 3: Cohere LLM and Cohere/Voyage/Jina embedding drivers.

All HTTP calls are mocked — no live network access required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from prompture.drivers import get_driver_for_model
from prompture.drivers.async_cohere_driver import AsyncCohereDriver
from prompture.drivers.async_cohere_embedding_driver import AsyncCohereEmbeddingDriver
from prompture.drivers.async_jina_embedding_driver import AsyncJinaEmbeddingDriver
from prompture.drivers.async_voyage_embedding_driver import AsyncVoyageEmbeddingDriver
from prompture.drivers.cohere_driver import CohereDriver
from prompture.drivers.cohere_embedding_driver import CohereEmbeddingDriver
from prompture.drivers.embedding_registry import get_embedding_driver_for_model
from prompture.drivers.jina_embedding_driver import JinaEmbeddingDriver
from prompture.drivers.voyage_embedding_driver import VoyageEmbeddingDriver

# ── HTTP mock helpers ────────────────────────────────────────────────────


def _mock_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = payload
    mock.raise_for_status = MagicMock()
    mock.status_code = status
    mock.text = ""
    return mock


def _cohere_chat_payload(text: str = "Hello back!") -> dict[str, Any]:
    return {
        "id": "resp-123",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
        "finish_reason": "complete",
        "usage": {
            "billed_units": {"input_tokens": 10, "output_tokens": 20},
            "tokens": {"input_tokens": 10, "output_tokens": 20},
        },
    }


def _cohere_chat_tool_payload() -> dict[str, Any]:
    return {
        "id": "resp-456",
        "message": {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        },
        "finish_reason": "tool_calls",
        "usage": {"billed_units": {"input_tokens": 30, "output_tokens": 5}},
    }


def _cohere_embed_payload(n: int = 2, dim: int = 4) -> dict[str, Any]:
    vectors = [[float(i) / 10 for _ in range(dim)] for i in range(n)]
    return {
        "id": "embed-1",
        "embeddings": {"float": vectors},
        "meta": {"billed_units": {"input_tokens": 1200}},
    }


def _voyage_embed_payload(n: int = 2, dim: int = 4) -> dict[str, Any]:
    return {
        "data": [{"embedding": [float(i) / 10] * dim, "index": i} for i in range(n)],
        "model": "voyage-3.5",
        "usage": {"total_tokens": 1800},
    }


def _jina_embed_payload(n: int = 2, dim: int = 4) -> dict[str, Any]:
    return {
        "data": [{"embedding": [float(i) / 10] * dim, "index": i, "object": "embedding"} for i in range(n)],
        "model": "jina-embeddings-v3",
        "usage": {"prompt_tokens": 2200, "total_tokens": 2200},
    }


# ─────────────────────────────── Cohere LLM ───────────────────────────────


class TestCohereLLM:
    def test_generate_returns_text_and_tokens(self):
        d = CohereDriver(api_key="sk-test", model="command-r-plus")
        with patch(
            "prompture.drivers.cohere_driver.requests.post",
            return_value=_mock_response(_cohere_chat_payload("Bonjour!")),
        ) as mp:
            result = d.generate("Hello?", {})
        assert result["text"] == "Bonjour!"
        meta = result["meta"]
        assert meta["prompt_tokens"] == 10
        assert meta["completion_tokens"] == 20
        assert meta["total_tokens"] == 30
        # Cost should be > 0 thanks to cohere.json rates.
        assert meta["cost"] > 0
        # Posted to v2 chat endpoint
        called_url = mp.call_args.args[0]
        assert called_url.endswith("/v2/chat")

    def test_generate_messages(self):
        d = CohereDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.cohere_driver.requests.post",
            return_value=_mock_response(_cohere_chat_payload("ok")),
        ):
            result = d.generate_messages([{"role": "user", "content": "Hi"}], {"max_tokens": 64})
        assert result["text"] == "ok"

    def test_tool_use(self):
        d = CohereDriver(api_key="sk-test")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "...",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        with patch(
            "prompture.drivers.cohere_driver.requests.post",
            return_value=_mock_response(_cohere_chat_tool_payload()),
        ):
            result = d.generate_messages_with_tools([{"role": "user", "content": "Weather in Paris?"}], tools, {})
        assert result["stop_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["name"] == "get_weather"
        assert tc["arguments"] == {"city": "Paris"}

    def test_http_error_surfaces_body(self):
        import requests as _rq

        d = CohereDriver(api_key="sk-test")
        fake_resp = MagicMock()
        fake_resp.text = '{"error": "missing required field"}'
        fake_resp.status_code = 400
        err = _rq.exceptions.HTTPError("400")
        err.response = fake_resp
        with patch(
            "prompture.drivers.cohere_driver.requests.post",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError, match="missing required field"):
                d.generate("hi", {})

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="COHERE_API_KEY"):
            CohereDriver(api_key=None)

    def test_async_generate(self):
        d = AsyncCohereDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None, timeout=None):
            return _mock_response(_cohere_chat_payload("async-hi"))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            result = asyncio.run(d.generate("hello", {}))
        assert result["text"] == "async-hi"
        assert result["meta"]["total_tokens"] == 30


# ─────────────────────────── Cohere embeddings ────────────────────────────


class TestCohereEmbedding:
    def test_embed_returns_vectors_and_usage(self):
        d = CohereEmbeddingDriver(api_key="sk-test", model="embed-v4.0")
        with patch(
            "prompture.drivers.cohere_embedding_driver.requests.post",
            return_value=_mock_response(_cohere_embed_payload(2, 4)),
        ) as mp:
            result = d.embed(["doc one", "doc two"], {})
        assert len(result["embeddings"]) == 2
        assert all(isinstance(v, list) for v in result["embeddings"])
        meta = result["meta"]
        assert meta["model_name"] == "cohere/embed-v4.0"
        assert meta["dimensions"] == 4
        assert meta["total_tokens"] == 1200
        assert meta["cost"] > 0
        # input_type default forwarded
        sent_payload = mp.call_args.kwargs["json"]
        assert sent_payload["input_type"] == "search_document"
        assert sent_payload["embedding_types"] == ["float"]

    def test_embed_input_type_override(self):
        d = CohereEmbeddingDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.cohere_embedding_driver.requests.post",
            return_value=_mock_response(_cohere_embed_payload(1, 4)),
        ) as mp:
            d.embed(["query text"], {"input_type": "search_query"})
        assert mp.call_args.kwargs["json"]["input_type"] == "search_query"

    def test_async_embed(self):
        d = AsyncCohereEmbeddingDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_cohere_embed_payload(2, 4))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            result = asyncio.run(d.embed(["a", "b"], {}))
        assert len(result["embeddings"]) == 2
        assert result["meta"]["total_tokens"] == 1200


# ─────────────────────────── Voyage embeddings ────────────────────────────


class TestVoyageEmbedding:
    def test_embed_returns_vectors_and_usage(self):
        d = VoyageEmbeddingDriver(api_key="sk-test", model="voyage-3.5")
        with patch(
            "prompture.drivers.voyage_embedding_driver.requests.post",
            return_value=_mock_response(_voyage_embed_payload(2, 4)),
        ) as mp:
            result = d.embed(["doc one", "doc two"], {})
        assert len(result["embeddings"]) == 2
        meta = result["meta"]
        assert meta["model_name"] == "voyage/voyage-3.5"
        assert meta["dimensions"] == 4
        assert meta["total_tokens"] == 1800
        assert meta["cost"] > 0
        sent_payload = mp.call_args.kwargs["json"]
        assert sent_payload["input_type"] == "document"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
            VoyageEmbeddingDriver(api_key=None)

    def test_async_embed(self):
        d = AsyncVoyageEmbeddingDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_voyage_embed_payload(3, 4))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            result = asyncio.run(d.embed(["a", "b", "c"], {"input_type": "query"}))
        assert len(result["embeddings"]) == 3
        assert result["meta"]["total_tokens"] == 1800


# ──────────────────────────── Jina embeddings ─────────────────────────────


class TestJinaEmbedding:
    def test_embed_returns_vectors_and_usage(self):
        d = JinaEmbeddingDriver(api_key="sk-test", model="jina-embeddings-v3")
        with patch(
            "prompture.drivers.jina_embedding_driver.requests.post",
            return_value=_mock_response(_jina_embed_payload(2, 4)),
        ) as mp:
            result = d.embed(["doc one", "doc two"], {})
        assert len(result["embeddings"]) == 2
        meta = result["meta"]
        assert meta["model_name"] == "jina/jina-embeddings-v3"
        assert meta["dimensions"] == 4
        assert meta["total_tokens"] == 2200
        assert meta["cost"] > 0
        sent_payload = mp.call_args.kwargs["json"]
        assert sent_payload["task"] == "retrieval.passage"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="JINA_API_KEY"):
            JinaEmbeddingDriver(api_key=None)

    def test_async_embed(self):
        d = AsyncJinaEmbeddingDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_jina_embed_payload(2, 4))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            result = asyncio.run(d.embed(["a", "b"], {}))
        assert len(result["embeddings"]) == 2


# ─────────────────────────── Registry routing ────────────────────────────


class TestRegistryRouting:
    def test_get_driver_for_model_cohere(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "sk-test")
        driver = get_driver_for_model("cohere/command-r-plus")
        assert isinstance(driver, CohereDriver)
        assert driver.model == "command-r-plus"

    def test_get_embedding_driver_for_model_cohere(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "sk-test")
        driver = get_embedding_driver_for_model("cohere/embed-v4.0")
        assert isinstance(driver, CohereEmbeddingDriver)
        assert driver.model == "embed-v4.0"

    def test_get_embedding_driver_for_model_voyage(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "sk-test")
        driver = get_embedding_driver_for_model("voyage/voyage-3.5")
        assert isinstance(driver, VoyageEmbeddingDriver)
        assert driver.model == "voyage-3.5"

    def test_get_embedding_driver_for_model_jina(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "sk-test")
        driver = get_embedding_driver_for_model("jina/jina-embeddings-v3")
        assert isinstance(driver, JinaEmbeddingDriver)
        assert driver.model == "jina-embeddings-v3"


# ─────────────────────── Phase 3 rerank pricing fill-in ────────────────────


class TestRerankPricingPopulated:
    """Phase 3 also populated rerank cost blocks in cohere/voyage/jina.json."""

    def test_cohere_rerank_cost_computed(self, monkeypatch):
        from prompture.drivers.cohere_rerank_driver import CohereRerankDriver

        monkeypatch.setenv("COHERE_API_KEY", "sk-test")
        d = CohereRerankDriver(model="rerank-v3.5")
        with patch(
            "prompture.drivers.cohere_rerank_driver.requests.post",
            return_value=_mock_response(
                {
                    "results": [{"index": 0, "relevance_score": 0.9}],
                    "meta": {"billed_units": {"search_units": 1}},
                }
            ),
        ):
            d.rerank("q", ["a", "b"])
        assert d.last_usage["pricing_unknown"] is False
        assert d.last_usage["cost"] > 0.0

    def test_voyage_rerank_cost_computed(self, monkeypatch):
        from prompture.drivers.voyage_rerank_driver import VoyageRerankDriver

        monkeypatch.setenv("VOYAGE_API_KEY", "sk-test")
        d = VoyageRerankDriver(model="rerank-2.5")
        with patch(
            "prompture.drivers.voyage_rerank_driver.requests.post",
            return_value=_mock_response(
                {
                    "data": [{"index": 0, "relevance_score": 0.9}],
                    "usage": {"total_tokens": 1000},
                }
            ),
        ):
            d.rerank("q", ["a", "b"])
        assert d.last_usage["pricing_unknown"] is False
        assert d.last_usage["cost"] > 0.0

    def test_jina_rerank_cost_computed(self, monkeypatch):
        from prompture.drivers.jina_rerank_driver import JinaRerankDriver

        monkeypatch.setenv("JINA_API_KEY", "sk-test")
        d = JinaRerankDriver(model="jina-reranker-v2-base-multilingual")
        with patch(
            "prompture.drivers.jina_rerank_driver.requests.post",
            return_value=_mock_response(
                {
                    "results": [{"index": 0, "relevance_score": 0.9}],
                    "usage": {"total_tokens": 1000},
                }
            ),
        ):
            d.rerank("q", ["a", "b"])
        assert d.last_usage["pricing_unknown"] is False
        assert d.last_usage["cost"] > 0.0
