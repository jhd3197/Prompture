"""Phase 2: RAGPipeline reranker string resolution + Ollama JSON schema wire format.

Two independent feature areas covered:

1. ``RAGPipeline(reranker="provider/model")`` resolves the string via
   :func:`prompture.drivers.rerank_registry.get_rerank_driver_for_model`.
2. ``OllamaDriver`` / ``AsyncOllamaDriver`` pass ``json_schema`` directly to
   Ollama's ``format`` parameter, supporting native constrained decoding
   without falling back to prompted repair.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# 1. Reranker string resolution
# ---------------------------------------------------------------------------


class TestResolveReranker:
    def _import(self):
        from prompture.rag.pipeline import _resolve_reranker

        return _resolve_reranker

    def test_none_passes_through(self) -> None:
        assert self._import()(None, prefer_async=False) is None

    def test_already_instantiated_driver_passes_through(self) -> None:
        fake_driver = object()
        out = self._import()(fake_driver, prefer_async=False)
        assert out is fake_driver

    def test_malformed_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="provider/model"):
            self._import()("cohere", prefer_async=False)

    def test_provider_model_string_calls_sync_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stub the sync registry. Verifies we go through it, not directly to a provider SDK.
        from prompture.drivers import rerank_registry

        sentinel = MagicMock(name="sync_driver")
        monkeypatch.setattr(
            rerank_registry,
            "get_rerank_driver_for_model",
            lambda s: sentinel if s == "cohere/rerank-v3.5" else None,
        )
        out = self._import()("cohere/rerank-v3.5", prefer_async=False)
        assert out is sentinel

    def test_prefer_async_tries_async_then_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Async registry raises -> sync registry must be used.
        from prompture.drivers import rerank_registry

        sync_sentinel = MagicMock(name="sync_driver")
        monkeypatch.setattr(
            rerank_registry,
            "get_async_rerank_driver_for_model",
            MagicMock(side_effect=RuntimeError("no async driver")),
        )
        monkeypatch.setattr(
            rerank_registry,
            "get_rerank_driver_for_model",
            lambda s: sync_sentinel,
        )
        out = self._import()("voyage/rerank-2.5", prefer_async=True)
        assert out is sync_sentinel

    def test_prefer_async_uses_async_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prompture.drivers import rerank_registry

        async_sentinel = MagicMock(name="async_driver")
        sync_sentinel = MagicMock(name="sync_driver")
        monkeypatch.setattr(
            rerank_registry,
            "get_async_rerank_driver_for_model",
            lambda s: async_sentinel,
        )
        monkeypatch.setattr(
            rerank_registry,
            "get_rerank_driver_for_model",
            lambda s: sync_sentinel,
        )
        out = self._import()("cohere/rerank-v3.5", prefer_async=True)
        assert out is async_sentinel


class TestRAGPipelineRerankerWiring:
    def test_pipeline_resolves_string_in_constructor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prompture.drivers import rerank_registry
        from prompture.rag import RAGPipeline

        fake = MagicMock(name="resolved_reranker")
        monkeypatch.setattr(
            rerank_registry,
            "get_rerank_driver_for_model",
            lambda s: fake,
        )
        # Minimal stub retriever / llm.
        retriever = MagicMock(name="retriever")
        retriever.retrieve = MagicMock(return_value=[])
        llm = MagicMock(name="llm", model="fake/model", model_name="fake/model")

        pipeline = RAGPipeline(retriever=retriever, llm=llm, reranker="cohere/rerank-v3.5")
        assert pipeline.reranker is fake

    def test_pipeline_rejects_bad_string(self) -> None:
        from prompture.rag import RAGPipeline

        retriever = MagicMock(name="retriever")
        llm = MagicMock(name="llm")
        with pytest.raises(ValueError, match="provider/model"):
            RAGPipeline(retriever=retriever, llm=llm, reranker="cohere")  # missing /model

    def test_async_pipeline_resolves_string_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from prompture.drivers import rerank_registry
        from prompture.rag import AsyncRAGPipeline

        async_fake = MagicMock(name="async_reranker")
        monkeypatch.setattr(
            rerank_registry,
            "get_async_rerank_driver_for_model",
            lambda s: async_fake,
        )
        retriever = MagicMock(name="retriever")
        llm = MagicMock(name="llm")
        pipeline = AsyncRAGPipeline(retriever=retriever, llm=llm, reranker="cohere/rerank-v3.5")
        assert pipeline.reranker is async_fake


# ---------------------------------------------------------------------------
# 2. Ollama native JSON schema wire format
# ---------------------------------------------------------------------------


class _StubResponse:
    """Stand-in for requests.Response from Ollama."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


def _ollama_chat_response() -> dict[str, Any]:
    return {
        "message": {"role": "assistant", "content": '{"x":1}'},
        "prompt_eval_count": 10,
        "eval_count": 5,
        "done": True,
        "done_reason": "stop",
    }


@pytest.fixture
def ollama_driver_with_capture(monkeypatch: pytest.MonkeyPatch):
    """Yield (driver, captured_payloads_list)."""
    from prompture.drivers import ollama_driver as od

    # Skip the health check probe.
    monkeypatch.setattr(od.OllamaDriver, "_validate_connection", lambda self: None)

    captured: list[dict[str, Any]] = []

    def _stub_post(url: str, *, json: dict[str, Any], **kw: Any) -> _StubResponse:
        captured.append({"url": url, "payload": json})
        return _StubResponse(_ollama_chat_response())

    monkeypatch.setattr(od.requests, "post", _stub_post)

    driver = od.OllamaDriver(endpoint="http://localhost:11434/api/generate", model="llama3.1:8b")
    return driver, captured


class TestOllamaJsonSchema:
    SCHEMA = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }

    def test_capability_flag_set(self) -> None:
        from prompture.drivers.ollama_driver import OllamaDriver

        assert OllamaDriver.supports_json_schema is True
        assert OllamaDriver.supports_json_mode is True

    def test_json_schema_passes_into_format(self, ollama_driver_with_capture) -> None:
        driver, captured = ollama_driver_with_capture
        driver.generate_messages(
            [{"role": "user", "content": "give me x"}],
            options={"json_mode": True, "json_schema": self.SCHEMA},
        )
        assert len(captured) == 1
        payload = captured[0]["payload"]
        assert payload["format"] == self.SCHEMA, (
            f"Expected the JSON schema to be sent verbatim as `format`; got {payload.get('format')!r}"
        )

    def test_schema_alone_implies_json_mode(self, ollama_driver_with_capture) -> None:
        # New behaviour: passing json_schema without an explicit json_mode flag
        # should still enable structured output (schema implies JSON mode).
        driver, captured = ollama_driver_with_capture
        driver.generate_messages(
            [{"role": "user", "content": "give me x"}],
            options={"json_schema": self.SCHEMA},
        )
        payload = captured[0]["payload"]
        assert payload["format"] == self.SCHEMA

    def test_json_mode_without_schema_uses_loose_grammar(self, ollama_driver_with_capture) -> None:
        # Backwards-compat: json_mode=True with no schema -> "json" sentinel.
        driver, captured = ollama_driver_with_capture
        driver.generate_messages(
            [{"role": "user", "content": "anything"}],
            options={"json_mode": True},
        )
        payload = captured[0]["payload"]
        assert payload["format"] == "json"

    def test_no_json_mode_omits_format(self, ollama_driver_with_capture) -> None:
        driver, captured = ollama_driver_with_capture
        driver.generate_messages(
            [{"role": "user", "content": "anything"}],
            options={},
        )
        payload = captured[0]["payload"]
        assert "format" not in payload

    def test_uses_chat_endpoint_not_generate(self, ollama_driver_with_capture) -> None:
        driver, captured = ollama_driver_with_capture
        driver.generate_messages(
            [{"role": "user", "content": "x"}],
            options={"json_schema": self.SCHEMA},
        )
        assert captured[0]["url"].endswith("/api/chat")


class TestAsyncOllamaJsonSchema:
    SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}}

    @pytest.mark.asyncio
    async def test_async_driver_passes_schema_through_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from prompture.drivers import async_ollama_driver as aod

        captured: list[dict[str, Any]] = []

        class _StubResp:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                return _ollama_chat_response()

        class _StubAsyncClient:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> _StubAsyncClient:
                return self

            async def __aexit__(self, *a: Any) -> None:
                return None

            async def post(self, url: str, *, json: dict[str, Any], **kw: Any) -> _StubResp:
                captured.append({"url": url, "payload": json})
                return _StubResp()

        monkeypatch.setattr(aod.httpx, "AsyncClient", _StubAsyncClient)

        driver = aod.AsyncOllamaDriver(endpoint="http://localhost:11434/api/generate", model="llama3.1:8b")
        await driver.generate_messages(
            [{"role": "user", "content": "x"}],
            options={"json_schema": self.SCHEMA},
        )
        assert captured[0]["payload"]["format"] == self.SCHEMA
