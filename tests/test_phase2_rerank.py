"""Unit tests for Phase 2 rerank drivers: Cohere, Voyage AI, Jina AI.

All HTTP calls are mocked — no live network access required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from prompture.drivers.async_cohere_rerank_driver import AsyncCohereRerankDriver
from prompture.drivers.async_jina_rerank_driver import AsyncJinaRerankDriver
from prompture.drivers.async_voyage_rerank_driver import AsyncVoyageRerankDriver
from prompture.drivers.cohere_rerank_driver import CohereRerankDriver
from prompture.drivers.jina_rerank_driver import JinaRerankDriver
from prompture.drivers.rerank_base import (
    AsyncRerankDriver,
    RerankDriver,
    RerankResult,
)
from prompture.drivers.rerank_registry import (
    ASYNC_RERANK_DRIVER_REGISTRY,
    RERANK_DRIVER_REGISTRY,
    get_async_rerank_driver_for_model,
    get_rerank_driver_for_model,
)
from prompture.drivers.voyage_rerank_driver import VoyageRerankDriver

DOCS = [
    "Berlin is the capital of Germany.",
    "Paris is the capital of France.",
    "Madrid is in Spain.",
]


# ── Helpers to build provider-shaped fake HTTP responses ────────────────────


def _mock_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a fake requests.Response-like object."""
    mock = MagicMock()
    mock.json.return_value = payload
    mock.raise_for_status = MagicMock()
    mock.status_code = status
    mock.text = ""
    return mock


def _cohere_payload(docs: list[str], *, include_documents: bool) -> dict[str, Any]:
    # Returned results sorted by relevance_score descending (highest first).
    results: list[dict[str, Any]] = [
        {"index": 1, "relevance_score": 0.95},
        {"index": 2, "relevance_score": 0.60},
        {"index": 0, "relevance_score": 0.20},
    ]
    if include_documents:
        for r in results:
            r["document"] = {"text": docs[r["index"]]}
    return {"results": results, "meta": {"billed_units": {"search_units": 1}}}


def _voyage_payload(docs: list[str], *, include_documents: bool) -> dict[str, Any]:
    results: list[dict[str, Any]] = [
        {"index": 1, "relevance_score": 0.91},
        {"index": 2, "relevance_score": 0.50},
        {"index": 0, "relevance_score": 0.10},
    ]
    if include_documents:
        for r in results:
            r["document"] = docs[r["index"]]
    return {"data": results, "usage": {"total_tokens": 42}}


def _jina_payload(docs: list[str], *, include_documents: bool) -> dict[str, Any]:
    results: list[dict[str, Any]] = [
        {"index": 1, "relevance_score": 0.88},
        {"index": 2, "relevance_score": 0.55},
        {"index": 0, "relevance_score": 0.15},
    ]
    if include_documents:
        for r in results:
            r["document"] = {"text": docs[r["index"]]}
    return {"results": results, "usage": {"total_tokens": 30}}


# ─────────────────────────────── Cohere ───────────────────────────────────


class TestCohereRerank:
    def test_rerank_returns_results_sorted_by_score(self):
        d = CohereRerankDriver(api_key="sk-test", model="rerank-v3.5")
        with patch(
            "prompture.drivers.cohere_rerank_driver.requests.post",
            return_value=_mock_response(_cohere_payload(DOCS, include_documents=False)),
        ) as mp:
            results = d.rerank("capital of France?", DOCS, top_n=3)
        assert len(results) == 3
        assert all(isinstance(r, RerankResult) for r in results)
        # Sorted descending by relevance_score
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)
        # Top hit is the "Paris" document (index 1)
        assert results[0].index == 1
        # Posted to the correct URL
        called_url = mp.call_args.args[0]
        assert called_url == CohereRerankDriver.BASE_URL
        # last_usage populated
        assert d.last_usage["model_name"] == "cohere/rerank-v3.5"
        assert d.last_usage["search_units"] == 1
        assert d.last_usage["pricing_unknown"] is True

    def test_return_documents_false_leaves_document_none(self):
        d = CohereRerankDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.cohere_rerank_driver.requests.post",
            return_value=_mock_response(_cohere_payload(DOCS, include_documents=False)),
        ):
            results = d.rerank("q", DOCS, return_documents=False)
        assert all(r.document is None for r in results)

    def test_return_documents_true_populates_document(self):
        d = CohereRerankDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.cohere_rerank_driver.requests.post",
            return_value=_mock_response(_cohere_payload(DOCS, include_documents=True)),
        ):
            results = d.rerank("q", DOCS, return_documents=True)
        assert results[0].document == DOCS[1]
        assert all(r.document is not None for r in results)

    def test_http_error_raises_runtime_error(self):
        d = CohereRerankDriver(api_key="sk-test")
        fake_resp = MagicMock()
        fake_resp.text = '{"error": "bad request"}'
        fake_resp.status_code = 400

        def _raise():
            raise __import__("requests").exceptions.HTTPError("400", response=fake_resp)

        mock_post = MagicMock()
        mock_post.return_value.raise_for_status = _raise
        mock_post.return_value.text = '{"error": "bad request"}'
        # Easier path: have post raise directly.
        import requests as _rq

        err = _rq.exceptions.HTTPError("400 Bad Request")
        err.response = fake_resp
        with patch(
            "prompture.drivers.cohere_rerank_driver.requests.post",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError, match="bad request"):
                d.rerank("q", DOCS)

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="COHERE_API_KEY"):
            CohereRerankDriver(api_key=None)

    def test_async_rerank(self):
        import asyncio

        d = AsyncCohereRerankDriver(api_key="sk-test", model="rerank-v3.5")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_cohere_payload(DOCS, include_documents=True))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            results = asyncio.run(d.rerank("q", DOCS, top_n=2, return_documents=True))
        assert results[0].index == 1
        assert results[0].document == DOCS[1]


# ─────────────────────────────── Voyage ───────────────────────────────────


class TestVoyageRerank:
    def test_rerank_returns_results_sorted_by_score(self):
        d = VoyageRerankDriver(api_key="sk-test", model="rerank-2.5")
        with patch(
            "prompture.drivers.voyage_rerank_driver.requests.post",
            return_value=_mock_response(_voyage_payload(DOCS, include_documents=False)),
        ) as mp:
            results = d.rerank("q", DOCS, top_n=3)
        assert len(results) == 3
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].index == 1
        assert mp.call_args.args[0] == VoyageRerankDriver.BASE_URL
        assert d.last_usage["total_tokens"] == 42
        assert d.last_usage["model_name"] == "voyage/rerank-2.5"

    def test_return_documents_true(self):
        d = VoyageRerankDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.voyage_rerank_driver.requests.post",
            return_value=_mock_response(_voyage_payload(DOCS, include_documents=True)),
        ):
            results = d.rerank("q", DOCS, return_documents=True)
        assert results[0].document == DOCS[1]

    def test_return_documents_false(self):
        d = VoyageRerankDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.voyage_rerank_driver.requests.post",
            return_value=_mock_response(_voyage_payload(DOCS, include_documents=False)),
        ):
            results = d.rerank("q", DOCS, return_documents=False)
        assert all(r.document is None for r in results)

    def test_http_error_raises_runtime_error(self):
        import requests as _rq

        d = VoyageRerankDriver(api_key="sk-test")
        fake_resp = MagicMock()
        fake_resp.text = '{"error": "unauthorized"}'
        fake_resp.status_code = 401
        err = _rq.exceptions.HTTPError("401")
        err.response = fake_resp
        with patch(
            "prompture.drivers.voyage_rerank_driver.requests.post",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError, match="unauthorized"):
                d.rerank("q", DOCS)

    def test_async_rerank(self):
        import asyncio

        d = AsyncVoyageRerankDriver(api_key="sk-test", model="rerank-2.5")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_voyage_payload(DOCS, include_documents=False))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            results = asyncio.run(d.rerank("q", DOCS))
        assert results[0].index == 1


# ──────────────────────────────── Jina ────────────────────────────────────


class TestJinaRerank:
    def test_rerank_returns_results_sorted_by_score(self):
        d = JinaRerankDriver(api_key="sk-test", model="jina-reranker-v2-base-multilingual")
        with patch(
            "prompture.drivers.jina_rerank_driver.requests.post",
            return_value=_mock_response(_jina_payload(DOCS, include_documents=False)),
        ) as mp:
            results = d.rerank("q", DOCS, top_n=3)
        assert len(results) == 3
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].index == 1
        assert mp.call_args.args[0] == JinaRerankDriver.BASE_URL
        assert d.last_usage["total_tokens"] == 30

    def test_return_documents_true(self):
        d = JinaRerankDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.jina_rerank_driver.requests.post",
            return_value=_mock_response(_jina_payload(DOCS, include_documents=True)),
        ):
            results = d.rerank("q", DOCS, return_documents=True)
        assert results[0].document == DOCS[1]

    def test_return_documents_false(self):
        d = JinaRerankDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.jina_rerank_driver.requests.post",
            return_value=_mock_response(_jina_payload(DOCS, include_documents=False)),
        ):
            results = d.rerank("q", DOCS, return_documents=False)
        assert all(r.document is None for r in results)

    def test_http_error_raises_runtime_error(self):
        import requests as _rq

        d = JinaRerankDriver(api_key="sk-test")
        fake_resp = MagicMock()
        fake_resp.text = '{"detail": "rate limited"}'
        fake_resp.status_code = 429
        err = _rq.exceptions.HTTPError("429")
        err.response = fake_resp
        with patch(
            "prompture.drivers.jina_rerank_driver.requests.post",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError, match="rate limited"):
                d.rerank("q", DOCS)

    def test_async_rerank(self):
        import asyncio

        d = AsyncJinaRerankDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_jina_payload(DOCS, include_documents=True))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            results = asyncio.run(d.rerank("q", DOCS, return_documents=True))
        assert results[0].document == DOCS[1]


# ─────────────────────────── Registry routing ────────────────────────────


class TestRerankRegistry:
    def test_get_rerank_driver_for_model_cohere(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "sk-test")
        driver = get_rerank_driver_for_model("cohere/rerank-v3.5")
        assert isinstance(driver, CohereRerankDriver)
        assert driver.model == "rerank-v3.5"

    def test_get_rerank_driver_for_model_voyage(self, monkeypatch):
        monkeypatch.setenv("VOYAGE_API_KEY", "sk-test")
        driver = get_rerank_driver_for_model("voyage/rerank-2.5")
        assert isinstance(driver, VoyageRerankDriver)
        assert driver.model == "rerank-2.5"

    def test_get_rerank_driver_for_model_jina(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "sk-test")
        driver = get_rerank_driver_for_model("jina/jina-reranker-v2-base-multilingual")
        assert isinstance(driver, JinaRerankDriver)
        assert driver.model == "jina-reranker-v2-base-multilingual"

    def test_get_async_rerank_driver_for_model(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "sk-test")
        driver = get_async_rerank_driver_for_model("cohere/rerank-v3.5")
        assert isinstance(driver, AsyncCohereRerankDriver)

    def test_registries_have_all_three_providers(self):
        assert "cohere" in RERANK_DRIVER_REGISTRY
        assert "voyage" in RERANK_DRIVER_REGISTRY
        assert "jina" in RERANK_DRIVER_REGISTRY
        assert "cohere" in ASYNC_RERANK_DRIVER_REGISTRY
        assert "voyage" in ASYNC_RERANK_DRIVER_REGISTRY
        assert "jina" in ASYNC_RERANK_DRIVER_REGISTRY

    def test_base_classes_are_abstract(self):
        # Verify the abstract bases cannot be instantiated directly.
        with pytest.raises(TypeError):
            RerankDriver()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            AsyncRerankDriver()  # type: ignore[abstract]


# Silence unused-import warning if httpx is required for type narrowing.
_ = httpx
