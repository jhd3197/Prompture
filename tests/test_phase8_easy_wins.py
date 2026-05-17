"""Unit tests for Phase 8: Nomic + Mixedbread embeddings/rerank + GitHub Models profile.

All HTTP calls are mocked — no live network access required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from prompture.drivers.async_mxbai_embedding_driver import AsyncMxbaiEmbeddingDriver
from prompture.drivers.async_mxbai_rerank_driver import AsyncMxbaiRerankDriver
from prompture.drivers.async_nomic_embedding_driver import AsyncNomicEmbeddingDriver
from prompture.drivers.async_openai_compatible_driver import AsyncOpenAICompatibleDriver
from prompture.drivers.embedding_registry import get_embedding_driver_for_model
from prompture.drivers.mxbai_embedding_driver import MxbaiEmbeddingDriver
from prompture.drivers.mxbai_rerank_driver import MxbaiRerankDriver
from prompture.drivers.nomic_embedding_driver import NomicEmbeddingDriver
from prompture.drivers.openai_compatible_driver import (
    OPENAI_COMPATIBLE_PROFILES,
    OpenAICompatibleDriver,
)
from prompture.drivers.rerank_base import RerankResult
from prompture.drivers.rerank_registry import get_rerank_driver_for_model

# ── HTTP mock helpers ────────────────────────────────────────────────────


def _mock_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = payload
    mock.raise_for_status = MagicMock()
    mock.status_code = status
    mock.text = ""
    return mock


def _nomic_embed_payload(n: int = 2, dim: int = 4) -> dict[str, Any]:
    return {
        "embeddings": [[float(i) / 10] * dim for i in range(n)],
        "usage": {"total_tokens": 1500},
    }


def _mxbai_embed_payload(n: int = 2, dim: int = 4) -> dict[str, Any]:
    return {
        "data": [{"embedding": [float(i) / 10] * dim, "index": i, "object": "embedding"} for i in range(n)],
        "model": "mxbai-embed-large-v1",
        "usage": {"prompt_tokens": 800, "total_tokens": 800},
    }


def _mxbai_rerank_payload() -> dict[str, Any]:
    # Intentionally unsorted to verify we sort by score.
    return {
        "data": [
            {"index": 0, "score": 0.3, "input": "alpha"},
            {"index": 1, "score": 0.95, "input": "beta"},
            {"index": 2, "score": 0.6, "input": "gamma"},
        ],
        "usage": {"total_tokens": 200},
    }


# ─────────────────────────── Nomic embeddings ─────────────────────────────


class TestNomicEmbedding:
    def test_embed_returns_vectors_and_usage(self):
        d = NomicEmbeddingDriver(api_key="sk-test", model="nomic-embed-text-v1.5")
        with patch(
            "prompture.drivers.nomic_embedding_driver.requests.post",
            return_value=_mock_response(_nomic_embed_payload(2, 4)),
        ) as mp:
            result = d.embed(["doc one", "doc two"], {})
        assert len(result["embeddings"]) == 2
        assert all(isinstance(v, list) for v in result["embeddings"])
        meta = result["meta"]
        assert meta["model_name"] == "nomic/nomic-embed-text-v1.5"
        assert meta["dimensions"] == 4
        assert meta["total_tokens"] == 1500
        assert meta["cost"] > 0
        sent = mp.call_args.kwargs["json"]
        assert sent["task_type"] == "search_document"
        assert sent["model"] == "nomic-embed-text-v1.5"
        assert sent["texts"] == ["doc one", "doc two"]

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("NOMIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="NOMIC_API_KEY"):
            NomicEmbeddingDriver(api_key=None)

    def test_http_error_surfaces_body(self):
        import requests as _rq

        d = NomicEmbeddingDriver(api_key="sk-test")
        fake_resp = MagicMock()
        fake_resp.text = '{"error": "bad model"}'
        fake_resp.status_code = 400
        err = _rq.exceptions.HTTPError("400")
        err.response = fake_resp
        with patch(
            "prompture.drivers.nomic_embedding_driver.requests.post",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError, match="bad model"):
                d.embed(["x"], {})

    def test_async_embed(self):
        d = AsyncNomicEmbeddingDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_nomic_embed_payload(2, 4))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            result = asyncio.run(d.embed(["a", "b"], {}))
        assert len(result["embeddings"]) == 2
        assert result["meta"]["total_tokens"] == 1500


# ─────────────────────────── Mxbai embeddings ─────────────────────────────


class TestMxbaiEmbedding:
    def test_embed_returns_vectors_and_usage(self):
        d = MxbaiEmbeddingDriver(api_key="sk-test", model="mxbai-embed-large-v1")
        with patch(
            "prompture.drivers.mxbai_embedding_driver.requests.post",
            return_value=_mock_response(_mxbai_embed_payload(2, 4)),
        ) as mp:
            result = d.embed(["doc one", "doc two"], {})
        assert len(result["embeddings"]) == 2
        meta = result["meta"]
        assert meta["model_name"] == "mixedbread/mxbai-embed-large-v1"
        assert meta["dimensions"] == 4
        assert meta["total_tokens"] == 800
        assert meta["cost"] > 0
        sent = mp.call_args.kwargs["json"]
        assert sent["model"] == "mxbai-embed-large-v1"
        assert sent["input"] == ["doc one", "doc two"]
        assert sent["encoding_format"] == "float"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MIXEDBREAD_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MIXEDBREAD_API_KEY"):
            MxbaiEmbeddingDriver(api_key=None)

    def test_async_embed(self):
        d = AsyncMxbaiEmbeddingDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_mxbai_embed_payload(2, 4))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            result = asyncio.run(d.embed(["a", "b"], {}))
        assert len(result["embeddings"]) == 2
        assert result["meta"]["total_tokens"] == 800


# ─────────────────────────── Mxbai rerank ─────────────────────────────────


class TestMxbaiRerank:
    def test_rerank_returns_sorted_results(self):
        d = MxbaiRerankDriver(api_key="sk-test", model="mxbai-rerank-large-v1")
        with patch(
            "prompture.drivers.mxbai_rerank_driver.requests.post",
            return_value=_mock_response(_mxbai_rerank_payload()),
        ) as mp:
            results = d.rerank("query", ["alpha", "beta", "gamma"], top_n=3)
        assert isinstance(results, list)
        assert all(isinstance(r, RerankResult) for r in results)
        assert [r.index for r in results] == [1, 2, 0]  # sorted by score desc
        assert results[0].relevance_score == pytest.approx(0.95)

        # last_usage populated; cost > 0 thanks to mixedbread.json rates.
        assert d.last_usage["model_name"] == "mixedbread/mxbai-rerank-large-v1"
        assert d.last_usage["total_tokens"] == 200
        assert d.last_usage["pricing_unknown"] is False
        assert d.last_usage["cost"] > 0.0

        sent = mp.call_args.kwargs["json"]
        assert sent["model"] == "mxbai-rerank-large-v1"
        assert sent["query"] == "query"
        assert sent["input"] == ["alpha", "beta", "gamma"]
        assert sent["top_k"] == 3
        assert sent["return_input"] is False

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MIXEDBREAD_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MIXEDBREAD_API_KEY"):
            MxbaiRerankDriver(api_key=None)

    def test_async_rerank(self):
        d = AsyncMxbaiRerankDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_mxbai_rerank_payload())

        with patch("httpx.AsyncClient.post", new=_fake_post):
            results = asyncio.run(d.rerank("q", ["alpha", "beta", "gamma"]))
        assert [r.index for r in results] == [1, 2, 0]
        assert d.last_usage["cost"] > 0.0


# ────────────────────── GitHub Models openai_compatible ───────────────────


class TestGithubModelsProfile:
    def test_profile_registered(self):
        assert "github_models" in OPENAI_COMPATIBLE_PROFILES
        prof = OPENAI_COMPATIBLE_PROFILES["github_models"]
        assert prof["endpoint"] == "https://models.github.ai/inference"
        assert prof["env_var"] == "GITHUB_TOKEN"

    def test_driver_resolves_github_models_endpoint(self):
        d = OpenAICompatibleDriver(
            profile="github_models",
            api_key="ghp_test",
            model="openai/gpt-4o-mini",
        )
        assert d.endpoint == "https://models.github.ai/inference"
        assert d.profile == "github_models"
        assert d.api_key == "ghp_test"

    def test_async_driver_resolves_github_models_endpoint(self):
        d = AsyncOpenAICompatibleDriver(
            profile="github_models",
            api_key="ghp_test",
            model="openai/gpt-4o-mini",
        )
        assert d.endpoint == "https://models.github.ai/inference"
        assert d.profile == "github_models"

    def test_env_var_resolution(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_from_env")
        d = OpenAICompatibleDriver(profile="github_models", model="openai/gpt-4o-mini")
        assert d.api_key == "ghp_from_env"


# ─────────────────────────── Registry routing ────────────────────────────


class TestRegistryRouting:
    def test_get_embedding_driver_for_model_nomic(self, monkeypatch):
        monkeypatch.setenv("NOMIC_API_KEY", "sk-test")
        driver = get_embedding_driver_for_model("nomic/nomic-embed-text-v1.5")
        assert isinstance(driver, NomicEmbeddingDriver)
        assert driver.model == "nomic-embed-text-v1.5"

    def test_get_embedding_driver_for_model_mixedbread(self, monkeypatch):
        monkeypatch.setenv("MIXEDBREAD_API_KEY", "sk-test")
        driver = get_embedding_driver_for_model("mixedbread/mxbai-embed-large-v1")
        assert isinstance(driver, MxbaiEmbeddingDriver)
        assert driver.model == "mxbai-embed-large-v1"

    def test_get_embedding_driver_for_model_mxbai_alias(self, monkeypatch):
        monkeypatch.setenv("MIXEDBREAD_API_KEY", "sk-test")
        driver = get_embedding_driver_for_model("mxbai/mxbai-embed-large-v1")
        assert isinstance(driver, MxbaiEmbeddingDriver)

    def test_get_rerank_driver_for_model_mixedbread(self, monkeypatch):
        monkeypatch.setenv("MIXEDBREAD_API_KEY", "sk-test")
        driver = get_rerank_driver_for_model("mixedbread/mxbai-rerank-large-v1")
        assert isinstance(driver, MxbaiRerankDriver)
        assert driver.model == "mxbai-rerank-large-v1"

    def test_get_rerank_driver_for_model_mxbai_alias(self, monkeypatch):
        monkeypatch.setenv("MIXEDBREAD_API_KEY", "sk-test")
        driver = get_rerank_driver_for_model("mxbai/mxbai-rerank-large-v1")
        assert isinstance(driver, MxbaiRerankDriver)
