"""Unit tests for Phase 9 moderation drivers: OpenAI and Mistral.

All HTTP calls are mocked — no live network access required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from prompture.drivers.async_mistral_moderation_driver import AsyncMistralModerationDriver
from prompture.drivers.async_openai_moderation_driver import AsyncOpenAIModerationDriver
from prompture.drivers.mistral_moderation_driver import MistralModerationDriver
from prompture.drivers.moderation_base import (
    AsyncModerationDriver,
    ModerationDriver,
    ModerationResult,
)
from prompture.drivers.moderation_registry import (
    ASYNC_MODERATION_DRIVER_REGISTRY,
    MODERATION_DRIVER_REGISTRY,
    get_async_moderation_driver_for_model,
    get_moderation_driver_for_model,
)
from prompture.drivers.openai_moderation_driver import OpenAIModerationDriver

# ── Helpers ────────────────────────────────────────────────────────────────


def _mock_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a fake requests.Response-like object."""
    mock = MagicMock()
    mock.json.return_value = payload
    mock.raise_for_status = MagicMock()
    mock.status_code = status
    mock.text = ""
    return mock


def _openai_payload(n_results: int = 1, flagged: bool = True) -> dict[str, Any]:
    categories = {
        "sexual": False,
        "hate": False,
        "harassment": flagged,
        "violence": False,
        "self-harm": False,
    }
    category_scores = {
        "sexual": 0.001,
        "hate": 0.0002,
        "harassment": 0.87 if flagged else 0.01,
        "violence": 0.001,
        "self-harm": 0.0001,
    }
    return {
        "id": "modr-test",
        "model": "omni-moderation-latest",
        "results": [
            {
                "flagged": flagged,
                "categories": dict(categories),
                "category_scores": dict(category_scores),
            }
            for _ in range(n_results)
        ],
    }


def _mistral_payload(n_results: int = 1, flagged: bool = True) -> dict[str, Any]:
    """Mistral payload — note: no top-level ``flagged`` field per item."""
    categories = {
        "sexual": False,
        "hate_and_discrimination": False,
        "violence_and_threats": flagged,
        "dangerous_and_criminal_content": False,
        "selfharm": False,
        "health": False,
        "financial": False,
        "law": False,
        "pii": False,
    }
    category_scores = {
        "sexual": 0.0001,
        "hate_and_discrimination": 0.0001,
        "violence_and_threats": 0.92 if flagged else 0.01,
        "dangerous_and_criminal_content": 0.001,
        "selfharm": 0.0001,
        "health": 0.0001,
        "financial": 0.0001,
        "law": 0.0001,
        "pii": 0.0001,
    }
    return {
        "id": "mod-test",
        "model": "mistral-moderation-latest",
        "results": [
            {
                "categories": dict(categories),
                "category_scores": dict(category_scores),
            }
            for _ in range(n_results)
        ],
    }


# ────────────────────────────── OpenAI ────────────────────────────────────


class TestOpenAIModeration:
    def test_moderate_string_returns_single_result(self):
        d = OpenAIModerationDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.openai_moderation_driver.requests.post",
            return_value=_mock_response(_openai_payload(n_results=1, flagged=True)),
        ) as mp:
            result = d.moderate("I will hurt someone")
        assert isinstance(result, ModerationResult)
        assert result.flagged is True
        assert result.categories["harassment"] is True
        assert result.category_scores["harassment"] > 0.5
        # Posted to the correct URL with right model
        called_url = mp.call_args.args[0]
        assert called_url == OpenAIModerationDriver.BASE_URL
        body = mp.call_args.kwargs["json"]
        assert body["model"] == "omni-moderation-latest"
        assert body["input"] == "I will hurt someone"

    def test_moderate_list_returns_list_of_results(self):
        d = OpenAIModerationDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.openai_moderation_driver.requests.post",
            return_value=_mock_response(_openai_payload(n_results=2, flagged=False)),
        ):
            results = d.moderate(["benign text 1", "benign text 2"])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ModerationResult) for r in results)
        assert all(r.flagged is False for r in results)

    def test_last_usage_populated_and_cost_zero(self):
        d = OpenAIModerationDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.openai_moderation_driver.requests.post",
            return_value=_mock_response(_openai_payload()),
        ):
            d.moderate("hello")
        assert d.last_usage["model_name"] == "openai/omni-moderation-latest"
        assert d.last_usage["input_count"] == 1
        # OpenAI moderation is free — cost=0, pricing_unknown=False (we know it's free).
        assert d.last_usage["cost"] == 0.0
        assert d.last_usage["pricing_unknown"] is False

    def test_http_error_raises_runtime_error(self):
        import requests as _rq

        d = OpenAIModerationDriver(api_key="sk-test")
        fake_resp = MagicMock()
        fake_resp.text = '{"error": "bad request"}'
        fake_resp.status_code = 400
        err = _rq.exceptions.HTTPError("400")
        err.response = fake_resp
        with patch(
            "prompture.drivers.openai_moderation_driver.requests.post",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError, match="bad request"):
                d.moderate("anything")

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIModerationDriver(api_key=None)

    def test_async_moderate_string(self):
        d = AsyncOpenAIModerationDriver(api_key="sk-test", model="omni-moderation-latest")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_openai_payload(n_results=1, flagged=True))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            result = asyncio.run(d.moderate("violent text"))
        assert isinstance(result, ModerationResult)
        assert result.flagged is True
        assert d.last_usage["model_name"] == "openai/omni-moderation-latest"

    def test_async_moderate_list(self):
        d = AsyncOpenAIModerationDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_openai_payload(n_results=3, flagged=False))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            results = asyncio.run(d.moderate(["a", "b", "c"]))
        assert isinstance(results, list)
        assert len(results) == 3


# ───────────────────────────── Mistral ────────────────────────────────────


class TestMistralModeration:
    def test_moderate_string_computes_flagged_from_categories(self):
        d = MistralModerationDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.mistral_moderation_driver.requests.post",
            return_value=_mock_response(_mistral_payload(n_results=1, flagged=True)),
        ) as mp:
            result = d.moderate("violent threat")
        # Mistral does NOT return flagged — driver computes any(categories.values())
        assert isinstance(result, ModerationResult)
        assert result.flagged is True
        assert result.categories["violence_and_threats"] is True
        # Other categories are False
        assert result.categories["sexual"] is False
        called_url = mp.call_args.args[0]
        assert called_url == MistralModerationDriver.BASE_URL

    def test_moderate_string_all_categories_false_flagged_false(self):
        d = MistralModerationDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.mistral_moderation_driver.requests.post",
            return_value=_mock_response(_mistral_payload(n_results=1, flagged=False)),
        ):
            result = d.moderate("benign text")
        assert result.flagged is False

    def test_moderate_list_returns_list(self):
        d = MistralModerationDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.mistral_moderation_driver.requests.post",
            return_value=_mock_response(_mistral_payload(n_results=2, flagged=False)),
        ):
            results = d.moderate(["text a", "text b"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_last_usage_cost_computed_from_rate_json(self):
        d = MistralModerationDriver(api_key="sk-test")
        with patch(
            "prompture.drivers.mistral_moderation_driver.requests.post",
            return_value=_mock_response(_mistral_payload()),
        ):
            d.moderate("hello")
        assert d.last_usage["model_name"] == "mistral/mistral-moderation-latest"
        # Mistral pricing is known (in rate JSON) — pricing_unknown is False.
        assert d.last_usage["pricing_unknown"] is False

    def test_http_error_raises_runtime_error(self):
        import requests as _rq

        d = MistralModerationDriver(api_key="sk-test")
        fake_resp = MagicMock()
        fake_resp.text = '{"error": "unauthorized"}'
        fake_resp.status_code = 401
        err = _rq.exceptions.HTTPError("401")
        err.response = fake_resp
        with patch(
            "prompture.drivers.mistral_moderation_driver.requests.post",
            side_effect=err,
        ):
            with pytest.raises(RuntimeError, match="unauthorized"):
                d.moderate("anything")

    def test_async_moderate(self):
        d = AsyncMistralModerationDriver(api_key="sk-test")

        async def _fake_post(self, url, headers=None, json=None):
            return _mock_response(_mistral_payload(n_results=1, flagged=True))

        with patch("httpx.AsyncClient.post", new=_fake_post):
            result = asyncio.run(d.moderate("violent text"))
        assert isinstance(result, ModerationResult)
        assert result.flagged is True


# ──────────────────────── Registry routing ────────────────────────────────


class TestModerationRegistry:
    def test_get_moderation_driver_for_model_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        driver = get_moderation_driver_for_model("openai/omni-moderation-latest")
        assert isinstance(driver, OpenAIModerationDriver)
        assert driver.model == "omni-moderation-latest"

    def test_get_moderation_driver_for_model_mistral(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
        driver = get_moderation_driver_for_model("mistral/mistral-moderation-latest")
        assert isinstance(driver, MistralModerationDriver)
        assert driver.model == "mistral-moderation-latest"

    def test_get_async_moderation_driver_for_model_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        driver = get_async_moderation_driver_for_model("openai/omni-moderation-latest")
        assert isinstance(driver, AsyncOpenAIModerationDriver)

    def test_get_async_moderation_driver_for_model_mistral(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
        driver = get_async_moderation_driver_for_model("mistral/mistral-moderation-latest")
        assert isinstance(driver, AsyncMistralModerationDriver)

    def test_registries_have_openai_and_mistral(self):
        assert "openai" in MODERATION_DRIVER_REGISTRY
        assert "mistral" in MODERATION_DRIVER_REGISTRY
        assert "openai" in ASYNC_MODERATION_DRIVER_REGISTRY
        assert "mistral" in ASYNC_MODERATION_DRIVER_REGISTRY

    def test_base_classes_are_abstract(self):
        with pytest.raises(TypeError):
            ModerationDriver()  # type: ignore[abstract]
        with pytest.raises(TypeError):
            AsyncModerationDriver()  # type: ignore[abstract]


# Silence unused-import warning if httpx is required for type narrowing.
_ = httpx
