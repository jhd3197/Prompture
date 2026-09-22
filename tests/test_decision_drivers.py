"""Unit tests for the decision ("System One") modality: TypeSafe, Kev, Laya.

All HTTP calls are mocked and the Laya package is stubbed — no live network
access and no model weights required.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from prompture.drivers.async_kev_decision_driver import AsyncKevDecisionDriver
from prompture.drivers.async_laya_decision_driver import AsyncLayaDecisionDriver
from prompture.drivers.async_typesafe_decision_driver import AsyncTypeSafeDecisionDriver
from prompture.drivers.decision_base import (
    AsyncDecisionDriver,
    Choice,
    ChoiceAnswer,
    DecisionDriver,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    calculate_decision_cost,
    normalize_questions,
    parse_answer,
)
from prompture.drivers.decision_registry import (
    ASYNC_DECISION_DRIVER_REGISTRY,
    DECISION_DRIVER_REGISTRY,
    get_async_decision_driver_for_model,
    get_decision_driver_for_model,
)
from prompture.drivers.kev_decision_driver import KevDecisionDriver
from prompture.drivers.laya_decision_driver import LayaDecisionDriver
from prompture.drivers.systemone_compatible_driver import build_systemone_url
from prompture.drivers.typesafe_decision_driver import TypeSafeDecisionDriver
from prompture.infra.discovery import get_available_decision_models

# ── Helpers ────────────────────────────────────────────────────────────────


def _mock_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a fake requests.Response-like object."""
    mock = MagicMock()
    mock.json.return_value = payload
    mock.raise_for_status = MagicMock()
    mock.status_code = status
    mock.text = ""
    return mock


def _payload(model: str = "jev-1.13.0", input_tokens: int = 1_000_000) -> dict[str, Any]:
    """A response carrying one of each answer type."""
    return {
        "model": model,
        "answers": {
            "is_urgent": {"type": "noul", "noul": 0.95},
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                "confidence": 0.81,
            },
            "frustration": {
                "type": "score",
                "score": 1.05,
                "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                "confidence": 0.92,
            },
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": 20},
    }


QUESTIONS = {
    "is_urgent": Noul("Does this convey urgency?"),
    "department": Choice(
        "Which team should handle this?",
        criteria={"billing": "Payments, invoicing, refunds", "technical": None, "sales": None},
    ),
    "frustration": Score("How frustrated is the customer?", criteria=["Calm", "Frustrated", "Very angry"]),
}

STATE = "Help! My payouts have been failing for 3 days."


# ── Question serialization ─────────────────────────────────────────────────


class TestQuestions:
    def test_noul_payload(self):
        q = Noul("Urgent?", criteria={"true": "yes it is", "false": "no"})
        assert q.to_payload() == {
            "type": "noul",
            "instructions": "Urgent?",
            "criteria": {"true": "yes it is", "false": "no"},
        }

    def test_noul_omits_absent_criteria(self):
        assert "criteria" not in Noul("Urgent?").to_payload()

    def test_choice_payload_keeps_none_rubrics(self):
        q = Choice("Which?", criteria={"a": "the a one", "b": None})
        assert q.to_payload()["criteria"] == {"a": "the a one", "b": None}

    def test_score_payload_is_ordered_list(self):
        q = Score("How bad?", criteria=["fine", "bad", "awful"])
        assert q.to_payload()["criteria"] == ["fine", "bad", "awful"]

    def test_empty_criteria_rejected(self):
        with pytest.raises(ValueError, match="at least one option"):
            Choice("Which?", criteria={})
        with pytest.raises(ValueError, match="at least one level"):
            Score("How bad?", criteria=[])

    def test_normalize_accepts_raw_dicts(self):
        out = normalize_questions({"x": {"type": "noul", "instructions": "Urgent?"}})
        assert out == {"x": {"type": "noul", "instructions": "Urgent?"}}

    def test_normalize_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="unknown type"):
            normalize_questions({"x": {"type": "vibes", "instructions": "?"}})

    def test_normalize_rejects_empty_map(self):
        with pytest.raises(ValueError, match="At least one question"):
            normalize_questions({})

    def test_normalize_rejects_non_mapping(self):
        with pytest.raises(ValueError, match="must be a Question instance"):
            normalize_questions({"x": "just a string"})


# ── Answer parsing ─────────────────────────────────────────────────────────


class TestAnswers:
    def test_noul_answer(self):
        a = parse_answer("q", {"type": "noul", "noul": 0.95})
        assert isinstance(a, NoulAnswer)
        assert a.noul == 0.95
        assert a.value == 0.95
        assert a.type == "noul"

    def test_noul_confidence_is_distance_from_coin_flip(self):
        assert parse_answer("q", {"type": "noul", "noul": 0.5}).confidence == 0.0
        assert parse_answer("q", {"type": "noul", "noul": 1.0}).confidence == 1.0
        assert parse_answer("q", {"type": "noul", "noul": 0.0}).confidence == 1.0

    def test_choice_answer(self):
        a = parse_answer("q", _payload()["answers"]["department"])
        assert isinstance(a, ChoiceAnswer)
        assert a.choice == "billing"
        assert a.confidence == 0.81
        assert a.ranked()[0] == ("billing", 0.88)
        assert a.value == "billing"

    def test_choice_falls_back_to_argmax_when_choice_missing(self):
        a = parse_answer("q", {"type": "choice", "probabilities": {"a": 0.2, "b": 0.8}})
        assert a.choice == "b"

    def test_score_answer(self):
        a = parse_answer("q", _payload()["answers"]["frustration"])
        assert isinstance(a, ScoreAnswer)
        assert a.score == 1.05
        assert a.nearest_level == "Frustrated"
        assert a.value == 1.05

    def test_unknown_answer_type_rejected(self):
        with pytest.raises(ValueError, match="unknown type"):
            parse_answer("q", {"type": "vibes"})


# ── Pricing ────────────────────────────────────────────────────────────────


class TestPricing:
    def test_typesafe_bills_input_only(self):
        cost, unknown = calculate_decision_cost(
            "typesafe", "jev-latest", input_tokens=1_000_000, output_tokens=1_000_000
        )
        assert unknown is False
        assert cost == pytest.approx(0.042)

    def test_self_hosted_is_free_but_known(self):
        for provider, model in (("kev", "kev-4b"), ("laya", "router")):
            cost, unknown = calculate_decision_cost(provider, model, input_tokens=5_000_000)
            assert cost == 0.0
            assert unknown is False, f"{provider}/{model} should have a known (zero) price"

    def test_unknown_model_flags_pricing_unknown(self):
        cost, unknown = calculate_decision_cost("typesafe", "jev-99", input_tokens=1000)
        assert cost == 0.0
        assert unknown is True


# ── URL handling ───────────────────────────────────────────────────────────


class TestUrlBuilding:
    @pytest.mark.parametrize(
        "given",
        [
            "https://api.typesafe.ai",
            "https://api.typesafe.ai/",
            "https://api.typesafe.ai/v1/systemone",
        ],
    )
    def test_normalizes_to_full_endpoint(self, given):
        assert build_systemone_url(given) == "https://api.typesafe.ai/v1/systemone"


# ── Sync HTTP drivers ──────────────────────────────────────────────────────


class TestTypeSafeDecisionDriver:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            TypeSafeDecisionDriver(api_key=None)

    def test_defaults(self):
        d = TypeSafeDecisionDriver(api_key="k")
        assert d.url == "https://api.typesafe.ai/v1/systemone"
        assert d.model == "jev-latest"
        assert d.headers["Authorization"] == "Bearer k"
        assert isinstance(d, DecisionDriver)

    def test_decide_parses_every_primitive(self):
        d = TypeSafeDecisionDriver(api_key="k")
        with patch("requests.post", return_value=_mock_response(_payload())) as post:
            result = d.decide(STATE, QUESTIONS)

        sent = post.call_args.kwargs["json"]
        assert sent["state"] == STATE
        assert sent["model"] == "jev-latest"
        assert set(sent["questions"]) == {"is_urgent", "department", "frustration"}
        assert sent["questions"]["is_urgent"]["type"] == "noul"

        assert result.model == "jev-1.13.0"
        assert result["is_urgent"].noul == 0.95
        assert result["department"].choice == "billing"
        assert result["frustration"].score == 1.05
        assert result.values() == {"is_urgent": 0.95, "department": "billing", "frustration": 1.05}
        assert "department" in result
        assert sorted(result) == ["department", "frustration", "is_urgent"]

    def test_usage_and_cost_recorded(self):
        d = TypeSafeDecisionDriver(api_key="k")
        with patch("requests.post", return_value=_mock_response(_payload())):
            result = d.decide(STATE, QUESTIONS)

        assert d.last_usage["model_name"] == "typesafe/jev-latest"
        assert d.last_usage["questions"] == 3
        assert d.last_usage["input_tokens"] == 1_000_000
        assert d.last_usage["total_tokens"] == 1_000_020
        assert d.last_usage["cost"] == pytest.approx(0.042)
        assert d.last_usage["pricing_unknown"] is False
        assert result.usage["cost"] == d.last_usage["cost"]

    def test_unknown_alias_falls_back_to_reported_model_for_pricing(self):
        d = TypeSafeDecisionDriver(api_key="k", model="jev-experimental")
        with patch("requests.post", return_value=_mock_response(_payload())):
            d.decide(STATE, QUESTIONS)
        # "jev-experimental" has no rate entry; the server said jev-1.13.0, which does.
        assert d.last_usage["pricing_unknown"] is False
        assert d.last_usage["cost"] == pytest.approx(0.042)

    def test_model_override_per_call(self):
        d = TypeSafeDecisionDriver(api_key="k")
        with patch("requests.post", return_value=_mock_response(_payload())) as post:
            d.decide(STATE, QUESTIONS, model="jev-preview")
        assert post.call_args.kwargs["json"]["model"] == "jev-preview"

    def test_ask_single_question(self):
        d = TypeSafeDecisionDriver(api_key="k")
        single = {"model": "jev-1.13.0", "answers": {"answer": {"type": "noul", "noul": 0.7}}, "usage": {}}
        with patch("requests.post", return_value=_mock_response(single)):
            answer = d.ask(STATE, Noul("Urgent?"))
        assert answer.noul == 0.7

    def test_http_error_is_wrapped(self):
        d = TypeSafeDecisionDriver(api_key="k")
        err_response = MagicMock()
        err_response.text = '{"error": "bad request"}'
        exc = __import__("requests").exceptions.HTTPError("400 Client Error", response=err_response)
        bad = MagicMock()
        bad.raise_for_status.side_effect = exc
        with patch("requests.post", return_value=bad), pytest.raises(RuntimeError) as ei:
            d.decide(STATE, QUESTIONS)
        assert "typesafe decision API request failed" in str(ei.value)
        assert "bad request" in str(ei.value)


class TestKevDecisionDriver:
    def test_no_api_key_required(self, monkeypatch):
        monkeypatch.delenv("KEV_API_KEY", raising=False)
        monkeypatch.delenv("KEV_BASE_URL", raising=False)
        d = KevDecisionDriver()
        assert d.url == "http://127.0.0.1:8009/v1/systemone"
        assert "Authorization" not in d.headers

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("KEV_BASE_URL", "http://gpu-box:9123")
        assert KevDecisionDriver().url == "http://gpu-box:9123/v1/systemone"

    def test_explicit_base_url_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("KEV_BASE_URL", "http://gpu-box:9123")
        d = KevDecisionDriver(base_url="http://127.0.0.1:8009")
        assert d.url == "http://127.0.0.1:8009/v1/systemone"

    def test_decide_is_free_but_priced(self):
        d = KevDecisionDriver(model="kev-4b")
        with patch("requests.post", return_value=_mock_response(_payload(model="kev-4b"))):
            d.decide(STATE, QUESTIONS)
        assert d.last_usage["model_name"] == "kev/kev-4b"
        assert d.last_usage["cost"] == 0.0
        assert d.last_usage["pricing_unknown"] is False


# ── Async HTTP drivers ─────────────────────────────────────────────────────


class TestAsyncDecisionDrivers:
    def test_async_typesafe_decide(self):
        d = AsyncTypeSafeDecisionDriver(api_key="k")
        assert isinstance(d, AsyncDecisionDriver)

        mock_response = MagicMock()
        mock_response.json.return_value = _payload()
        mock_response.raise_for_status = MagicMock()

        async def run():
            client = MagicMock()
            client.post = MagicMock(return_value=_awaitable(mock_response))
            client.__aenter__ = MagicMock(return_value=_awaitable(client))
            client.__aexit__ = MagicMock(return_value=_awaitable(None))
            with patch.object(httpx, "AsyncClient", return_value=client):
                return await d.decide(STATE, QUESTIONS)

        result = asyncio.run(run())
        assert result["department"].choice == "billing"
        assert d.last_usage["cost"] == pytest.approx(0.042)

    def test_async_kev_needs_no_key(self, monkeypatch):
        monkeypatch.delenv("KEV_API_KEY", raising=False)
        monkeypatch.delenv("KEV_BASE_URL", raising=False)
        assert AsyncKevDecisionDriver().url == "http://127.0.0.1:8009/v1/systemone"


def _awaitable(value):
    """Wrap *value* in a coroutine, so MagicMock can stand in for an async call."""

    async def _coro():
        return value

    return _coro()


# ── Laya (in-process) ──────────────────────────────────────────────────────


class _FakeRouter:
    """Stand-in for ``laya.Router`` that records what it was asked."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple[Any, dict]] = []

    def predict(self, state, questions, **options):
        self.calls.append((state, questions))
        body = _payload(model="laya")
        body["routing"] = {
            "model": "multilingual",
            "repo": "convaiinnovations/laya/multilingual",
            "reason": "non-Latin script",
        }
        body.pop("usage")
        return body

    def unload(self):
        self.kwargs["unloaded"] = True


@pytest.fixture
def fake_laya(monkeypatch):
    """Install a stub ``laya`` module so no weights are downloaded."""
    routers: list[_FakeRouter] = []

    def _router(**kwargs):
        r = _FakeRouter(**kwargs)
        routers.append(r)
        return r

    module = SimpleNamespace(Router=_router, load=lambda repo, **kw: _FakeRouter(repo=repo, **kw))
    monkeypatch.setitem(sys.modules, "laya", module)
    return routers


class TestLayaDecisionDriver:
    def test_defaults_to_router_with_preload(self, fake_laya):
        d = LayaDecisionDriver()
        assert d.model == "router"
        d.decide(STATE, QUESTIONS)
        assert fake_laya[0].kwargs["preload"] is True

    def test_reports_routed_checkpoint_not_alias(self, fake_laya):
        d = LayaDecisionDriver()
        result = d.decide(STATE, QUESTIONS)
        assert result.model == "multilingual"
        assert d.last_usage["routing"]["reason"] == "non-Latin script"

    def test_answers_parse_identically_to_http_providers(self, fake_laya):
        result = LayaDecisionDriver().decide(STATE, QUESTIONS)
        assert result["is_urgent"].noul == 0.95
        assert result["department"].choice == "billing"
        assert result["frustration"].nearest_level == "Frustrated"

    def test_usage_is_free_with_no_tokens(self, fake_laya):
        d = LayaDecisionDriver()
        d.decide(STATE, QUESTIONS)
        assert d.last_usage["model_name"] == "laya/router"
        assert d.last_usage["cost"] == 0.0
        assert d.last_usage["pricing_unknown"] is False
        assert d.last_usage["total_tokens"] == 0

    def test_named_checkpoint_uses_subfolder(self, fake_laya, monkeypatch):
        loaded: dict[str, Any] = {}

        def _load(repo, **kw):
            loaded["repo"] = repo
            loaded.update(kw)
            return _FakeRouter()

        monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(Router=_FakeRouter, load=_load))
        LayaDecisionDriver(model="multilingual").decide(STATE, QUESTIONS)
        assert loaded == {"repo": "convaiinnovations/laya", "subfolder": "multilingual"}

    def test_missing_package_raises_actionable_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fail(name, *args, **kwargs):
            if name == "laya":
                raise ImportError("No module named 'laya'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setitem(sys.modules, "laya", None)
        monkeypatch.delitem(sys.modules, "laya")
        monkeypatch.setattr(builtins, "__import__", _fail)
        with pytest.raises(ImportError, match=r"prompture\[laya\]"):
            LayaDecisionDriver().decide(STATE, QUESTIONS)

    def test_async_wrapper_delegates(self, fake_laya):
        d = AsyncLayaDecisionDriver()
        result = asyncio.run(d.decide(STATE, QUESTIONS))
        assert result["department"].choice == "billing"
        assert d.last_usage["model_name"] == "laya/router"


# ── Registry wiring ────────────────────────────────────────────────────────


class TestDecisionRegistry:
    def test_all_three_providers_registered(self):
        for provider in ("typesafe", "kev", "laya"):
            assert provider in DECISION_DRIVER_REGISTRY
            assert provider in ASYNC_DECISION_DRIVER_REGISTRY

    def test_factory_builds_right_class_and_model(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        d = get_decision_driver_for_model("typesafe/jev-preview")
        assert isinstance(d, TypeSafeDecisionDriver)
        assert d.model == "jev-preview"

        k = get_decision_driver_for_model("kev/kev-9b")
        assert isinstance(k, KevDecisionDriver)
        assert k.model == "kev-9b"

    def test_async_factory(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        d = get_async_decision_driver_for_model("typesafe/jev-latest")
        assert isinstance(d, AsyncTypeSafeDecisionDriver)
        assert d.supports_async is True

    def test_bare_provider_uses_default_model(self, monkeypatch):
        monkeypatch.delenv("KEV_BASE_URL", raising=False)
        assert get_decision_driver_for_model("kev").model == "kev-latest"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            get_decision_driver_for_model("nope/whatever")


class TestDecisionDiscovery:
    def test_lists_configured_providers_only(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        monkeypatch.delenv("KEV_BASE_URL", raising=False)
        models = get_available_decision_models()
        assert "typesafe/jev-latest" in models
        assert not any(m.startswith("kev/") for m in models)

    def test_kev_listed_once_endpoint_configured(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.setenv("KEV_BASE_URL", "http://127.0.0.1:8009")
        models = get_available_decision_models()
        assert "kev/kev-4b" in models
        assert not any(m.startswith("typesafe/") for m in models)

    def test_nothing_configured_lists_no_hosted_models(self, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.delenv("KEV_BASE_URL", raising=False)
        models = get_available_decision_models()
        assert not any(m.startswith(("typesafe/", "kev/")) for m in models)
