"""Routing strategies, stickiness, combos, aliases and auto/ models."""

from __future__ import annotations

import json
import random
from typing import Any

import pytest
from test_resilience import Factory, FakeDriver

from prompture.resilience import BreakerRegistry, ResilientDriver, clear_key_pools, register_key_pool
from prompture.resilience.strategies import Orderer, RouteStats
from prompture.resilience.virtual import (
    AUTO_MODES,
    _interleave_providers,
    _is_chat_model,
    clear_virtual_models,
    get_combo,
    list_virtual_models,
    load_combos,
    register_combo,
    register_model_alias,
    resolve_model_alias,
    resolve_virtual_model,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_virtual_models()
    clear_key_pools()
    yield
    clear_virtual_models()
    clear_key_pools()


def _ident(x: str) -> str:
    return x


# ---------------------------------------------------------------------------
# Orderer
# ---------------------------------------------------------------------------


class TestOrderer:
    items = ["a/1", "b/2", "c/3"]

    def test_priority_keeps_order(self):
        assert Orderer("priority").order(self.items, _ident) == self.items

    def test_round_robin_rotates(self):
        o = Orderer("round_robin")
        firsts = [o.order(self.items, _ident)[0] for _ in range(3)]
        assert firsts == ["a/1", "b/2", "c/3"]

    def test_weighted_respects_weights(self):
        o = Orderer("weighted", weights=[0.0, 0.0, 1.0], rng=random.Random(1))
        assert o.order(self.items, _ident)[0] == "c/3"
        with pytest.raises(ValueError):
            Orderer("weighted", weights=[1.0]).order(self.items, _ident)

    def test_latency_prefers_fast_and_explores_unmeasured(self):
        stats = RouteStats()
        stats.record_success("a/1", 900)
        stats.record_success("b/2", 100)
        order = Orderer("latency", stats=stats).order(self.items, _ident)
        assert order == ["c/3", "b/2", "a/1"]  # c unmeasured → explored first

    def test_p2c_puts_faster_of_two_first(self):
        stats = RouteStats()
        stats.record_success("a/1", 500)
        stats.record_success("b/2", 50)
        stats.record_success("c/3", 900)
        for seed in range(10):
            first = Orderer("p2c", stats=stats, rng=random.Random(seed)).order(self.items, _ident)[0]
            assert first != "c/3"  # the slowest never wins a pairwise comparison

    def test_last_good(self):
        clock = iter([1.0, 2.0])
        stats = RouteStats(clock=lambda: next(clock))
        stats.record_success("c/3")
        stats.record_success("b/2")
        assert Orderer("last_good", stats=stats).order(self.items, _ident)[0] == "b/2"
        assert Orderer("last_good", stats=RouteStats()).order(self.items, _ident) == self.items

    def test_cheapest_uses_pricing_and_puts_unknown_last(self):
        prices = {"a/1": 5.0, "b/2": None, "c/3": 0.5}
        o = Orderer("cheapest", price=prices.get)
        assert o.order(self.items, _ident) == ["c/3", "a/1", "b/2"]

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            Orderer("vibes")

    def test_ewma(self):
        stats = RouteStats(alpha=0.5)
        stats.record_success("x", 100)
        stats.record_success("x", 300)
        assert stats.get("x").ewma_latency_ms == pytest.approx(200)


# ---------------------------------------------------------------------------
# ResilientDriver integration
# ---------------------------------------------------------------------------


def test_driver_strategy_and_route_summary():
    stats = RouteStats()
    stats.record_success("b/two", 10)
    stats.record_success("a/one", 999)
    drv = ResilientDriver(
        ["a/one", "b/two"],
        strategy="latency",
        stats=stats,
        breakers=BreakerRegistry(),
        factory=Factory({}),
        sleep=lambda s: None,
    )
    resp = drv.generate("hi", {})
    assert resp["meta"]["route"]["served_by"] == "b/two"
    assert resp["meta"]["route"]["strategy"] == "latency"


def test_success_updates_stats():
    stats = RouteStats()
    drv = ResilientDriver(["a/one"], stats=stats, breakers=BreakerRegistry(), factory=Factory({}), sleep=lambda s: None)
    drv.generate("hi", {})
    assert stats.get("a/one").successes == 1
    assert stats.get("a/one").ewma_latency_ms is not None


def test_sticky_pins_conversation_to_one_key():
    register_key_pool("a", ["k1", "k2", "k3"])
    drv = ResilientDriver(["a/one"], sticky=True, breakers=BreakerRegistry(), factory=Factory({}), sleep=lambda s: None)
    convo = [{"role": "system", "content": "sys"}, {"role": "user", "content": "first question"}]
    keys = set()
    for turn in range(4):
        msgs = [*convo, {"role": "assistant", "content": "x"}, {"role": "user", "content": f"follow-up {turn}"}]
        keys.add(drv.generate_messages(msgs, {})["meta"]["route"]["key_id"])
    assert len(keys) == 1

    other = [{"role": "user", "content": f"unrelated {i}"} for i in range(12)]
    spread = {drv.generate_messages([m], {})["meta"]["route"]["key_id"] for m in other}
    assert len(spread) > 1


def test_non_sticky_rotates_keys():
    register_key_pool("a", ["k1", "k2"])
    drv = ResilientDriver(["a/one"], breakers=BreakerRegistry(), factory=Factory({}), sleep=lambda s: None)
    msgs = [{"role": "user", "content": "same"}]
    keys = {drv.generate_messages(msgs, {})["meta"]["route"]["key_id"] for _ in range(2)}
    assert len(keys) == 2


# ---------------------------------------------------------------------------
# Combos, aliases, auto
# ---------------------------------------------------------------------------


class TestVirtualModels:
    def test_combo_resolves_to_resilient_driver(self):
        register_combo("chat", ["openai/gpt-4o", "ollama/llama3.1:8b"], strategy="round_robin", sticky=True)
        name, drv = resolve_virtual_model("combo/chat")
        assert name == "combo/chat"
        assert isinstance(drv, ResilientDriver)
        assert drv.model == "combo/chat"
        assert [t.model for t in drv.targets] == ["openai/gpt-4o", "ollama/llama3.1:8b"]
        assert drv._plan.strategy == "round_robin"
        assert drv._sticky is True

    def test_get_driver_for_model_understands_combos(self):
        from prompture.drivers import get_driver_for_model
        from prompture.drivers.async_registry import get_async_driver_for_model
        from prompture.resilience import AsyncResilientDriver

        register_combo("c", ["ollama/llama3.1:8b"])
        assert isinstance(get_driver_for_model("combo/c"), ResilientDriver)
        assert isinstance(get_async_driver_for_model("combo/c"), AsyncResilientDriver)

    def test_unknown_combo(self):
        with pytest.raises(ValueError, match="Unknown combo 'combo/nope'"):
            resolve_virtual_model("combo/nope")

    def test_alias_chain_and_cycle(self):
        register_model_alias("fast", "smart")
        register_model_alias("smart", "ollama/llama3.1:8b")
        assert resolve_model_alias("fast") == "ollama/llama3.1:8b"
        assert resolve_virtual_model("fast") == ("ollama/llama3.1:8b", None)
        register_model_alias("ollama/llama3.1:8b", "fast")
        with pytest.raises(ValueError, match="cycle"):
            resolve_model_alias("fast")

    def test_alias_to_combo(self):
        register_combo("chat", ["ollama/llama3.1:8b"])
        register_model_alias("default", "combo/chat")
        _, drv = resolve_virtual_model("default")
        assert isinstance(drv, ResilientDriver)

    def test_alias_inside_targets_expands_key_pool_of_real_provider(self):
        register_key_pool("openai", ["k1", "k2"])
        register_model_alias("gpt", "openai/gpt-4o")
        drv = ResilientDriver(["gpt"], breakers=BreakerRegistry(), factory=Factory({}), sleep=lambda s: None)
        assert [t.model for t in drv.targets] == ["openai/gpt-4o", "openai/gpt-4o"]

    def test_validation(self):
        with pytest.raises(ValueError, match="needs at least one target"):
            register_combo("empty", [])
        with pytest.raises(ValueError, match="unknown strategy"):
            register_combo("x", ["a/b"], strategy="nope")
        with pytest.raises(ValueError, match="weights"):
            register_combo("x", ["a/b", "c/d"], strategy="weighted", weights=[1.0])

    def test_load_combos_from_file(self, tmp_path):
        path = tmp_path / "combos.json"
        path.write_text(
            json.dumps(
                {
                    "combos": {
                        "chat": {"targets": ["a/1", "b/2"], "strategy": "latency", "policy": {"max_attempts": 3}},
                        "cheap": ["c/3"],
                    },
                    "aliases": {"default": "combo/chat"},
                }
            )
        )
        loaded = load_combos(path)
        assert {c.name for c in loaded} == {"chat", "cheap"}
        chat = get_combo("combo/chat")
        assert chat.strategy == "latency"
        assert chat.policy.max_attempts == 3
        assert resolve_model_alias("default") == "combo/chat"

    def test_env_file_loaded_lazily(self, tmp_path, monkeypatch):
        import prompture.resilience.virtual as virtual

        path = tmp_path / "combos.json"
        path.write_text(json.dumps({"combos": {"envcombo": ["a/1"]}}))
        monkeypatch.setenv("PROMPTURE_COMBOS_FILE", str(path))
        monkeypatch.setattr(virtual, "_env_loaded", False)
        assert get_combo("envcombo") is not None

    def test_list_virtual_models(self):
        register_combo("chat", ["a/1"])
        register_model_alias("fast", "a/1")
        names = list_virtual_models()
        assert names[:2] == ["combo/chat", "fast"]
        assert "auto/cheap" in names

    def test_auto_driver(self, monkeypatch):
        from prompture.pipeline.routing import ModelRouter

        catalog = {
            "openai/gpt-4o-mini": "budget",
            "groq/llama-3.1-8b-instant": "budget",
            "openai/text-embedding-3-small": "budget",
            "claude/claude-opus-4-6": "premium",
            "openai/gpt-4o": "standard",
        }
        monkeypatch.setattr(ModelRouter, "available_models", lambda self: list(catalog))
        monkeypatch.setattr(ModelRouter, "model_tier", lambda self, m: catalog[m])
        prices = {"openai/gpt-4o-mini": 0.75, "groq/llama-3.1-8b-instant": 0.13, "openai/gpt-4o": 12.5}
        monkeypatch.setattr("prompture.resilience.strategies._price_per_mtok", prices.get)

        _, drv = resolve_virtual_model("auto/cheap")
        models = [t.model for t in drv.targets]
        assert "openai/text-embedding-3-small" not in models
        assert models[:2] == ["groq/llama-3.1-8b-instant", "openai/gpt-4o-mini"]
        assert drv._plan.strategy == AUTO_MODES["cheap"][1]
        assert drv.model == "auto/cheap"

        _, best = resolve_virtual_model("auto/best")
        assert best.targets[0].model == "claude/claude-opus-4-6"

        with pytest.raises(ValueError, match="Unknown auto mode"):
            resolve_virtual_model("auto/wat")

    def test_auto_with_nothing_configured(self, monkeypatch):
        from prompture.pipeline.routing import ModelRouter

        monkeypatch.setattr(ModelRouter, "available_models", lambda self: [])
        with pytest.raises(ValueError, match="no configured models"):
            resolve_virtual_model("auto/cheap")


def test_is_chat_model_filters_media():
    assert _is_chat_model("openai/gpt-4o-mini")
    for name in [
        "openai/text-embedding-3-small",
        "cohere/rerank-v3",
        "openai/whisper-1",
        "google/lyria-3",
        "openai/dall-e-3",
    ]:
        assert not _is_chat_model(name), name


def test_interleave_providers():
    assert _interleave_providers(["a/1", "a/2", "b/1", "a/3", "c/1"]) == ["a/1", "b/1", "c/1", "a/2", "a/3"]


def test_combo_driver_fails_over_like_any_resilient_driver():
    from test_resilience import _http_error

    combo = register_combo("x", [FakeDriver("a/one", [_http_error(503, "down")]), FakeDriver("b/two")])
    drv = combo.driver(breakers=BreakerRegistry(), sleep=lambda s: None)
    resp: dict[str, Any] = drv.generate("hi", {})
    assert resp["meta"]["route"]["served_by"] == "b/two"
