"""Tests for the pluggable pricing-source registry.

Covers:
- Source ordering (priority-based, lowest first wins).
- LocalKBPricingSource reads ``cost`` blocks from infra/rates/*.json.
- ModelsDevPricingSource reads cost from cached models.dev data.
- Custom user sources can be registered, replacing or stacking with built-ins.
- Settings ``pricing_source`` flag picks the right built-ins on init.
- Backwards compat: ``get_model_rates`` keeps the same signature/return shape.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import patch

import pytest

from prompture.infra import pricing
from prompture.infra.model_rates import get_model_rates


@pytest.fixture(autouse=True)
def _reset_registry():
    """Force a fresh registry for every test (and restore defaults after)."""
    pricing.clear_pricing_sources()
    yield
    pricing.clear_pricing_sources()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubSource:
    def __init__(
        self,
        name: str,
        priority: int,
        rates: Optional[dict[str, float]] = None,
    ) -> None:
        self.name = name
        self.priority = priority
        self._rates = rates
        self.calls: list[tuple[str, str]] = []

    def get_rates(self, provider: str, model_id: str):
        self.calls.append((provider, model_id))
        return self._rates


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_resolve_returns_first_priority(self):
        cheap = _StubSource("cheap", priority=10, rates={"input": 1.0, "output": 2.0})
        expensive = _StubSource("expensive", priority=100, rates={"input": 99.0, "output": 99.0})
        pricing.register_pricing_source(expensive)
        pricing.register_pricing_source(cheap)

        result = pricing.resolve_rates("openai", "anything")
        assert result == {"input": 1.0, "output": 2.0}
        # Lower priority should not even be queried — short-circuit
        assert cheap.calls == [("openai", "anything")]
        assert expensive.calls == []

    def test_falls_through_when_first_returns_none(self):
        empty = _StubSource("empty", priority=10, rates=None)
        full = _StubSource("full", priority=100, rates={"input": 1.0, "output": 2.0})
        pricing.register_pricing_source(empty)
        pricing.register_pricing_source(full)

        result = pricing.resolve_rates("openai", "x")
        assert result == {"input": 1.0, "output": 2.0}
        assert empty.calls == [("openai", "x")]
        assert full.calls == [("openai", "x")]

    def test_falls_through_when_first_returns_partial_rates(self):
        # input but no output → not usable, must fall through
        partial = _StubSource("partial", priority=10, rates={"input": 1.0})
        full = _StubSource("full", priority=100, rates={"input": 5.0, "output": 10.0})
        pricing.register_pricing_source(partial)
        pricing.register_pricing_source(full)

        result = pricing.resolve_rates("openai", "x")
        assert result == {"input": 5.0, "output": 10.0}

    def test_replace_overwrites_same_name(self):
        original = _StubSource("foo", 10, rates={"input": 1.0, "output": 1.0})
        pricing.register_pricing_source(original)

        with pytest.raises(ValueError):
            pricing.register_pricing_source(_StubSource("foo", 10, rates={"input": 9.0, "output": 9.0}))

        replacement = _StubSource("foo", 10, rates={"input": 9.0, "output": 9.0})
        pricing.register_pricing_source(replacement, replace=True)

        result = pricing.resolve_rates("any", "any")
        assert result == {"input": 9.0, "output": 9.0}

    def test_unregister(self):
        pricing.register_pricing_source(_StubSource("a", 10, rates={"input": 1.0, "output": 1.0}))
        assert pricing.unregister_pricing_source("a") is True
        assert pricing.unregister_pricing_source("a") is False
        assert pricing.resolve_rates("any", "any") is None

    def test_source_exception_is_swallowed(self):
        class _Boom:
            name = "boom"
            priority = 10

            def get_rates(self, provider, model_id):
                raise RuntimeError("kaboom")

        pricing.register_pricing_source(_Boom())
        pricing.register_pricing_source(_StubSource("ok", 100, rates={"input": 1.0, "output": 1.0}))

        # Resolver must not raise; the failing source is skipped
        assert pricing.resolve_rates("any", "any") == {"input": 1.0, "output": 1.0}


# ---------------------------------------------------------------------------
# Built-in: LocalKBPricingSource
# ---------------------------------------------------------------------------


class TestLocalKBPricingSource:
    def test_loads_seeded_openai_gpt55(self):
        # gpt-5.5 was seeded with verified pricing in this PR.
        src = pricing.LocalKBPricingSource()
        rates = src.get_rates("openai", "gpt-5.5")
        assert rates == {"input": 5.0, "output": 30.0, "cache_read": 0.5}

    def test_returns_none_for_unknown_model(self):
        src = pricing.LocalKBPricingSource()
        assert src.get_rates("openai", "made-up-model-xyz") is None

    def test_returns_none_for_known_model_without_cost_block(self):
        # gpt-5-nano has capabilities but no cost block in openai.json.
        src = pricing.LocalKBPricingSource()
        assert src.get_rates("openai", "gpt-5-nano") is None

    def test_strips_date_suffix_for_lookup(self):
        src = pricing.LocalKBPricingSource()
        # gpt-5.5 with a date stamp should resolve to the base model's pricing
        rates = src.get_rates("openai", "gpt-5.5-2026-01-15")
        assert rates == {"input": 5.0, "output": 30.0, "cache_read": 0.5}


# ---------------------------------------------------------------------------
# Built-in: ModelsDevPricingSource
# ---------------------------------------------------------------------------


class TestModelsDevPricingSource:
    def test_returns_rates_when_lookup_finds_entry(self):
        src = pricing.ModelsDevPricingSource()
        with patch(
            "prompture.infra.model_rates._lookup_model",
            return_value={"cost": {"input": 1.5, "output": 2.5, "cache_read": 0.15}},
        ):
            assert src.get_rates("openai", "x") == {
                "input": 1.5,
                "output": 2.5,
                "cache_read": 0.15,
            }

    def test_returns_none_when_no_entry(self):
        src = pricing.ModelsDevPricingSource()
        with patch("prompture.infra.model_rates._lookup_model", return_value=None):
            assert src.get_rates("openai", "x") is None

    def test_returns_none_when_entry_has_no_cost(self):
        src = pricing.ModelsDevPricingSource()
        with patch("prompture.infra.model_rates._lookup_model", return_value={"name": "x"}):
            assert src.get_rates("openai", "x") is None


# ---------------------------------------------------------------------------
# Default-source bootstrapping by setting
# ---------------------------------------------------------------------------


class TestDefaultBootstrap:
    def _bootstrap_with_mode(self, mode: str) -> list[str]:
        pricing.clear_pricing_sources()
        pricing._initialize_default_sources(mode=mode)
        return [s.name for s in pricing.get_pricing_sources()]

    def test_local_first_registers_both(self):
        assert self._bootstrap_with_mode("local_first") == ["local_kb", "models_dev"]

    def test_local_only_registers_only_local(self):
        assert self._bootstrap_with_mode("local_only") == ["local_kb"]

    def test_models_dev_only_registers_only_models_dev(self):
        assert self._bootstrap_with_mode("models_dev_only") == ["models_dev"]

    def test_unknown_mode_falls_back_to_local_first(self):
        assert self._bootstrap_with_mode("nonsense") == ["local_kb", "models_dev"]


# ---------------------------------------------------------------------------
# get_model_rates backwards compatibility & local-first behaviour
# ---------------------------------------------------------------------------


class TestGetModelRatesIntegration:
    def test_local_kb_wins_over_models_dev(self):
        """The seeded openai gpt-5.5 entry should be returned even if
        models.dev would have served different numbers."""
        pricing.clear_pricing_sources()
        pricing._initialize_default_sources(mode="local_first")
        with patch(
            "prompture.infra.model_rates._lookup_model",
            return_value={"cost": {"input": 999.0, "output": 999.0}},
        ):
            rates = get_model_rates("openai", "gpt-5.5")
        assert rates is not None
        assert rates["input"] == 5.0  # from local KB, NOT 999.0 from mocked models.dev
        assert rates["cache_read"] == 0.5

    def test_falls_through_to_models_dev_when_not_in_local_kb(self):
        pricing.clear_pricing_sources()
        pricing._initialize_default_sources(mode="local_first")
        with patch(
            "prompture.infra.model_rates._lookup_model",
            return_value={"cost": {"input": 7.7, "output": 8.8}},
        ):
            rates = get_model_rates("openai", "model-not-in-local-kb")
        assert rates == {"input": 7.7, "output": 8.8}

    def test_local_only_returns_none_when_missing(self):
        pricing.clear_pricing_sources()
        pricing._initialize_default_sources(mode="local_only")
        with patch(
            "prompture.infra.model_rates._lookup_model",
            return_value={"cost": {"input": 7.7, "output": 8.8}},
        ):
            rates = get_model_rates("openai", "model-not-in-local-kb")
        # Even though models.dev would return rates, local_only refuses to fall through.
        assert rates is None
