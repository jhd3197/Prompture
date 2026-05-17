"""Tests for :func:`prompture.estimate_call_cost`."""

from __future__ import annotations

from unittest.mock import patch

from prompture import CostEstimate, estimate_call_cost, estimate_cost, estimate_tokens


# ---------------------------------------------------------------------------
# token counting
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_non_empty_returns_positive(self):
        assert estimate_tokens("Hello world!") > 0

    def test_longer_text_more_tokens(self):
        a = estimate_tokens("short")
        b = estimate_tokens("this is a considerably longer piece of text content")
        assert b > a


# ---------------------------------------------------------------------------
# estimate_call_cost — shape
# ---------------------------------------------------------------------------


class TestEstimateCallCost:
    def test_returns_dataclass(self):
        e = estimate_call_cost(
            "openai/gpt-4o-mini",
            prompt="Hello, world!",
            completion=200,
        )
        assert isinstance(e, CostEstimate)
        assert e.model == "openai/gpt-4o-mini"
        assert e.input_tokens > 0
        assert e.output_tokens == 200
        assert e.total_tokens == e.input_tokens + e.output_tokens
        assert e.total_cost == round(e.input_cost + e.output_cost, 6)

    def test_accepts_int_tokens_directly(self):
        e = estimate_call_cost(
            "openai/gpt-4o-mini",
            prompt=1000,
            completion=500,
        )
        assert e.input_tokens == 1000
        assert e.output_tokens == 500

    def test_accepts_text_for_completion(self):
        e = estimate_call_cost(
            "openai/gpt-4o-mini",
            prompt="hi",
            completion="this is the expected reply",
        )
        assert e.output_tokens > 0

    def test_default_expected_completion_tokens(self):
        e = estimate_call_cost("openai/gpt-4o-mini", prompt="hi")
        # No completion provided → falls back to 500.
        assert e.output_tokens == 500

    def test_custom_expected_completion_tokens(self):
        e = estimate_call_cost(
            "openai/gpt-4o-mini",
            prompt="hi",
            expected_completion_tokens=50,
        )
        assert e.output_tokens == 50


# ---------------------------------------------------------------------------
# Pricing resolution
# ---------------------------------------------------------------------------


class TestPricingResolution:
    def test_uses_real_rates_when_available(self):
        # Stub the rate lookup so the test stays deterministic.
        def _fake_rates(provider, model):
            if provider == "fake" and model == "fast":
                return {"input": 1.0, "output": 2.0}  # per 1M tokens
            return None

        with patch("prompture.infra.model_rates.get_model_rates", _fake_rates):
            e = estimate_call_cost("fake/fast", prompt=1_000_000, completion=1_000_000)
        assert e.rates_available is True
        # Per 1M tokens: input=1$, output=2$, total=3$.
        assert e.input_cost == 1.0
        assert e.output_cost == 2.0
        assert e.total_cost == 3.0

    def test_missing_rates_returns_zero_cost(self):
        with patch("prompture.infra.model_rates.get_model_rates", lambda p, m: None):
            e = estimate_call_cost("unknown/model", prompt=1000, completion=500)
        assert e.rates_available is False
        assert e.input_cost == 0.0
        assert e.output_cost == 0.0
        assert e.total_cost == 0.0
        # Tokens still counted.
        assert e.total_tokens == 1500

    def test_zero_rates_treated_as_unavailable(self):
        with patch(
            "prompture.infra.model_rates.get_model_rates",
            lambda p, m: {"input": 0.0, "output": 0.0},
        ):
            e = estimate_call_cost("free/local", prompt=1000, completion=500)
        assert e.rates_available is False
        assert e.total_cost == 0.0

    def test_model_without_provider_prefix(self):
        # No '/' in model string → can't resolve provider → rates None.
        e = estimate_call_cost("orphan-model", prompt=10, completion=10)
        assert e.rates_available is False


# ---------------------------------------------------------------------------
# Token-counter label
# ---------------------------------------------------------------------------


class TestTokenCounterLabel:
    def test_int_only_inputs_yield_exact(self):
        e = estimate_call_cost("openai/gpt-4o-mini", prompt=100, completion=200)
        assert e.token_counter == "exact"

    def test_text_input_labels_estimator(self):
        e = estimate_call_cost("openai/gpt-4o-mini", prompt="hello world", completion=100)
        # Either tiktoken (if installed) or heuristic.
        assert e.token_counter in {"tiktoken", "heuristic"}


# ---------------------------------------------------------------------------
# Back-compat: legacy estimate_cost still works
# ---------------------------------------------------------------------------


class TestLegacyEstimateCost:
    def test_estimate_cost_returns_float(self):
        cost = estimate_cost("openai/gpt-4o-mini", 1000, 500)
        assert isinstance(cost, float)
        assert cost >= 0.0
