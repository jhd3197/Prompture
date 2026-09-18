"""Cost detail regressions: billing dimensions, precision, and provenance."""

from unittest.mock import patch

import pytest

from prompture.infra.cost_mixin import CostMixin
from prompture.infra.pricing import ResolvedRates


@pytest.fixture
def calculator():
    return CostMixin()


def cost(calculator, rates, provider="openai", model="test", prompt=1000, output=100, **kwargs):
    with patch("prompture.infra.model_rates.get_model_rates", return_value=rates):
        return calculator._calculate_cost_details(provider, model, prompt, output, **kwargs)


class TestCostReporting:
    def test_unknown_distinct_from_free(self, calculator):
        unknown = cost(calculator, None)
        free = cost(calculator, {"input": 0, "output": 0})
        assert unknown["cost"] == free["cost"] == 0
        assert unknown["cost_status"] == "unknown"
        assert not unknown["rates_available"]
        assert free["cost_status"] == "estimated"
        assert free["rates_available"]

    def test_mixed_ttls_override_requested_ttl(self, calculator):
        result = cost(
            calculator,
            {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75},
            provider="claude",
            prompt=1000,
            output=0,
            cached_tokens=400,
            cache_creation_tokens=500,
            cache_creation_5m_tokens=200,
            cache_creation_1h_tokens=300,
            cache_write_multiplier=1.6,
        )
        assert result["cost"] == pytest.approx(0.0003 + 0.00012 + 0.00075 + 0.0018)
        assert result["cache_savings"] == pytest.approx(0.0027 - 0.00012 - 0.00075 - 0.0018)
        assert result["cost_status"] == "estimated"

    @pytest.mark.parametrize("prompt, expected", [(272000, 1.39), (300000, 3.045)])
    def test_long_context_boundary(self, calculator, prompt, expected):
        result = cost(calculator, {"input": 5, "output": 30}, model="gpt-5.5", prompt=prompt, output=1000)
        assert result["cost"] == pytest.approx(expected)

    def test_dated_snapshot_and_stacked_modifiers(self, calculator):
        result = cost(
            calculator,
            {"input": 5, "output": 30},
            model="gpt-5.5-2026-04-23",
            prompt=300000,
            output=1000,
            service_tier="flex",
            inference_geo="regional",
        )
        assert result["cost"] == pytest.approx(3.045 * 0.5 * 1.1)

    def test_claude_geo_and_tool_charges(self, calculator):
        result = cost(
            calculator,
            {"input": 3, "output": 15},
            provider="claude",
            model="claude-sonnet-4-6",
            inference_geo="us",
            tool_usage={"web_search_requests": 2, "web_fetch_requests": 1},
        )
        assert result["cost"] == pytest.approx(0.0045 * 1.1 + 0.02)

    def test_unverified_dimensions_are_partial(self, calculator):
        result = cost(
            calculator,
            {"input": 5, "output": 30},
            service_tier="priority",
            tool_usage={"code_interpreter": 1},
            usage_complete=False,
        )
        assert result["cost_status"] == "partial"
        assert set(result["pricing"]["unpriced"]) == {
            "service_tier:priority",
            "tool:code_interpreter",
            "incomplete_usage",
        }

    def test_small_cost_and_provenance(self, calculator):
        rates = ResolvedRates({"input": 0.05, "output": 0.1}, "custom_contract")
        result = cost(calculator, rates, prompt=1, output=0)
        assert result["cost"] == pytest.approx(0.00000005)
        assert result["pricing"]["source"] == "custom_contract"
        assert result["pricing"]["rates_per_million"] == {"input": 0.05, "output": 0.1}
        assert list(rates) == ["input", "output"]

    def test_unknown_model_does_not_inherit_long_context_rule(self, calculator):
        result = cost(calculator, {"input": 5, "output": 30}, model="gpt-5.5-custom", prompt=300000, output=1000)
        assert result["cost"] == pytest.approx(1.53)

    def test_breakdown_mismatch_is_visible(self, calculator):
        result = cost(
            calculator,
            {"input": 3, "output": 15},
            provider="claude",
            cache_creation_tokens=500,
            cache_creation_5m_tokens=100,
        )
        assert result["cost_status"] == "partial"
        assert "cache_creation_breakdown_mismatch" in result["pricing"]["unpriced"]
