"""Tests for budget enforcement (Phase 6)."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest

from prompture.agents.agent import Agent
from prompture.agents.conversation import Conversation
from prompture.drivers.base import Driver
from prompture.exceptions import BudgetExceededError
from prompture.infra.budget import (
    BudgetPolicy,
    BudgetState,
    enforce_budget,
    estimate_cost,
    estimate_tokens,
)

# ---------------------------------------------------------------------------
# Mock driver
# ---------------------------------------------------------------------------


class MockDriver(Driver):
    """Minimal mock driver for budget tests."""

    supports_messages = True
    supports_tool_use = False

    def __init__(
        self,
        responses: list[str] | None = None,
        cost_per_call: float = 0.001,
        tokens_per_call: int = 15,
    ):
        self.responses = list(responses or ["ok"])
        self._call_count = 0
        self.model = "mock-model"
        self._cost_per_call = cost_per_call
        self._tokens_per_call = tokens_per_call

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._make_response()

    def generate_messages(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]:
        return self._make_response()

    def _make_response(self) -> dict[str, Any]:
        idx = min(self._call_count, len(self.responses) - 1)
        text = self.responses[idx]
        self._call_count += 1
        return {
            "text": text,
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": self._tokens_per_call,
                "cost": self._cost_per_call,
                "raw_response": {},
                "model_name": "mock-model",
            },
        }


# =========================================================================
# estimate_tokens
# =========================================================================


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_text_heuristic(self):
        # Without tiktoken the heuristic is len(text) // 4
        result = estimate_tokens("Hello world")
        assert result >= 1

    def test_long_text(self):
        text = "a" * 400
        result = estimate_tokens(text)
        assert result >= 50  # at least 400 // 4 = 100 via heuristic

    def test_tiktoken_mock(self):
        """When tiktoken is available, it should use it."""
        # We don't require tiktoken to be installed, so we test both paths.
        result = estimate_tokens("Hello, this is a test.")
        assert isinstance(result, int)
        assert result >= 1


# =========================================================================
# estimate_cost
# =========================================================================


class TestEstimateCost:
    def test_known_model_with_mocked_rates(self):
        with patch(
            "prompture.infra.model_rates.get_model_rates",
            return_value={"input": 3.0, "output": 15.0},
        ):
            cost = estimate_cost("openai/gpt-4", 1000, 500)
            # (1000 * 3 + 500 * 15) / 1_000_000 = 0.0105
            assert abs(cost - 0.0105) < 1e-6

    def test_unknown_model_returns_zero(self):
        with patch("prompture.infra.model_rates.get_model_rates", return_value=None):
            assert estimate_cost("fake/model", 1000, 500) == 0.0

    def test_no_model_id(self):
        assert estimate_cost("noslash", 100, 50) == 0.0


# =========================================================================
# BudgetState
# =========================================================================


class TestBudgetState:
    def test_not_exceeded(self):
        state = BudgetState(cost_used=0.5, max_cost=1.0, tokens_used=100, max_tokens=1000)
        assert not state.exceeded
        assert state.cost_remaining == 0.5
        assert state.tokens_remaining == 900
        assert abs(state.cost_fraction - 0.5) < 1e-6

    def test_exceeded_by_cost(self):
        state = BudgetState(cost_used=1.5, max_cost=1.0)
        assert state.exceeded
        assert state.cost_remaining == 0.0

    def test_exceeded_by_tokens(self):
        state = BudgetState(tokens_used=1500, max_tokens=1000)
        assert state.exceeded
        assert state.tokens_remaining == 0

    def test_no_limits(self):
        state = BudgetState(cost_used=100.0, tokens_used=999999)
        assert not state.exceeded
        assert state.cost_remaining is None
        assert state.tokens_remaining is None
        assert state.cost_fraction is None

    def test_cost_fraction_zero_max(self):
        state = BudgetState(cost_used=0.5, max_cost=0.0)
        assert state.cost_fraction is None


# =========================================================================
# enforce_budget
# =========================================================================


class TestEnforceBudget:
    def test_hard_stop_raises(self):
        state = BudgetState(cost_used=1.5, max_cost=1.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            enforce_budget(state, BudgetPolicy.hard_stop)
        assert exc_info.value.budget_state is state

    def test_hard_stop_ok(self):
        state = BudgetState(cost_used=0.5, max_cost=1.0)
        assert enforce_budget(state, BudgetPolicy.hard_stop) is None

    def test_warn_logs(self, caplog):
        state = BudgetState(cost_used=1.5, max_cost=1.0)
        with caplog.at_level(logging.WARNING, logger="prompture.budget"):
            result = enforce_budget(state, BudgetPolicy.warn_and_continue)
        assert result is None
        assert "Budget exceeded" in caplog.text

    def test_warn_no_log_when_ok(self, caplog):
        state = BudgetState(cost_used=0.5, max_cost=1.0)
        with caplog.at_level(logging.WARNING, logger="prompture.budget"):
            result = enforce_budget(state, BudgetPolicy.warn_and_continue)
        assert result is None
        assert "Budget exceeded" not in caplog.text

    def test_degrade_returns_fallback(self):
        state = BudgetState(cost_used=0.85, max_cost=1.0)  # 85% > 80% threshold
        result = enforce_budget(
            state,
            BudgetPolicy.degrade,
            fallback_models=["cheap/model"],
            current_model="expensive/model",
        )
        assert result == "cheap/model"

    def test_degrade_at_80_percent(self):
        state = BudgetState(cost_used=0.80, max_cost=1.0)  # exactly at threshold
        result = enforce_budget(
            state,
            BudgetPolicy.degrade,
            fallback_models=["cheap/model"],
            current_model="expensive/model",
        )
        assert result == "cheap/model"

    def test_degrade_below_threshold(self):
        state = BudgetState(cost_used=0.79, max_cost=1.0)  # below threshold
        result = enforce_budget(
            state,
            BudgetPolicy.degrade,
            fallback_models=["cheap/model"],
            current_model="expensive/model",
        )
        assert result is None

    def test_degrade_no_fallbacks_raises(self):
        state = BudgetState(cost_used=1.5, max_cost=1.0)
        with pytest.raises(BudgetExceededError):
            enforce_budget(state, BudgetPolicy.degrade)

    def test_degrade_callback_fires(self):
        state = BudgetState(cost_used=0.9, max_cost=1.0)
        calls: list[tuple[str, str, Any]] = []

        def on_fallback(old: str, new: str, bs: Any) -> None:
            calls.append((old, new, bs))

        enforce_budget(
            state,
            BudgetPolicy.degrade,
            fallback_models=["cheap/model"],
            current_model="expensive/model",
            on_model_fallback=on_fallback,
        )
        assert len(calls) == 1
        assert calls[0][0] == "expensive/model"
        assert calls[0][1] == "cheap/model"

    def test_degrade_skips_current_model(self):
        state = BudgetState(cost_used=0.9, max_cost=1.0)
        result = enforce_budget(
            state,
            BudgetPolicy.degrade,
            fallback_models=["expensive/model", "cheap/model"],
            current_model="expensive/model",
        )
        assert result == "cheap/model"


# =========================================================================
# BudgetExceededError
# =========================================================================


class TestBudgetExceededError:
    def test_backward_compat(self):
        err = BudgetExceededError("over budget")
        assert str(err) == "over budget"
        assert err.budget_state is None

    def test_with_budget_state(self):
        state = BudgetState(cost_used=1.5, max_cost=1.0)
        err = BudgetExceededError("over", budget_state=state)
        assert err.budget_state is state


# =========================================================================
# Conversation budget integration
# =========================================================================


class TestConversationBudget:
    def test_no_budget_backward_compat(self):
        """Without budget params, Conversation works exactly as before."""
        driver = MockDriver(responses=["Hello"])
        conv = Conversation(driver=driver)
        resp = conv.ask("Hi")
        assert resp == "Hello"
        assert conv.budget_remaining is None

    def test_hard_stop_raises_on_ask(self):
        driver = MockDriver(responses=["first", "second"], cost_per_call=0.006)
        conv = Conversation(
            driver=driver,
            max_cost=0.01,
            budget_policy=BudgetPolicy.hard_stop,
        )
        # First call: cost = 0.006 < 0.01 -- ok
        conv.ask("Hello")
        # Second call: cost already at 0.006, then after: 0.012 > 0.01
        # But _check_budget runs before the call, at cost=0.006 not yet exceeded
        conv.ask("World")
        # Third call: cost = 0.012, now exceeded
        with pytest.raises(BudgetExceededError):
            conv.ask("Boom")

    def test_warn_continues(self, caplog):
        driver = MockDriver(responses=["first", "second", "third"], cost_per_call=0.006)
        conv = Conversation(
            driver=driver,
            max_cost=0.01,
            budget_policy=BudgetPolicy.warn_and_continue,
        )
        conv.ask("Hello")  # cost = 0.006
        conv.ask("World")  # cost = 0.012
        with caplog.at_level(logging.WARNING, logger="prompture.budget"):
            conv.ask("Still going")  # cost = 0.018, should warn
        assert "Budget exceeded" in caplog.text

    def test_degrade_switches_model(self):
        driver = MockDriver(responses=["first", "second", "third"], cost_per_call=0.009)
        conv = Conversation(
            driver=driver,
            max_cost=0.01,
            budget_policy=BudgetPolicy.degrade,
            fallback_models=["cheap/model"],
        )
        conv.ask("Hello")  # cost = 0.009, 90% >= 80% threshold

        # After the first call, usage shows cost=0.009
        # Next call will trigger degrade and _switch_model
        with patch("prompture.agents.conversation.get_driver_for_model") as mock_get:
            cheap_driver = MockDriver(responses=["from cheap"])
            mock_get.return_value = cheap_driver
            conv.ask("World")

        mock_get.assert_called_once_with("cheap/model")
        assert conv._model_name == "cheap/model"

    def test_budget_remaining_property(self):
        driver = MockDriver(responses=["Hello"], cost_per_call=0.003)
        conv = Conversation(
            driver=driver,
            max_cost=0.01,
            max_tokens=1000,
            budget_policy=BudgetPolicy.hard_stop,
        )
        conv.ask("Hi")

        remaining = conv.budget_remaining
        assert remaining is not None
        assert remaining["cost_remaining"] == pytest.approx(0.007, abs=1e-6)
        assert remaining["tokens_remaining"] == 985
        assert remaining["exceeded"] is False
        assert remaining["active_model"] == ""

    def test_max_tokens_enforcement(self):
        driver = MockDriver(
            responses=["a", "b", "c"],
            tokens_per_call=500,
            cost_per_call=0.0,
        )
        conv = Conversation(
            driver=driver,
            max_tokens=1000,
            budget_policy=BudgetPolicy.hard_stop,
        )
        conv.ask("Hello")  # 500 tokens
        conv.ask("World")  # 1000 tokens
        with pytest.raises(BudgetExceededError):
            conv.ask("Boom")  # 1000 >= 1000, exceeded


# =========================================================================
# Agent budget integration
# =========================================================================


class TestAgentBudget:
    def test_legacy_soft_preserved(self):
        """Without budget_policy, max_cost triggers soft skip (no exception)."""
        driver = MockDriver(responses=["Hello"])
        agent = Agent(driver=driver, max_cost=0.0001)
        # The old behavior: max_cost is a soft limit that just skips retries
        result = agent.run("Hi")
        assert result.output_text == "Hello"

    def test_hard_stop_on_agent(self):
        """With budget_policy=hard_stop, agent raises on budget exceeded."""
        # The agent will try to enforce budget before the conv.ask() call.
        # Since it's a fresh session with 0 cost, the first call should succeed.
        driver = MockDriver(responses=["ok"], cost_per_call=0.001)
        agent = Agent(
            driver=driver,
            max_cost=0.001,
            budget_policy=BudgetPolicy.hard_stop,
        )
        # First run succeeds (pre-call check has 0 cost)
        result = agent.run("Hi")
        assert result.output_text == "ok"

    def test_budget_forwarded_to_conv(self):
        """When budget_policy is set, params are forwarded to Conversation."""
        driver = MockDriver(responses=["ok"])
        agent = Agent(
            driver=driver,
            max_cost=1.0,
            max_tokens=5000,
            budget_policy=BudgetPolicy.warn_and_continue,
            fallback_models=["cheap/model"],
        )
        result = agent.run("Hi")
        assert result.output_text == "ok"

    def test_max_tokens_on_agent(self):
        driver = MockDriver(responses=["ok"], tokens_per_call=100)
        agent = Agent(
            driver=driver,
            max_tokens=50,
            budget_policy=BudgetPolicy.hard_stop,
        )
        # The conversation-level budget check kicks in; the first call
        # passes the agent-level check (fresh session), but the conversation
        # check after accumulation will block the next call.
        result = agent.run("Hi")
        assert result.output_text == "ok"

    def test_is_over_budget_with_max_tokens(self):
        """_is_over_budget returns True when max_tokens is exceeded."""
        from prompture.infra.session import UsageSession

        driver = MockDriver()
        agent = Agent(driver=driver, max_tokens=100)
        session = UsageSession()
        session.total_tokens = 150
        assert agent._is_over_budget(session) is True

    def test_is_over_budget_with_max_cost(self):
        """_is_over_budget returns True when max_cost is exceeded."""
        from prompture.infra.session import UsageSession

        driver = MockDriver()
        agent = Agent(driver=driver, max_cost=0.01)
        session = UsageSession()
        session.cost = 0.02
        assert agent._is_over_budget(session) is True

    def test_is_over_budget_neither(self):
        """_is_over_budget returns False when neither limit is set."""
        from prompture.infra.session import UsageSession

        driver = MockDriver()
        agent = Agent(driver=driver)
        session = UsageSession()
        session.cost = 100.0
        session.total_tokens = 999999
        assert agent._is_over_budget(session) is False
