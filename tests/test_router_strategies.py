"""Tests for RouterAgent routing strategies (keyword, round-robin, LLM)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from prompture.agents.types import AgentResult, AgentState
from prompture.drivers.base import Driver
from prompture.groups.groups import RouterAgent, RoutingStrategy
from prompture.groups.types import GroupCallbacks


class MockDriver(Driver):
    supports_messages = True
    supports_tool_use = False

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or ["Hello from mock"])
        self._call_count = 0
        self.model = "mock-model"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._make_response()

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
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
                "total_tokens": 15,
                "cost": 0.001,
                "raw_response": {},
            },
        }


def _make_mock_agent(name: str, output_text: str = "output") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.description = f"Mock agent {name}"
    agent.output_key = None
    agent.run.return_value = AgentResult(
        output=output_text,
        output_text=output_text,
        messages=[],
        usage={},
        state=AgentState.idle,
        run_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "total_cost": 0.001},
    )
    return agent


# ---------------------------------------------------------------------------
# Keyword routing
# ---------------------------------------------------------------------------


class TestKeywordRouting:
    def test_basic_keyword_match(self):
        writer = _make_mock_agent("writer", "written content")
        coder = _make_mock_agent("coder", "code output")

        router = RouterAgent(
            agents=[writer, coder],
            strategy=RoutingStrategy.keyword,
            keywords={
                "writer": ["write", "story", "essay", "article"],
                "coder": ["code", "program", "function", "debug"],
            },
        )
        result = router.run("Write me a story about dragons")

        writer.run.assert_called_once()
        coder.run.assert_not_called()
        assert result.output_text == "written content"

    def test_keyword_multiple_matches_picks_best(self):
        writer = _make_mock_agent("writer", "written")
        coder = _make_mock_agent("coder", "coded")

        router = RouterAgent(
            agents=[writer, coder],
            strategy=RoutingStrategy.keyword,
            keywords={
                "writer": ["write", "story"],
                "coder": ["code", "program", "debug", "write"],
            },
        )
        # "write code" matches writer(1: write) and coder(2: code, write)
        router.run("write code to debug")

        coder.run.assert_called_once()  # coder has more keyword matches

    def test_keyword_no_match_uses_fallback(self):
        writer = _make_mock_agent("writer", "written")
        fallback = _make_mock_agent("fallback", "fallback output")

        router = RouterAgent(
            agents=[writer],
            strategy=RoutingStrategy.keyword,
            keywords={"writer": ["write"]},
            fallback=fallback,
        )
        result = router.run("Completely unrelated query")

        writer.run.assert_not_called()
        fallback.run.assert_called_once()
        assert result.output_text == "fallback output"

    def test_keyword_no_match_no_fallback(self):
        writer = _make_mock_agent("writer", "written")

        router = RouterAgent(
            agents=[writer],
            strategy=RoutingStrategy.keyword,
            keywords={"writer": ["write"]},
        )
        result = router.run("Something else entirely")

        writer.run.assert_not_called()
        # Returns the reason string when no match
        assert "No keywords matched" in result.output_text


# ---------------------------------------------------------------------------
# Round-robin routing
# ---------------------------------------------------------------------------


class TestRoundRobinRouting:
    def test_cycles_through_agents(self):
        a = _make_mock_agent("agent_a", "a output")
        b = _make_mock_agent("agent_b", "b output")
        c = _make_mock_agent("agent_c", "c output")

        router = RouterAgent(
            agents=[a, b, c],
            strategy=RoutingStrategy.round_robin,
        )

        r1 = router.run("first")
        assert r1.output_text == "a output"

        r2 = router.run("second")
        assert r2.output_text == "b output"

        r3 = router.run("third")
        assert r3.output_text == "c output"

        # Wraps around
        r4 = router.run("fourth")
        assert r4.output_text == "a output"

    def test_round_robin_single_agent(self):
        a = _make_mock_agent("agent_a", "a output")

        router = RouterAgent(
            agents=[a],
            strategy=RoutingStrategy.round_robin,
        )

        for _ in range(3):
            result = router.run("test")
            assert result.output_text == "a output"


# ---------------------------------------------------------------------------
# LLM routing (existing behavior preserved)
# ---------------------------------------------------------------------------


class TestLLMRouting:
    def test_llm_routing_default(self):
        writer = _make_mock_agent("writer", "written content")
        coder = _make_mock_agent("coder", "code output")

        routing_driver = MockDriver(responses=["writer"])

        router = RouterAgent(
            driver=routing_driver,
            agents=[writer, coder],
            # strategy defaults to "llm"
        )
        result = router.run("Write me a story")

        writer.run.assert_called_once()
        assert result.output_text == "written content"

    def test_llm_routing_explicit(self):
        writer = _make_mock_agent("writer", "written")

        routing_driver = MockDriver(responses=["writer"])

        router = RouterAgent(
            driver=routing_driver,
            agents=[writer],
            strategy=RoutingStrategy.llm,
        )
        router.run("test")

        writer.run.assert_called_once()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestRouterCallbacks:
    def test_on_route_decision_fires(self):
        decisions: list[tuple[str, str, float]] = []
        callbacks = GroupCallbacks(
            on_route_decision=lambda name, reason, conf: decisions.append((name, reason, conf)),
        )

        a = _make_mock_agent("agent_a", "output")
        router = RouterAgent(
            agents=[a],
            strategy=RoutingStrategy.round_robin,
            callbacks=callbacks,
        )
        router.run("test")

        assert len(decisions) == 1
        assert decisions[0][0] == "agent_a"
        assert decisions[0][2] == 1.0  # round-robin confidence

    def test_keyword_route_decision_callback(self):
        decisions: list[tuple[str, str, float]] = []
        callbacks = GroupCallbacks(
            on_route_decision=lambda name, reason, conf: decisions.append((name, reason, conf)),
        )

        writer = _make_mock_agent("writer", "output")
        router = RouterAgent(
            agents=[writer],
            strategy=RoutingStrategy.keyword,
            keywords={"writer": ["write", "story"]},
            callbacks=callbacks,
        )
        router.run("Write a story")

        assert len(decisions) == 1
        assert decisions[0][0] == "writer"
        assert "keyword" in decisions[0][1].lower()
        assert decisions[0][2] > 0


# ---------------------------------------------------------------------------
# RoutingStrategy constants
# ---------------------------------------------------------------------------


class TestRoutingStrategy:
    def test_strategy_values(self):
        assert RoutingStrategy.llm == "llm"
        assert RoutingStrategy.keyword == "keyword"
        assert RoutingStrategy.round_robin == "round_robin"
