"""Tests for SequentialGroup step_conditions (waterfall mode)."""

from __future__ import annotations

from unittest.mock import MagicMock

from prompture.agents.types import AgentResult, AgentState
from prompture.groups.groups import SequentialGroup
from prompture.groups.types import GroupCallbacks


def _make_mock_agent(name: str, output_text: str = "output", output_key: str | None = None) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.description = f"Mock agent {name}"
    agent.output_key = output_key
    agent.run.return_value = AgentResult(
        output=output_text,
        output_text=output_text,
        messages=[],
        usage={},
        state=AgentState.idle,
        run_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "total_cost": 0.001},
    )
    return agent


class TestStepConditions:
    def test_all_conditions_true_runs_all(self):
        """When all conditions return True, all agents run."""
        a = _make_mock_agent("general", "I can help with that!")
        b = _make_mock_agent("specialist", "Here's the detailed answer")
        c = _make_mock_agent("senior", "Final review")

        def always_continue(output: str, state: dict) -> bool:
            return True

        group = SequentialGroup(
            [a, b, c],
            step_conditions=[always_continue, always_continue],
        )
        result = group.run("test")

        assert result.success
        assert len(result.agent_results) == 3
        a.run.assert_called_once()
        b.run.assert_called_once()
        c.run.assert_called_once()

    def test_early_stop_on_false(self):
        """When a condition returns False, the group stops early."""
        a = _make_mock_agent("general", "I know the answer!")
        b = _make_mock_agent("specialist", "should not run")
        c = _make_mock_agent("senior", "should not run")

        def needs_escalation(output: str, state: dict) -> bool:
            return "i don't know" in output.lower()

        group = SequentialGroup(
            [a, b, c],
            step_conditions=[needs_escalation, needs_escalation],
        )
        result = group.run("test")

        assert result.success
        assert len(result.agent_results) == 1
        assert "general" in result.agent_results
        b.run.assert_not_called()
        c.run.assert_not_called()

    def test_escalation_to_second_agent(self):
        """First agent can't answer, second can."""
        a = _make_mock_agent("general", "I don't know about that topic")
        b = _make_mock_agent("specialist", "Here's the expert answer")
        c = _make_mock_agent("senior", "should not run")

        def needs_escalation(output: str, state: dict) -> bool:
            return "i don't know" in output.lower()

        group = SequentialGroup(
            [a, b, c],
            step_conditions=[needs_escalation, needs_escalation],
        )
        result = group.run("test")

        assert result.success
        assert len(result.agent_results) == 2
        a.run.assert_called_once()
        b.run.assert_called_once()
        c.run.assert_not_called()

    def test_escalation_stopped_at_metadata(self):
        """_escalation_stopped_at should be set in shared state."""
        a = _make_mock_agent("general", "I got this!", output_key="answer")

        def needs_escalation(output: str, state: dict) -> bool:
            return "i don't know" in output.lower()

        group = SequentialGroup(
            [a, _make_mock_agent("specialist", "unused")],
            step_conditions=[needs_escalation],
        )
        result = group.run("test")

        assert result.shared_state["_escalation_stopped_at"] == "general"

    def test_fewer_conditions_than_agents(self):
        """Missing conditions default to always-continue."""
        a = _make_mock_agent("a", "output a")
        b = _make_mock_agent("b", "output b")
        c = _make_mock_agent("c", "output c")

        # Only one condition — applies to first agent only
        group = SequentialGroup(
            [a, b, c],
            step_conditions=[lambda o, s: True],
        )
        result = group.run("test")

        assert len(result.agent_results) == 3

    def test_empty_conditions_list(self):
        """Empty conditions list means no conditions checked."""
        a = _make_mock_agent("a", "output")
        b = _make_mock_agent("b", "output")

        group = SequentialGroup([a, b], step_conditions=[])
        result = group.run("test")

        assert len(result.agent_results) == 2

    def test_no_step_conditions(self):
        """Without step_conditions, all agents run (backward compat)."""
        a = _make_mock_agent("a", "output")
        b = _make_mock_agent("b", "output")

        group = SequentialGroup([a, b])
        result = group.run("test")

        assert len(result.agent_results) == 2

    def test_on_step_skipped_callback(self):
        """on_step_skipped should fire for all remaining agents."""
        a = _make_mock_agent("general", "I know this")
        b = _make_mock_agent("specialist", "unused")
        c = _make_mock_agent("senior", "unused")

        skipped: list[tuple[str, str]] = []
        callbacks = GroupCallbacks(
            on_step_skipped=lambda name, reason: skipped.append((name, reason)),
        )

        group = SequentialGroup(
            [a, b, c],
            step_conditions=[lambda o, s: False],  # Stop after first
            callbacks=callbacks,
        )
        group.run("test")

        assert len(skipped) == 2
        assert skipped[0][0] == "specialist"
        assert skipped[1][0] == "senior"
        assert "general" in skipped[0][1]

    def test_condition_receives_output_and_state(self):
        """Conditions should receive the agent's output text and current state."""
        received: list[tuple[str, dict]] = []

        def capture_condition(output: str, state: dict) -> bool:
            received.append((output, dict(state)))
            return True

        a = _make_mock_agent("a", "agent a output", output_key="a_key")
        b = _make_mock_agent("b", "agent b output")

        group = SequentialGroup(
            [a, b],
            step_conditions=[capture_condition],
        )
        group.run("test")

        assert len(received) == 1
        assert received[0][0] == "agent a output"
        assert received[0][1]["a_key"] == "agent a output"

    def test_waterfall_pattern(self):
        """Full waterfall pattern: general -> specialist -> senior."""
        general = _make_mock_agent("general", "I'm not sure, outside my expertise")
        specialist = _make_mock_agent("specialist", "I can't help with that either, you should ask senior")
        senior = _make_mock_agent("senior", "Here's the definitive answer")

        def needs_escalation(output: str, state: dict) -> bool:
            signals = ["i'm not sure", "i can't help", "outside my expertise", "you should ask"]
            return any(s in output.lower() for s in signals)

        group = SequentialGroup(
            [general, specialist, senior],
            step_conditions=[needs_escalation, needs_escalation],
        )
        result = group.run("Complex question")

        assert result.success
        assert len(result.agent_results) == 3
        general.run.assert_called_once()
        specialist.run.assert_called_once()
        senior.run.assert_called_once()
