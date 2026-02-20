"""Multi-agent debate group.

Provides :class:`DebateGroup` and :class:`AsyncDebateGroup` for
structured debates between agents with rounds, positions, and
optional judge summaries.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..agents.types import AgentResult, AgentState
from .groups import _agent_name
from .types import (
    AgentError,
    GroupCallbacks,
    GroupResult,
    GroupStep,
    _aggregate_usage,
)

logger = logging.getLogger("prompture.groups.debate")


@dataclass
class DebateEntry:
    """A single entry in a debate transcript."""

    round: int
    agent_name: str
    position: str | None
    content: str
    timestamp: float
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateResult(GroupResult):
    """Outcome of a debate execution.

    Extends :class:`GroupResult` with debate-specific fields.
    """

    transcript: list[DebateEntry] = field(default_factory=list)
    rounds_completed: int = 0
    judge_verdict: str | None = None


@dataclass
class DebateConfig:
    """Configuration for a multi-agent debate.

    Args:
        rounds: Number of debate rounds.
        positions: Mapping of agent name to position label
            (e.g. ``{"bot_a": "FOR", "bot_b": "AGAINST"}``).
        judge: Optional agent that summarizes the debate.
        judge_prompt_template: Template for the judge prompt.
            Must contain ``{topic}`` and ``{transcript}`` placeholders.
        show_position_in_prompt: Whether to inject position instructions
            into each agent's prompt.
    """

    rounds: int = 2
    positions: dict[str, str] | None = None
    judge: Any | None = None
    judge_prompt_template: str = (
        "The following debate took place on the topic: {topic}\n\n"
        "{transcript}\n\n"
        "Provide a balanced summary and verdict."
    )
    show_position_in_prompt: bool = True


class DebateGroup:
    """Multi-agent debate with rounds, positions, and optional judge.

    For each round, every agent receives the topic plus the full
    transcript of previous responses. If positions are configured,
    each agent's prompt is augmented with its position instruction.

    After all rounds, an optional judge agent receives the full
    transcript and produces a summary/verdict.

    Args:
        agents: List of agents participating in the debate.
        config: Debate configuration (rounds, positions, judge).
        state: Initial shared state dict.
        callbacks: Observability hooks.
    """

    def __init__(
        self,
        agents: list[Any],
        config: DebateConfig | None = None,
        *,
        state: dict[str, Any] | None = None,
        callbacks: GroupCallbacks | None = None,
    ) -> None:
        self._agents = [(a, _agent_name(a, i)) for i, a in enumerate(agents)]
        self._config = config or DebateConfig()
        self._state: dict[str, Any] = dict(state) if state else {}
        self._callbacks = callbacks or GroupCallbacks()
        self._stop_requested = False

    def stop(self) -> None:
        """Request graceful shutdown after the current agent finishes."""
        self._stop_requested = True

    def _build_prompt(
        self,
        topic: str,
        agent_name: str,
        transcript: list[DebateEntry],
        round_num: int,
    ) -> str:
        """Build the per-agent debate prompt."""
        parts: list[str] = [f"Topic: {topic}"]

        # Position instruction
        if self._config.show_position_in_prompt and self._config.positions:
            position = self._config.positions.get(agent_name)
            if position:
                parts.append(
                    f"\nYour position: {position}. "
                    "Argue this position and respond to previous arguments."
                )

        # Transcript so far
        if transcript:
            parts.append("\n--- Previous arguments ---")
            for entry in transcript:
                pos_label = f" [{entry.position}]" if entry.position else ""
                parts.append(f"\n{entry.agent_name}{pos_label} (round {entry.round}):\n{entry.content}")
            parts.append("\n--- End of previous arguments ---")

        parts.append(f"\nRound {round_num + 1}: Present your argument.")
        return "\n".join(parts)

    def _format_transcript(self, transcript: list[DebateEntry]) -> str:
        """Format the full transcript for the judge."""
        lines: list[str] = []
        for entry in transcript:
            pos_label = f" [{entry.position}]" if entry.position else ""
            lines.append(f"[Round {entry.round + 1}] {entry.agent_name}{pos_label}:")
            lines.append(entry.content)
            lines.append("")
        return "\n".join(lines)

    def run(self, topic: str) -> DebateResult:
        """Execute the debate."""
        self._stop_requested = False
        t0 = time.perf_counter()
        transcript: list[DebateEntry] = []
        timeline: list[GroupStep] = []
        agent_results: dict[str, Any] = {}
        errors: list[AgentError] = []
        usage_summaries: list[dict[str, Any]] = []
        rounds_completed = 0

        positions = self._config.positions or {}

        for round_num in range(self._config.rounds):
            if self._stop_requested:
                break

            if self._callbacks.on_round_start:
                self._callbacks.on_round_start(round_num)

            for agent, name in self._agents:
                if self._stop_requested:
                    break

                effective = self._build_prompt(topic, name, transcript, round_num)

                if self._callbacks.on_agent_start:
                    self._callbacks.on_agent_start(name, effective)

                step_t0 = time.perf_counter()
                try:
                    result = agent.run(effective)
                    duration_ms = (time.perf_counter() - step_t0) * 1000

                    result_key = f"{name}_round{round_num}"
                    agent_results[result_key] = result
                    usage = getattr(result, "run_usage", {})
                    usage_summaries.append(usage)

                    entry = DebateEntry(
                        round=round_num,
                        agent_name=name,
                        position=positions.get(name),
                        content=result.output_text,
                        timestamp=step_t0,
                        usage=usage,
                    )
                    transcript.append(entry)

                    timeline.append(
                        GroupStep(
                            agent_name=name,
                            step_type="debate_argument",
                            timestamp=step_t0,
                            duration_ms=duration_ms,
                            usage_delta=usage,
                        )
                    )

                    if self._callbacks.on_agent_complete:
                        self._callbacks.on_agent_complete(name, result)

                except Exception as exc:
                    duration_ms = (time.perf_counter() - step_t0) * 1000
                    err = AgentError(
                        agent_name=name,
                        error=exc,
                    )
                    errors.append(err)
                    timeline.append(
                        GroupStep(
                            agent_name=name,
                            step_type="debate_error",
                            timestamp=step_t0,
                            duration_ms=duration_ms,
                            error=str(exc),
                        )
                    )
                    if self._callbacks.on_agent_error:
                        self._callbacks.on_agent_error(name, exc)

            rounds_completed = round_num + 1

            if self._callbacks.on_round_complete:
                self._callbacks.on_round_complete(round_num)

        # Judge phase
        judge_verdict: str | None = None
        if self._config.judge and transcript and not self._stop_requested:
            judge = self._config.judge
            judge_name = getattr(judge, "name", "judge") or "judge"
            formatted = self._format_transcript(transcript)
            judge_prompt = (
                self._config.judge_prompt_template
                .replace("{topic}", topic)
                .replace("{transcript}", formatted)
            )

            if self._callbacks.on_agent_start:
                self._callbacks.on_agent_start(judge_name, judge_prompt)

            step_t0 = time.perf_counter()
            try:
                judge_result = judge.run(judge_prompt)
                duration_ms = (time.perf_counter() - step_t0) * 1000
                judge_verdict = judge_result.output_text
                agent_results[judge_name] = judge_result
                usage = getattr(judge_result, "run_usage", {})
                usage_summaries.append(usage)

                timeline.append(
                    GroupStep(
                        agent_name=judge_name,
                        step_type="debate_judge",
                        timestamp=step_t0,
                        duration_ms=duration_ms,
                        usage_delta=usage,
                    )
                )

                if self._callbacks.on_agent_complete:
                    self._callbacks.on_agent_complete(judge_name, judge_result)

            except Exception as exc:
                duration_ms = (time.perf_counter() - step_t0) * 1000
                errors.append(AgentError(agent_name=judge_name, error=exc))
                timeline.append(
                    GroupStep(
                        agent_name=judge_name,
                        step_type="debate_judge_error",
                        timestamp=step_t0,
                        duration_ms=duration_ms,
                        error=str(exc),
                    )
                )
                if self._callbacks.on_agent_error:
                    self._callbacks.on_agent_error(judge_name, exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return DebateResult(
            agent_results=agent_results,
            aggregate_usage=_aggregate_usage(*usage_summaries),
            shared_state=dict(self._state),
            elapsed_ms=elapsed_ms,
            timeline=timeline,
            errors=errors,
            success=len(errors) == 0,
            transcript=transcript,
            rounds_completed=rounds_completed,
            judge_verdict=judge_verdict,
        )


class AsyncDebateGroup:
    """Async version of :class:`DebateGroup`.

    See :class:`DebateGroup` for full documentation.
    """

    def __init__(
        self,
        agents: list[Any],
        config: DebateConfig | None = None,
        *,
        state: dict[str, Any] | None = None,
        callbacks: GroupCallbacks | None = None,
    ) -> None:
        self._agents = [(a, _agent_name(a, i)) for i, a in enumerate(agents)]
        self._config = config or DebateConfig()
        self._state: dict[str, Any] = dict(state) if state else {}
        self._callbacks = callbacks or GroupCallbacks()
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def _build_prompt(
        self,
        topic: str,
        agent_name: str,
        transcript: list[DebateEntry],
        round_num: int,
    ) -> str:
        parts: list[str] = [f"Topic: {topic}"]

        if self._config.show_position_in_prompt and self._config.positions:
            position = self._config.positions.get(agent_name)
            if position:
                parts.append(
                    f"\nYour position: {position}. "
                    "Argue this position and respond to previous arguments."
                )

        if transcript:
            parts.append("\n--- Previous arguments ---")
            for entry in transcript:
                pos_label = f" [{entry.position}]" if entry.position else ""
                parts.append(f"\n{entry.agent_name}{pos_label} (round {entry.round}):\n{entry.content}")
            parts.append("\n--- End of previous arguments ---")

        parts.append(f"\nRound {round_num + 1}: Present your argument.")
        return "\n".join(parts)

    def _format_transcript(self, transcript: list[DebateEntry]) -> str:
        lines: list[str] = []
        for entry in transcript:
            pos_label = f" [{entry.position}]" if entry.position else ""
            lines.append(f"[Round {entry.round + 1}] {entry.agent_name}{pos_label}:")
            lines.append(entry.content)
            lines.append("")
        return "\n".join(lines)

    async def run(self, topic: str) -> DebateResult:
        """Execute the debate (async)."""
        self._stop_requested = False
        t0 = time.perf_counter()
        transcript: list[DebateEntry] = []
        timeline: list[GroupStep] = []
        agent_results: dict[str, Any] = {}
        errors: list[AgentError] = []
        usage_summaries: list[dict[str, Any]] = []
        rounds_completed = 0

        positions = self._config.positions or {}

        for round_num in range(self._config.rounds):
            if self._stop_requested:
                break

            if self._callbacks.on_round_start:
                self._callbacks.on_round_start(round_num)

            for agent, name in self._agents:
                if self._stop_requested:
                    break

                effective = self._build_prompt(topic, name, transcript, round_num)

                if self._callbacks.on_agent_start:
                    self._callbacks.on_agent_start(name, effective)

                step_t0 = time.perf_counter()
                try:
                    result = await agent.run(effective)
                    duration_ms = (time.perf_counter() - step_t0) * 1000

                    result_key = f"{name}_round{round_num}"
                    agent_results[result_key] = result
                    usage = getattr(result, "run_usage", {})
                    usage_summaries.append(usage)

                    entry = DebateEntry(
                        round=round_num,
                        agent_name=name,
                        position=positions.get(name),
                        content=result.output_text,
                        timestamp=step_t0,
                        usage=usage,
                    )
                    transcript.append(entry)

                    timeline.append(
                        GroupStep(
                            agent_name=name,
                            step_type="debate_argument",
                            timestamp=step_t0,
                            duration_ms=duration_ms,
                            usage_delta=usage,
                        )
                    )

                    if self._callbacks.on_agent_complete:
                        self._callbacks.on_agent_complete(name, result)

                except Exception as exc:
                    duration_ms = (time.perf_counter() - step_t0) * 1000
                    err = AgentError(agent_name=name, error=exc)
                    errors.append(err)
                    timeline.append(
                        GroupStep(
                            agent_name=name,
                            step_type="debate_error",
                            timestamp=step_t0,
                            duration_ms=duration_ms,
                            error=str(exc),
                        )
                    )
                    if self._callbacks.on_agent_error:
                        self._callbacks.on_agent_error(name, exc)

            rounds_completed = round_num + 1

            if self._callbacks.on_round_complete:
                self._callbacks.on_round_complete(round_num)

        # Judge phase
        judge_verdict: str | None = None
        if self._config.judge and transcript and not self._stop_requested:
            judge = self._config.judge
            judge_name = getattr(judge, "name", "judge") or "judge"
            formatted = self._format_transcript(transcript)
            judge_prompt = (
                self._config.judge_prompt_template
                .replace("{topic}", topic)
                .replace("{transcript}", formatted)
            )

            if self._callbacks.on_agent_start:
                self._callbacks.on_agent_start(judge_name, judge_prompt)

            step_t0 = time.perf_counter()
            try:
                judge_result = await judge.run(judge_prompt)
                duration_ms = (time.perf_counter() - step_t0) * 1000
                judge_verdict = judge_result.output_text
                agent_results[judge_name] = judge_result
                usage = getattr(judge_result, "run_usage", {})
                usage_summaries.append(usage)

                timeline.append(
                    GroupStep(
                        agent_name=judge_name,
                        step_type="debate_judge",
                        timestamp=step_t0,
                        duration_ms=duration_ms,
                        usage_delta=usage,
                    )
                )

                if self._callbacks.on_agent_complete:
                    self._callbacks.on_agent_complete(judge_name, judge_result)

            except Exception as exc:
                duration_ms = (time.perf_counter() - step_t0) * 1000
                errors.append(AgentError(agent_name=judge_name, error=exc))
                timeline.append(
                    GroupStep(
                        agent_name=judge_name,
                        step_type="debate_judge_error",
                        timestamp=step_t0,
                        duration_ms=duration_ms,
                        error=str(exc),
                    )
                )
                if self._callbacks.on_agent_error:
                    self._callbacks.on_agent_error(judge_name, exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return DebateResult(
            agent_results=agent_results,
            aggregate_usage=_aggregate_usage(*usage_summaries),
            shared_state=dict(self._state),
            elapsed_ms=elapsed_ms,
            timeline=timeline,
            errors=errors,
            success=len(errors) == 0,
            transcript=transcript,
            rounds_completed=rounds_completed,
            judge_verdict=judge_verdict,
        )
