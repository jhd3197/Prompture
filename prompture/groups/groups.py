"""Synchronous multi-agent group coordination.

Provides :class:`SequentialGroup`, :class:`LoopGroup`,
:class:`RouterAgent`, and :class:`GroupAsAgent` for composing
multiple agents into deterministic workflows.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

from ..agents.types import AgentResult, AgentState
from .types import (
    AgentError,
    ErrorPolicy,
    GroupCallbacks,
    GroupResult,
    GroupStep,
    _aggregate_usage,
)

logger = logging.getLogger("prompture.groups")


# ------------------------------------------------------------------
# State injection helper
# ------------------------------------------------------------------


def _inject_state(template: str, state: dict[str, Any]) -> str:
    """Replace ``{key}`` placeholders with state values.

    Unknown keys pass through unchanged so downstream agents can
    still see the literal placeholder.
    """

    def _replacer(m: re.Match[str]) -> str:
        key = m.group(1)
        if key in state:
            return str(state[key])
        return m.group(0)  # leave unchanged

    return re.sub(r"\{(\w+)\}", _replacer, template)


# ------------------------------------------------------------------
# Agent entry normalisation
# ------------------------------------------------------------------

AgentEntry = Any  # Agent | tuple[Agent, str]


def _normalise_agents(agents: list[Any]) -> list[tuple[Any, str | None]]:
    """Convert a mixed list of ``Agent`` or ``(Agent, prompt_template)`` to uniform tuples."""
    result: list[tuple[Any, str | None]] = []
    for item in agents:
        if isinstance(item, tuple):
            result.append((item[0], item[1]))
        else:
            result.append((item, None))
    return result


def _agent_name(agent: Any, index: int) -> str:
    """Determine a display name for an agent."""
    name = getattr(agent, "name", "") or ""
    return name if name else f"agent_{index}"


# ------------------------------------------------------------------
# SequentialGroup
# ------------------------------------------------------------------


class SequentialGroup:
    """Execute agents in sequence, passing state between them.

    Each agent's ``output_key`` (if set) writes its output text
    into the shared state dict, making it available as ``{key}``
    in subsequent agent prompts.

    Args:
        agents: List of agents or ``(agent, prompt_template)`` tuples.
        state: Initial shared state dict.
        error_policy: How to handle agent failures.
        max_total_turns: Limit on total agent runs across the sequence.
        callbacks: Observability hooks.
        max_total_cost: Budget cap in USD.
    """

    def __init__(
        self,
        agents: list[Any],
        *,
        state: dict[str, Any] | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.fail_fast,
        max_total_turns: int | None = None,
        callbacks: GroupCallbacks | None = None,
        max_total_cost: float | None = None,
    ) -> None:
        self._agents = _normalise_agents(agents)
        self._state: dict[str, Any] = dict(state) if state else {}
        self._error_policy = error_policy
        self._max_total_turns = max_total_turns
        self._callbacks = callbacks or GroupCallbacks()
        self._max_total_cost = max_total_cost
        self._stop_requested = False

    def stop(self) -> None:
        """Request graceful shutdown after the current agent finishes."""
        self._stop_requested = True

    @property
    def shared_state(self) -> dict[str, Any]:
        """Return a copy of the current shared execution state."""
        return dict(self._state)

    def inject_state(self, state: dict[str, Any], *, recursive: bool = False) -> None:
        """Merge external key-value pairs into this group's shared state.

        Existing keys are NOT overwritten (uses setdefault semantics).

        Args:
            state: Key-value pairs to inject.
            recursive: If True, also inject into nested sub-groups.
        """
        for k, v in state.items():
            self._state.setdefault(k, v)
        if recursive:
            for agent, _ in self._agents:
                if hasattr(agent, "inject_state"):
                    agent.inject_state(state, recursive=True)

    def save(self, path: str) -> None:
        """Run and save result to file. Convenience wrapper."""
        result = self.run()
        result.save(path)

    def run(self, prompt: str = "") -> GroupResult:
        """Execute all agents in order."""
        self._stop_requested = False
        t0 = time.perf_counter()
        timeline: list[GroupStep] = []
        agent_results: dict[str, Any] = {}
        errors: list[AgentError] = []
        usage_summaries: list[dict[str, Any]] = []
        turns = 0

        for idx, (agent, custom_prompt) in enumerate(self._agents):
            if self._stop_requested:
                break

            name = _agent_name(agent, idx)

            # Build effective prompt
            if custom_prompt is not None:
                effective = _inject_state(custom_prompt, self._state)
            elif prompt:
                effective = _inject_state(prompt, self._state)
            else:
                effective = ""

            # Check budget
            if self._max_total_cost is not None:
                total_so_far = sum(s.get("total_cost", 0.0) for s in usage_summaries)
                if total_so_far >= self._max_total_cost:
                    logger.debug("Budget exceeded, stopping group")
                    break

            # Check max turns
            if self._max_total_turns is not None and turns >= self._max_total_turns:
                logger.debug("Max total turns reached")
                break

            # Fire callback
            if self._callbacks.on_agent_start:
                self._callbacks.on_agent_start(name, effective)

            step_t0 = time.perf_counter()
            try:
                result = agent.run(effective)
                duration_ms = (time.perf_counter() - step_t0) * 1000
                turns += 1

                agent_results[name] = result
                usage = getattr(result, "run_usage", {})
                usage_summaries.append(usage)

                # Write to shared state
                output_key = getattr(agent, "output_key", None)
                if output_key:
                    self._state[output_key] = result.output_text
                    if self._callbacks.on_state_update:
                        self._callbacks.on_state_update(output_key, result.output_text)

                timeline.append(
                    GroupStep(
                        agent_name=name,
                        step_type="agent_run",
                        timestamp=step_t0,
                        duration_ms=duration_ms,
                        usage_delta=usage,
                    )
                )

                if self._callbacks.on_agent_complete:
                    self._callbacks.on_agent_complete(name, result)

            except Exception as exc:
                duration_ms = (time.perf_counter() - step_t0) * 1000
                turns += 1
                err = AgentError(
                    agent_name=name,
                    error=exc,
                    output_key=getattr(agent, "output_key", None),
                )
                errors.append(err)
                timeline.append(
                    GroupStep(
                        agent_name=name,
                        step_type="agent_error",
                        timestamp=step_t0,
                        duration_ms=duration_ms,
                        error=str(exc),
                    )
                )

                if self._callbacks.on_agent_error:
                    self._callbacks.on_agent_error(name, exc)

                if self._error_policy == ErrorPolicy.raise_on_error:
                    raise
                if self._error_policy == ErrorPolicy.fail_fast:
                    break
                # continue_on_error / retry_failed: continue to next agent

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return GroupResult(
            agent_results=agent_results,
            aggregate_usage=_aggregate_usage(*usage_summaries),
            shared_state=dict(self._state),
            elapsed_ms=elapsed_ms,
            timeline=timeline,
            errors=errors,
            success=len(errors) == 0,
        )


# ------------------------------------------------------------------
# LoopGroup
# ------------------------------------------------------------------


class LoopGroup:
    """Repeat a sequence of agents until an exit condition is met.

    Args:
        agents: List of agents or ``(agent, prompt_template)`` tuples.
        exit_condition: Callable ``(state, iteration) -> bool``.
            When it returns ``True`` the loop stops.
        max_iterations: Hard cap on loop iterations.
        state: Initial shared state dict.
        error_policy: How to handle agent failures.
        callbacks: Observability hooks.
    """

    def __init__(
        self,
        agents: list[Any],
        *,
        exit_condition: Callable[[dict[str, Any], int], bool],
        max_iterations: int = 10,
        state: dict[str, Any] | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.fail_fast,
        callbacks: GroupCallbacks | None = None,
    ) -> None:
        self._agents = _normalise_agents(agents)
        self._exit_condition = exit_condition
        self._max_iterations = max_iterations
        self._state: dict[str, Any] = dict(state) if state else {}
        self._error_policy = error_policy
        self._callbacks = callbacks or GroupCallbacks()
        self._stop_requested = False

    def stop(self) -> None:
        """Request graceful shutdown."""
        self._stop_requested = True

    @property
    def shared_state(self) -> dict[str, Any]:
        """Return a copy of the current shared execution state."""
        return dict(self._state)

    def inject_state(self, state: dict[str, Any], *, recursive: bool = False) -> None:
        """Merge external key-value pairs into this group's shared state.

        Existing keys are NOT overwritten (uses setdefault semantics).

        Args:
            state: Key-value pairs to inject.
            recursive: If True, also inject into nested sub-groups.
        """
        for k, v in state.items():
            self._state.setdefault(k, v)
        if recursive:
            for agent, _ in self._agents:
                if hasattr(agent, "inject_state"):
                    agent.inject_state(state, recursive=True)

    def run(self, prompt: str = "") -> GroupResult:
        """Execute the loop."""
        self._stop_requested = False
        t0 = time.perf_counter()
        timeline: list[GroupStep] = []
        agent_results: dict[str, Any] = {}
        errors: list[AgentError] = []
        usage_summaries: list[dict[str, Any]] = []

        for iteration in range(self._max_iterations):
            if self._stop_requested:
                break
            if self._exit_condition(self._state, iteration):
                break

            for idx, (agent, custom_prompt) in enumerate(self._agents):
                if self._stop_requested:
                    break

                name = _agent_name(agent, idx)
                result_key = f"{name}_iter{iteration}"

                if custom_prompt is not None:
                    effective = _inject_state(custom_prompt, self._state)
                elif prompt:
                    effective = _inject_state(prompt, self._state)
                else:
                    effective = ""

                if self._callbacks.on_agent_start:
                    self._callbacks.on_agent_start(name, effective)

                step_t0 = time.perf_counter()
                try:
                    result = agent.run(effective)
                    duration_ms = (time.perf_counter() - step_t0) * 1000

                    agent_results[result_key] = result
                    usage = getattr(result, "run_usage", {})
                    usage_summaries.append(usage)

                    output_key = getattr(agent, "output_key", None)
                    if output_key:
                        self._state[output_key] = result.output_text
                        if self._callbacks.on_state_update:
                            self._callbacks.on_state_update(output_key, result.output_text)

                    timeline.append(
                        GroupStep(
                            agent_name=name,
                            step_type="agent_run",
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
                        output_key=getattr(agent, "output_key", None),
                    )
                    errors.append(err)
                    timeline.append(
                        GroupStep(
                            agent_name=name,
                            step_type="agent_error",
                            timestamp=step_t0,
                            duration_ms=duration_ms,
                            error=str(exc),
                        )
                    )

                    if self._callbacks.on_agent_error:
                        self._callbacks.on_agent_error(name, exc)

                    if self._error_policy == ErrorPolicy.raise_on_error:
                        raise
                    if self._error_policy == ErrorPolicy.fail_fast:
                        break

            # Check if error caused early exit
            if errors and self._error_policy == ErrorPolicy.fail_fast:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return GroupResult(
            agent_results=agent_results,
            aggregate_usage=_aggregate_usage(*usage_summaries),
            shared_state=dict(self._state),
            elapsed_ms=elapsed_ms,
            timeline=timeline,
            errors=errors,
            success=len(errors) == 0,
        )


# ------------------------------------------------------------------
# RouterAgent
# ------------------------------------------------------------------

_DEFAULT_ROUTING_PROMPT = """Given these specialists:
{agent_list}

Which should handle this? Reply with ONLY the name.

Request: {prompt}"""


class RouterAgent:
    """LLM-driven router that delegates to the best-matching agent.

    Args:
        model: Model string for the routing LLM call.
        agents: List of agents to route between.
        routing_prompt: Custom prompt template (must include ``{agent_list}``
            and ``{prompt}`` placeholders).
        fallback: Agent to use when routing fails.
        driver: Pre-built driver instance for the routing call.
    """

    def __init__(
        self,
        model: str = "",
        *,
        agents: list[Any],
        routing_prompt: str | None = None,
        fallback: Any | None = None,
        driver: Any | None = None,
        name: str = "",
        description: str = "",
        output_key: str | None = None,
    ) -> None:
        self._model = model
        self._driver = driver
        self._agents = {_agent_name(a, i): a for i, a in enumerate(agents)}
        self._routing_prompt = routing_prompt or _DEFAULT_ROUTING_PROMPT
        self._fallback = fallback
        self.name = name
        self.description = description
        self.output_key = output_key

    def run(self, prompt: str, *, deps: Any = None) -> AgentResult:
        """Route the prompt to the best agent and return its result."""
        from ..agents.conversation import Conversation

        # Build agent list for routing prompt
        agent_lines = []
        for name, agent in self._agents.items():
            desc = getattr(agent, "description", "") or ""
            agent_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        agent_list = "\n".join(agent_lines)

        routing_text = self._routing_prompt.replace("{agent_list}", agent_list).replace("{prompt}", prompt)

        # Single LLM call for routing
        kwargs: dict[str, Any] = {}
        if self._driver is not None:
            kwargs["driver"] = self._driver
        else:
            kwargs["model_name"] = self._model

        conv = Conversation(**kwargs)
        route_response = conv.ask(routing_text)

        # Fuzzy match against known agent names
        selected = self._fuzzy_match(route_response.strip())

        if selected is not None:
            return selected.run(prompt, deps=deps) if deps is not None else selected.run(prompt)
        elif self._fallback is not None:
            return self._fallback.run(prompt, deps=deps) if deps is not None else self._fallback.run(prompt)
        else:
            # Return routing response as fallback
            return AgentResult(
                output=route_response,
                output_text=route_response,
                messages=conv.messages,
                usage=conv.usage,
                state=AgentState.idle,
            )

    def _fuzzy_match(self, response: str) -> Any | None:
        """Find the best matching agent name in the LLM response."""
        response_lower = response.lower().strip()

        # Exact match
        for name, agent in self._agents.items():
            if name.lower() == response_lower:
                return agent

        # Substring match
        for name, agent in self._agents.items():
            if name.lower() in response_lower:
                return agent

        # Word-level match
        response_words = set(response_lower.split())
        for name, agent in self._agents.items():
            name_words = set(name.lower().replace("_", " ").split())
            if name_words & response_words:
                return agent

        return None


# ------------------------------------------------------------------
# GroupAsAgent
# ------------------------------------------------------------------


class GroupAsAgent:
    """Adapter that makes a group behave like an Agent for composability.

    Allows nesting groups inside other groups by presenting the same
    ``run(prompt) -> AgentResult`` interface.

    Args:
        group: The group to wrap (SequentialGroup, LoopGroup, etc.).
        name: Agent identity name.
        output_key: Shared state key for writing output.
    """

    def __init__(
        self,
        group: Any,
        *,
        name: str = "",
        output_key: str | None = None,
    ) -> None:
        self._group = group
        self.name = name
        self.output_key = output_key
        self.description = ""

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        """Run the wrapped group and return an AgentResult."""
        group_result = self._group.run(prompt)

        # Use the last agent's output text, or the shared state
        output_text = ""
        if group_result.agent_results:
            last_result = list(group_result.agent_results.values())[-1]
            output_text = getattr(last_result, "output_text", str(last_result))

        return AgentResult(
            output=output_text,
            output_text=output_text,
            messages=[],
            usage=group_result.aggregate_usage,
            state=AgentState.idle,
            run_usage=group_result.aggregate_usage,
        )

    def stop(self) -> None:
        """Propagate stop to the wrapped group."""
        if hasattr(self._group, "stop"):
            self._group.stop()
