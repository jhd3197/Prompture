"""Async multi-agent group coordination.

Provides :class:`ParallelGroup`, :class:`AsyncSequentialGroup`,
:class:`AsyncLoopGroup`, and :class:`AsyncRouterAgent` for composing
multiple async agents into deterministic workflows.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections.abc import Callable
from typing import Any

from ..agents.types import AgentResult, AgentState
from .groups import _agent_name, _inject_state, _normalise_agents
from .types import (
    AgentError,
    ErrorPolicy,
    GroupCallbacks,
    GroupResult,
    GroupStep,
    _aggregate_usage,
)

logger = logging.getLogger("prompture.async_groups")


# ------------------------------------------------------------------
# ParallelGroup
# ------------------------------------------------------------------


class ParallelGroup:
    """Execute agents concurrently and collect results.

    Agents read from a frozen snapshot of the shared state taken at
    the start of the run. Output key writes are applied after all
    agents complete, in agent index order.

    Args:
        agents: List of async agents or ``(agent, prompt_template)`` tuples.
        state: Initial shared state dict.
        error_policy: How to handle agent failures.
        timeout_ms: Per-agent timeout in milliseconds.
        callbacks: Observability hooks.
        max_total_cost: Budget cap in USD.
    """

    def __init__(
        self,
        agents: list[Any],
        *,
        state: dict[str, Any] | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.fail_fast,
        timeout_ms: float | None = None,
        callbacks: GroupCallbacks | None = None,
        max_total_cost: float | None = None,
    ) -> None:
        self._agents = _normalise_agents(agents)
        self._state: dict[str, Any] = dict(state) if state else {}
        self._error_policy = error_policy
        self._timeout_ms = timeout_ms
        self._callbacks = callbacks or GroupCallbacks()
        self._max_total_cost = max_total_cost
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

    async def run_async(self, prompt: str = "") -> GroupResult:
        """Execute all agents concurrently."""
        self._stop_requested = False
        t0 = time.perf_counter()

        # Frozen state snapshot for all agents (deep copy to isolate nested mutables)
        frozen_state = copy.deepcopy(self._state)

        async def _run_one(
            idx: int, agent: Any, custom_prompt: str | None
        ) -> tuple[int, str, AgentResult | None, AgentError | None, GroupStep]:
            name = _agent_name(agent, idx)

            if custom_prompt is not None:
                effective = _inject_state(custom_prompt, frozen_state)
            elif prompt:
                effective = _inject_state(prompt, frozen_state)
            else:
                effective = ""

            if self._callbacks.on_agent_start:
                self._callbacks.on_agent_start(name, effective)

            step_t0 = time.perf_counter()
            try:
                coro = agent.run(effective)
                if self._timeout_ms is not None:
                    result = await asyncio.wait_for(coro, timeout=self._timeout_ms / 1000)
                else:
                    result = await coro

                duration_ms = (time.perf_counter() - step_t0) * 1000
                step = GroupStep(
                    agent_name=name,
                    step_type="agent_run",
                    timestamp=step_t0,
                    duration_ms=duration_ms,
                    usage_delta=getattr(result, "run_usage", {}),
                )
                if self._callbacks.on_agent_complete:
                    self._callbacks.on_agent_complete(name, result)
                return idx, name, result, None, step

            except Exception as exc:
                duration_ms = (time.perf_counter() - step_t0) * 1000
                err = AgentError(
                    agent_name=name,
                    error=exc,
                    output_key=getattr(agent, "output_key", None),
                )
                step = GroupStep(
                    agent_name=name,
                    step_type="agent_error",
                    timestamp=step_t0,
                    duration_ms=duration_ms,
                    error=str(exc),
                )
                if self._callbacks.on_agent_error:
                    self._callbacks.on_agent_error(name, exc)
                return idx, name, None, err, step

        # Launch all agents concurrently
        tasks = [_run_one(idx, agent, custom_prompt) for idx, (agent, custom_prompt) in enumerate(self._agents)]
        completed = await asyncio.gather(*tasks, return_exceptions=False)

        # Sort by original index to maintain deterministic ordering
        completed_sorted = sorted(completed, key=lambda x: x[0])

        agent_results: dict[str, Any] = {}
        errors: list[AgentError] = []
        timeline: list[GroupStep] = []
        usage_summaries: list[dict[str, Any]] = []

        for idx, name, result, err, step in completed_sorted:
            timeline.append(step)
            if err is not None:
                errors.append(err)
            elif result is not None:
                agent_results[name] = result
                usage_summaries.append(getattr(result, "run_usage", {}))

                # Apply output_key writes in order
                agent_obj = self._agents[idx][0]
                output_key = getattr(agent_obj, "output_key", None)
                if output_key:
                    self._state[output_key] = result.output_text
                    if self._callbacks.on_state_update:
                        self._callbacks.on_state_update(output_key, result.output_text)

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

    def run(self, prompt: str = "") -> GroupResult:
        """Sync wrapper around :meth:`run_async`."""
        return asyncio.run(self.run_async(prompt))


# ------------------------------------------------------------------
# AsyncSequentialGroup
# ------------------------------------------------------------------


class AsyncSequentialGroup:
    """Async version of :class:`~prompture.groups.SequentialGroup`.

    Args:
        agents: List of async agents or ``(agent, prompt_template)`` tuples.
        state: Initial shared state dict.
        error_policy: How to handle agent failures.
        max_total_turns: Limit on total agent runs.
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

    async def run(self, prompt: str = "") -> GroupResult:
        """Execute all agents in sequence (async)."""
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

            if custom_prompt is not None:
                effective = _inject_state(custom_prompt, self._state)
            elif prompt:
                effective = _inject_state(prompt, self._state)
            else:
                effective = ""

            if self._max_total_cost is not None:
                total_so_far = sum(s.get("total_cost", 0.0) for s in usage_summaries)
                if total_so_far >= self._max_total_cost:
                    break

            if self._max_total_turns is not None and turns >= self._max_total_turns:
                break

            if self._callbacks.on_agent_start:
                self._callbacks.on_agent_start(name, effective)

            step_t0 = time.perf_counter()
            try:
                result = await agent.run(effective)
                duration_ms = (time.perf_counter() - step_t0) * 1000
                turns += 1

                agent_results[name] = result
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
# AsyncLoopGroup
# ------------------------------------------------------------------


_LOOP_GROUP_MAX_ITERATIONS_CEILING = 10_000


class AsyncLoopGroup:
    """Async version of :class:`~prompture.groups.LoopGroup`.

    Args:
        agents: List of async agents or ``(agent, prompt_template)`` tuples.
        exit_condition: Callable ``(state, iteration) -> bool``.
        max_iterations: Hard cap on loop iterations (capped at 10 000).
        state: Initial shared state dict.
        error_policy: How to handle agent failures.
        callbacks: Observability hooks.
        max_total_cost: Budget cap in USD.  When cumulative cost exceeds
            this value the loop stops.
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
        max_total_cost: float | None = None,
    ) -> None:
        self._agents = _normalise_agents(agents)
        self._exit_condition = exit_condition
        self._max_iterations = min(max_iterations, _LOOP_GROUP_MAX_ITERATIONS_CEILING)
        self._state: dict[str, Any] = dict(state) if state else {}
        self._error_policy = error_policy
        self._callbacks = callbacks or GroupCallbacks()
        self._max_total_cost = max_total_cost
        self._stop_requested = False

    def stop(self) -> None:
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

    async def run(self, prompt: str = "") -> GroupResult:
        """Execute the loop (async)."""
        self._stop_requested = False
        t0 = time.perf_counter()
        timeline: list[GroupStep] = []
        agent_results: dict[str, Any] = {}
        errors: list[AgentError] = []
        usage_summaries: list[dict[str, Any]] = []

        for iteration in range(self._max_iterations):
            if self._stop_requested:
                break

            try:
                if self._exit_condition(self._state, iteration):
                    break
            except Exception:
                logger.warning("Exit condition raised; stopping loop", exc_info=True)
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
                    result = await agent.run(effective)
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

            if errors and self._error_policy == ErrorPolicy.fail_fast:
                break

            # Check budget after each iteration
            if self._max_total_cost is not None:
                cumulative_cost = sum(s.get("total_cost", s.get("cost", 0.0)) for s in usage_summaries)
                if cumulative_cost >= self._max_total_cost:
                    logger.debug("Budget exceeded (%.4f >= %.4f), stopping loop", cumulative_cost, self._max_total_cost)
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
# AsyncRouterAgent
# ------------------------------------------------------------------

_DEFAULT_ROUTING_PROMPT = """Given these specialists:
{agent_list}

Which should handle this? Reply with ONLY the name.

Request: {prompt}"""


class AsyncRouterAgent:
    """Async LLM-driven router that delegates to the best-matching agent.

    Args:
        model: Model string for the routing LLM call.
        agents: List of async agents to route between.
        routing_prompt: Custom prompt template.
        fallback: Agent to use when routing fails.
        driver: Pre-built async driver instance.
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

    async def run(self, prompt: str, *, deps: Any = None) -> AgentResult:
        """Route the prompt to the best agent (async)."""
        from ..agents.async_conversation import AsyncConversation

        agent_lines = []
        for name, agent in self._agents.items():
            desc = getattr(agent, "description", "") or ""
            agent_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        agent_list = "\n".join(agent_lines)

        routing_text = self._routing_prompt.replace("{agent_list}", agent_list).replace("{prompt}", prompt)

        kwargs: dict[str, Any] = {}
        if self._driver is not None:
            kwargs["driver"] = self._driver
        else:
            kwargs["model_name"] = self._model

        conv = AsyncConversation(**kwargs)
        route_response = await conv.ask(routing_text)

        selected = self._fuzzy_match(route_response.strip())

        if selected is not None:
            return await selected.run(prompt, deps=deps) if deps is not None else await selected.run(prompt)  # type: ignore[no-any-return]
        elif self._fallback is not None:
            return await self._fallback.run(prompt, deps=deps) if deps is not None else await self._fallback.run(prompt)  # type: ignore[no-any-return]
        else:
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

        for name, agent in self._agents.items():
            if name.lower() == response_lower:
                return agent

        for name, agent in self._agents.items():
            if name.lower() in response_lower:
                return agent

        response_words = set(response_lower.split())
        for name, agent in self._agents.items():
            name_words = set(name.lower().replace("_", " ").split())
            if name_words & response_words:
                return agent

        return None
