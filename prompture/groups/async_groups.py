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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agents.async_conversation import AsyncConversation

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
        step_conditions: Per-step continue conditions for waterfall mode.
            A list of callables ``(output_text, shared_state) -> bool``,
            one per agent (missing entries default to always-continue).
            After each agent runs, its condition is checked. If it returns
            ``False``, the group stops and returns results so far.
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
        step_conditions: list[Callable[[str, dict[str, Any]], bool]] | None = None,
    ) -> None:
        self._agents = _normalise_agents(agents)
        self._state: dict[str, Any] = dict(state) if state else {}
        self._error_policy = error_policy
        self._max_total_turns = max_total_turns
        self._callbacks = callbacks or GroupCallbacks()
        self._max_total_cost = max_total_cost
        self._step_conditions = step_conditions
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

                # Step condition check (waterfall mode)
                if self._step_conditions is not None and idx < len(self._step_conditions):
                    condition = self._step_conditions[idx]
                    should_continue = condition(result.output_text, self._state)
                    if not should_continue:
                        self._state["_escalation_stopped_at"] = name
                        if self._callbacks.on_step_skipped:
                            for skip_idx in range(idx + 1, len(self._agents)):
                                skip_agent, _ = self._agents[skip_idx]
                                skip_name = _agent_name(skip_agent, skip_idx)
                                self._callbacks.on_step_skipped(skip_name, f"step condition false after {name}")
                        break

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

            if self._callbacks.on_round_start:
                self._callbacks.on_round_start(iteration)

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

            if self._callbacks.on_round_complete:
                self._callbacks.on_round_complete(iteration)

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
    """Async router that delegates to the best-matching agent.

    Supports multiple routing strategies (``llm``, ``keyword``, ``round_robin``).
    See :class:`~prompture.groups.RouterAgent` for full documentation.

    Args:
        model: Model string for the routing LLM call.
        agents: List of async agents to route between.
        strategy: Routing strategy.
        routing_prompt: Custom prompt template.
        fallback: Agent to use when routing is ambiguous.
        keywords: Agent name to keyword list mapping for keyword strategy.
        callbacks: Observability hooks (fires ``on_route_decision``).
        driver: Pre-built async driver instance.
    """

    def __init__(
        self,
        model: str = "",
        *,
        agents: list[Any],
        strategy: str = "llm",
        routing_prompt: str | None = None,
        fallback: Any | None = None,
        keywords: dict[str, list[str]] | None = None,
        callbacks: GroupCallbacks | None = None,
        driver: Any | None = None,
        name: str = "",
        description: str = "",
        output_key: str | None = None,
    ) -> None:
        self._model = model
        self._driver = driver
        self._agents = {_agent_name(a, i): a for i, a in enumerate(agents)}
        self._strategy = strategy
        self._routing_prompt = routing_prompt or _DEFAULT_ROUTING_PROMPT
        self._fallback = fallback
        self._keywords = keywords or {}
        self._callbacks = callbacks or GroupCallbacks()
        self.name = name
        self.description = description
        self.output_key = output_key
        self._rr_index = 0
        self._last_conv: AsyncConversation | None = None
        self._last_llm_response: str | None = None

    async def _classify(self, prompt: str) -> tuple[str | None, str, float]:
        """Classify which agent should handle the prompt."""
        if self._strategy == "keyword":
            return self._classify_keyword(prompt)
        elif self._strategy == "round_robin":
            return self._classify_round_robin()
        else:
            return await self._classify_llm(prompt)

    async def _classify_llm(self, prompt: str) -> tuple[str | None, str, float]:
        """LLM-based routing."""
        from ..agents.async_conversation import AsyncConversation

        agent_lines = []
        for aname, agent in self._agents.items():
            desc = getattr(agent, "description", "") or ""
            agent_lines.append(f"- {aname}: {desc}" if desc else f"- {aname}")
        agent_list = "\n".join(agent_lines)

        routing_text = self._routing_prompt.replace("{agent_list}", agent_list).replace("{prompt}", prompt)

        kwargs: dict[str, Any] = {}
        if self._driver is not None:
            kwargs["driver"] = self._driver
        else:
            kwargs["model_name"] = self._model

        conv = AsyncConversation(**kwargs)
        route_response = await conv.ask(routing_text)
        self._last_conv = conv

        matched_name = self._fuzzy_match_name(route_response.strip())
        self._last_llm_response = route_response.strip()
        if matched_name:
            return matched_name, f"LLM selected: {route_response.strip()}", 0.8
        return None, f"LLM response did not match: {route_response.strip()}", 0.0

    def _classify_keyword(self, prompt: str) -> tuple[str | None, str, float]:
        """Keyword-based routing."""
        prompt_lower = prompt.lower()
        best_name: str | None = None
        best_count = 0

        for aname in self._agents:
            kw_list = self._keywords.get(aname, [])
            count = sum(1 for kw in kw_list if kw.lower() in prompt_lower)
            if count > best_count:
                best_count = count
                best_name = aname

        if best_name and best_count > 0:
            confidence = min(best_count / max(len(self._keywords.get(best_name, [])), 1), 1.0)
            return best_name, f"Matched {best_count} keyword(s)", confidence
        return None, "No keywords matched", 0.0

    def _classify_round_robin(self) -> tuple[str | None, str, float]:
        """Round-robin routing."""
        agent_names = list(self._agents.keys())
        if not agent_names:
            return None, "No agents available", 0.0
        selected = agent_names[self._rr_index % len(agent_names)]
        self._rr_index += 1
        return selected, f"Round-robin index {self._rr_index - 1}", 1.0

    async def run(self, prompt: str, *, deps: Any = None) -> AgentResult:
        """Route the prompt to the best agent (async)."""
        self._last_conv = None
        self._last_llm_response = None
        agent_name, reason, confidence = await self._classify(prompt)

        if self._callbacks.on_route_decision:
            self._callbacks.on_route_decision(agent_name or "(none)", reason, confidence)

        if agent_name and agent_name in self._agents:
            selected = self._agents[agent_name]
            return await selected.run(prompt, deps=deps) if deps is not None else await selected.run(prompt)  # type: ignore[no-any-return]
        elif self._fallback is not None:
            if self._callbacks.on_route_decision and agent_name is None:
                self._callbacks.on_route_decision("(fallback)", "No match, using fallback", 0.0)
            return await self._fallback.run(prompt, deps=deps) if deps is not None else await self._fallback.run(prompt)  # type: ignore[no-any-return]
        else:
            text = self._last_llm_response if self._last_llm_response is not None else reason
            conv = getattr(self, "_last_conv", None)
            return AgentResult(
                output=text,
                output_text=text,
                messages=conv.messages if conv else [],
                usage=conv.usage if conv else {},
                state=AgentState.idle,
            )

    def _fuzzy_match_name(self, response: str) -> str | None:
        """Find the best matching agent name in the LLM response."""
        response_lower = response.lower().strip()

        for aname in self._agents:
            if aname.lower() == response_lower:
                return aname

        for aname in self._agents:
            if aname.lower() in response_lower:
                return aname

        response_words = set(response_lower.split())
        for aname in self._agents:
            name_words = set(aname.lower().replace("_", " ").split())
            if name_words & response_words:
                return aname

        return None

    def _fuzzy_match(self, response: str) -> Any | None:
        """Find the best matching agent in the LLM response (legacy)."""
        matched = self._fuzzy_match_name(response)
        return self._agents[matched] if matched else None
