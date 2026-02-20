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

                # Step condition check (waterfall mode)
                if self._step_conditions is not None and idx < len(self._step_conditions):
                    condition = self._step_conditions[idx]
                    should_continue = condition(result.output_text, self._state)
                    if not should_continue:
                        self._state["_escalation_stopped_at"] = name
                        # Fire on_step_skipped for remaining agents
                        if self._callbacks.on_step_skipped:
                            for skip_idx in range(idx + 1, len(self._agents)):
                                skip_agent, _ = self._agents[skip_idx]
                                skip_name = _agent_name(skip_agent, skip_idx)
                                self._callbacks.on_step_skipped(
                                    skip_name, f"step condition false after {name}"
                                )
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


_LOOP_GROUP_MAX_ITERATIONS_CEILING = 10_000


class LoopGroup:
    """Repeat a sequence of agents until an exit condition is met.

    Args:
        agents: List of agents or ``(agent, prompt_template)`` tuples.
        exit_condition: Callable ``(state, iteration) -> bool``.
            When it returns ``True`` the loop stops.
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

            if self._callbacks.on_round_complete:
                self._callbacks.on_round_complete(iteration)

            # Check if error caused early exit
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
# RouterAgent
# ------------------------------------------------------------------

_DEFAULT_ROUTING_PROMPT = """Given these specialists:
{agent_list}

Which should handle this? Reply with ONLY the name.

Request: {prompt}"""


class RoutingStrategy:
    """Routing strategy constants for :class:`RouterAgent`."""

    llm = "llm"
    keyword = "keyword"
    round_robin = "round_robin"


class RouterAgent:
    """Router that delegates to the best-matching agent.

    Supports multiple routing strategies:

    - **llm** (default): LLM classifies which agent should handle the request.
    - **keyword**: Match keywords in the message to agent keyword lists.
    - **round_robin**: Rotate through agents in order.

    Args:
        model: Model string for the routing LLM call.
        agents: List of agents to route between.
        strategy: Routing strategy (``"llm"``, ``"keyword"``, ``"round_robin"``).
        routing_prompt: Custom prompt template (must include ``{agent_list}``
            and ``{prompt}`` placeholders).
        fallback: Agent to use when routing is ambiguous.
        keywords: Agent name to keyword list mapping for keyword strategy.
        callbacks: Observability hooks (fires ``on_route_decision``).
        driver: Pre-built driver instance for the routing call.
    """

    def __init__(
        self,
        model: str = "",
        *,
        agents: list[Any],
        strategy: str = RoutingStrategy.llm,
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

    def _classify(self, prompt: str) -> tuple[str | None, str, float]:
        """Classify which agent should handle the prompt.

        Returns:
            Tuple of ``(agent_name, reason, confidence)``.
            ``agent_name`` is ``None`` when no agent matched.
        """
        if self._strategy == RoutingStrategy.keyword:
            return self._classify_keyword(prompt)
        elif self._strategy == RoutingStrategy.round_robin:
            return self._classify_round_robin()
        else:
            return self._classify_llm(prompt)

    def _classify_llm(self, prompt: str) -> tuple[str | None, str, float]:
        """LLM-based routing (original behavior)."""
        from ..agents.conversation import Conversation

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

        conv = Conversation(**kwargs)
        route_response = conv.ask(routing_text)
        self._last_conv = conv

        matched_name = self._fuzzy_match_name(route_response.strip())
        self._last_llm_response = route_response.strip()
        if matched_name:
            return matched_name, f"LLM selected: {route_response.strip()}", 0.8
        return None, f"LLM response did not match: {route_response.strip()}", 0.0

    def _classify_keyword(self, prompt: str) -> tuple[str | None, str, float]:
        """Keyword-based routing: match message tokens against keyword lists."""
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
        """Round-robin routing: rotate through agents in order."""
        agent_names = list(self._agents.keys())
        if not agent_names:
            return None, "No agents available", 0.0
        selected = agent_names[self._rr_index % len(agent_names)]
        self._rr_index += 1
        return selected, f"Round-robin index {self._rr_index - 1}", 1.0

    def run(self, prompt: str, *, deps: Any = None) -> AgentResult:
        """Route the prompt to the best agent and return its result."""
        self._last_conv = None
        self._last_llm_response = None
        agent_name, reason, confidence = self._classify(prompt)

        # Fire routing decision callback
        if self._callbacks.on_route_decision:
            self._callbacks.on_route_decision(agent_name or "(none)", reason, confidence)

        if agent_name and agent_name in self._agents:
            selected = self._agents[agent_name]
            return selected.run(prompt, deps=deps) if deps is not None else selected.run(prompt)  # type: ignore[no-any-return]
        elif self._fallback is not None:
            if self._callbacks.on_route_decision and agent_name is None:
                self._callbacks.on_route_decision("(fallback)", "No match, using fallback", 0.0)
            return self._fallback.run(prompt, deps=deps) if deps is not None else self._fallback.run(prompt)  # type: ignore[no-any-return]
        else:
            # Return the raw LLM response (if LLM strategy) or reason string
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
        """Find the best matching agent name in the LLM response. Returns the name string."""
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
