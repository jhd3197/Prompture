"""High-level deep-research / agentic harness on top of :class:`Agent`.

:class:`DeepAgent` extends :class:`Agent` with four built-in capabilities:

- **Planning** — a ``write_todos`` tool that externalises multi-step plans.
- **Virtual filesystem** — ``read_file``/``write_file``/``edit_file``/
  ``ls``/``glob``/``grep`` tools backed by an in-memory dict on
  :class:`DeepAgentState`.
- **Sub-agents** — a ``task`` tool that dispatches isolated subproblems
  to specialist :class:`Agent` instances configured via
  :class:`SubAgentSpec`.
- **Summarization** — a hook on the conversation that compresses old
  history into a single summary message when context grows large.

Each capability can be toggled independently. State is shared across
all built-in tools via a single :class:`DeepAgentState`, which is
snapshotted on the :class:`DeepAgentResult`.

Quick start::

    from prompture import create_deep_agent

    agent = create_deep_agent(
        model="openai/gpt-4o",
        tools=[my_search_tool, my_fetch_tool],
    )
    result = agent.run("Research the EU AI Act's deadlines for foundation models.")
    print(result.final_response)
    print(result.todos)
    print(result.files)
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Generator
from typing import Any

from pydantic import BaseModel

from ..drivers.base import Driver
from ..infra.budget import BudgetPolicy
from ..infra.callbacks import DriverCallbacks
from .agent import Agent, AgentIterator, LiveAgentResult, StreamedAgentResult
from .deep.planner import make_write_todos_tool
from .deep.subagents import make_task_tool
from .deep.summarizer import SummarizationMiddleware
from .deep.vfs import VFS_TOOL_NAMES, make_vfs_tools
from .deep_prompts import assemble_system_prompt, format_subagent_section
from .deep_state import DeepAgentResult, DeepAgentState, SubAgentSpec, Todo
from .persona import Persona
from .tools_schema import ToolDefinition, ToolRegistry
from .types import AgentCallbacks, AgentResult

logger = logging.getLogger("prompture.agents.deep_agent")

_RESERVED_TOOL_NAMES = {"write_todos", "task", *VFS_TOOL_NAMES}


class DeepAgent(Agent):
    """An :class:`Agent` with planning, VFS, sub-agents, and summarization."""

    def __init__(
        self,
        model: str = "",
        *,
        driver: Driver | None = None,
        tools: list[ToolDefinition] | list[Callable[..., Any]] | ToolRegistry | None = None,
        subagents: list[SubAgentSpec] | None = None,
        persona: Persona | str | None = None,
        system_prompt: str | Persona | None = None,
        enable_planning: bool = True,
        enable_vfs: bool = True,
        enable_summarization: bool = True,
        summarize_at_tokens: int = 80_000,
        summarize_keep_last_n: int = 6,
        summarizer_model: str | Driver | None = None,
        initial_files: dict[str, str] | None = None,
        max_iterations: int = 50,
        max_tool_result_length: int | None = 10_000,
        budget_policy: BudgetPolicy | str | None = None,
        max_cost: float | None = None,
        max_tokens: int | None = None,
        callbacks: DriverCallbacks | None = None,
        agent_callbacks: AgentCallbacks | None = None,
        options: dict[str, Any] | None = None,
        output_type: type[BaseModel] | None = None,
        name: str = "",
        description: str = "",
    ) -> None:
        # Resolve user tools to a flat list of ToolDefinition objects so
        # we can pass them to the sub-agent factory and detect collisions.
        self._user_tools: list[ToolDefinition] = _normalise_user_tools(tools)
        self._subagents: list[SubAgentSpec] = list(subagents or [])

        # Shared mutable state for built-in tools.
        self.deep_state: DeepAgentState = DeepAgentState(files=dict(initial_files or {}))

        # Track which capabilities are enabled for later introspection.
        self._enable_planning = bool(enable_planning)
        self._enable_vfs = bool(enable_vfs)
        self._enable_summarization = bool(enable_summarization)

        # Detect tool-name collisions with built-ins.
        builtin_names = self._enabled_builtin_names()
        collisions = {t.name for t in self._user_tools} & builtin_names
        if collisions:
            warnings.warn(
                f"User-supplied tool name(s) shadow DeepAgent built-ins: "
                f"{sorted(collisions)}. Built-ins take precedence.",
                stacklevel=2,
            )
            self._user_tools = [t for t in self._user_tools if t.name not in collisions]

        # Build the final tool list: user tools + built-ins.
        # Wrap as a ToolRegistry so Agent treats them as pre-built
        # ToolDefinitions (Agent's tool list arg only accepts callables).
        all_defs = list(self._user_tools) + self._build_builtin_tools(
            model_for_subagents=model,
            budget_policy=budget_policy,
            callbacks=callbacks,
            max_tool_result_length=max_tool_result_length,
            driver_for_subagents=driver,
        )
        tool_registry = ToolRegistry()
        for td in all_defs:
            tool_registry.add(td)

        # Compose the system prompt.
        persona_text = _resolve_persona_text(persona, system_prompt)
        subagent_section = format_subagent_section(self._subagents) if self._subagents else None
        composed_prompt = assemble_system_prompt(
            persona_text=persona_text,
            enable_planning=self._enable_planning,
            enable_vfs=self._enable_vfs,
            subagent_section=subagent_section,
        )

        # Initialise the underlying Agent.
        super().__init__(
            model=model,
            driver=driver,
            tools=tool_registry,
            system_prompt=composed_prompt,
            output_type=output_type,
            max_iterations=max_iterations,
            max_cost=max_cost,
            max_tokens=max_tokens,
            budget_policy=budget_policy,
            options=options,
            agent_callbacks=agent_callbacks,
            name=name,
            description=description,
            persistent_conversation=False,
            max_tool_result_length=max_tool_result_length,
        )

        # Install the summariser as the before_turn hook. Agent._build_conversation
        # reads this attribute and passes it through to the Conversation.
        if self._enable_summarization:
            summariser_driver = self._resolve_summariser(summarizer_model, model, driver)
            self._summarizer: SummarizationMiddleware | None = SummarizationMiddleware(
                threshold_tokens=summarize_at_tokens,
                keep_last_n=summarize_keep_last_n,
                state=self.deep_state,
                summariser=summariser_driver,
            )
            self._before_turn_hook: Callable[..., Any] | None = self._summarizer
        else:
            self._summarizer = None
            self._before_turn_hook = None

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def todos(self) -> list[Todo]:
        """Current todo list. Mutated by the ``write_todos`` tool."""
        return self.deep_state.todos

    @property
    def files(self) -> dict[str, str]:
        """Virtual filesystem contents. Mutated by VFS tools."""
        return self.deep_state.files

    def reset(self) -> None:
        """Clear deep state AND any persistent conversation history."""
        self.deep_state.reset()
        self.clear_history()

    def run(self, prompt: str, *, deps: Any = None) -> DeepAgentResult:  # type: ignore[override]
        """Run the agent and return a :class:`DeepAgentResult`.

        The result extends :class:`AgentResult` with a snapshot of the
        :class:`DeepAgentState` (todos, files, sub-agent calls, summaries).
        """
        base = super().run(prompt, deps=deps)
        return _upgrade_result(base, self.deep_state)

    def _deep_gen(self, gen: Generator[Any, Any, AgentResult]) -> Generator[Any, Any, DeepAgentResult]:
        """Pass *gen* through, upgrading only its final value.

        ``yield from`` forwards sends and throws unchanged, so the streaming
        behaviour is identical — just the returned result carries the deep
        state snapshot.
        """
        base = yield from gen
        return _upgrade_result(base, self.deep_state)

    def iter(self, prompt: str, *, deps: Any = None) -> AgentIterator:
        """Like :meth:`Agent.iter`, but ``.result`` is a :class:`DeepAgentResult`."""
        return AgentIterator(self._deep_gen(self._execute_iter(prompt, deps)))

    def run_stream(self, prompt: str, *, deps: Any = None) -> StreamedAgentResult:
        """Like :meth:`Agent.run_stream`, but ``.result`` is a :class:`DeepAgentResult`."""
        return StreamedAgentResult(self._deep_gen(self._execute_stream(prompt, deps)))

    def run_live(self, prompt: str, *, deps: Any = None) -> LiveAgentResult:
        """Like :meth:`Agent.run_live`, but ``.result`` is a :class:`DeepAgentResult`."""
        return LiveAgentResult(self._deep_gen(self._execute_live(prompt, deps)))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enabled_builtin_names(self) -> set[str]:
        names: set[str] = set()
        if self._enable_planning:
            names.add("write_todos")
        if self._enable_vfs:
            names |= VFS_TOOL_NAMES
        if self._subagents:
            names.add("task")
        return names

    def _build_builtin_tools(
        self,
        model_for_subagents: str,
        budget_policy: BudgetPolicy | str | None,
        callbacks: DriverCallbacks | None,
        max_tool_result_length: int | None,
        driver_for_subagents: Driver | None = None,
    ) -> list[ToolDefinition]:
        from ..infra.budget import resolve_budget_policy

        tools: list[ToolDefinition] = []
        if self._enable_planning:
            tools.append(make_write_todos_tool(self.deep_state))
        if self._enable_vfs:
            tools.extend(make_vfs_tools(self.deep_state))
        if self._subagents:
            resolved_budget = resolve_budget_policy(budget_policy)
            tools.append(
                make_task_tool(
                    specs=self._subagents,
                    state=self.deep_state,
                    parent_model=model_for_subagents,
                    parent_user_tools=self._user_tools,
                    parent_budget_policy=resolved_budget,
                    parent_callbacks=callbacks,
                    parent_max_tool_result_length=max_tool_result_length,
                    parent_driver=driver_for_subagents,
                )
            )
        return tools

    @staticmethod
    def _resolve_summariser(
        summarizer_model: str | Driver | None,
        agent_model: str,
        agent_driver: Driver | None,
    ) -> str | Driver:
        """Pick the driver/model used for summarization calls."""
        if isinstance(summarizer_model, Driver):
            return summarizer_model
        if isinstance(summarizer_model, str) and summarizer_model.strip():
            return summarizer_model
        if agent_driver is not None:
            return agent_driver
        return agent_model


# ----------------------------------------------------------------------
# Convenience factory + helpers
# ----------------------------------------------------------------------


def create_deep_agent(
    model: str = "",
    *,
    tools: list[ToolDefinition] | list[Callable[..., Any]] | ToolRegistry | None = None,
    subagents: list[SubAgentSpec] | None = None,
    persona: Persona | str | None = None,
    system_prompt: str | Persona | None = None,
    enable_planning: bool = True,
    enable_vfs: bool = True,
    enable_summarization: bool = True,
    summarize_at_tokens: int = 80_000,
    summarize_keep_last_n: int = 6,
    summarizer_model: str | Driver | None = None,
    initial_files: dict[str, str] | None = None,
    max_iterations: int = 50,
    max_tool_result_length: int | None = 10_000,
    budget_policy: BudgetPolicy | str | None = None,
    max_cost: float | None = None,
    max_tokens: int | None = None,
    callbacks: DriverCallbacks | None = None,
    agent_callbacks: AgentCallbacks | None = None,
    driver: Driver | None = None,
    options: dict[str, Any] | None = None,
    output_type: type[BaseModel] | None = None,
    name: str = "",
    description: str = "",
) -> DeepAgent:
    """Convenience factory mirroring :func:`deepagents.create_deep_agent`."""
    return DeepAgent(
        model=model,
        driver=driver,
        tools=tools,
        subagents=subagents,
        persona=persona,
        system_prompt=system_prompt,
        enable_planning=enable_planning,
        enable_vfs=enable_vfs,
        enable_summarization=enable_summarization,
        summarize_at_tokens=summarize_at_tokens,
        summarize_keep_last_n=summarize_keep_last_n,
        summarizer_model=summarizer_model,
        initial_files=initial_files,
        max_iterations=max_iterations,
        max_tool_result_length=max_tool_result_length,
        budget_policy=budget_policy,
        max_cost=max_cost,
        max_tokens=max_tokens,
        callbacks=callbacks,
        agent_callbacks=agent_callbacks,
        options=options,
        output_type=output_type,
        name=name,
        description=description,
    )


def _normalise_user_tools(
    tools: list[ToolDefinition] | list[Callable[..., Any]] | ToolRegistry | None,
) -> list[ToolDefinition]:
    """Coerce the user's tools argument into a flat ``list[ToolDefinition]``."""
    if tools is None:
        return []
    if isinstance(tools, ToolRegistry):
        return list(tools.definitions)
    out: list[ToolDefinition] = []
    for item in tools:
        if isinstance(item, ToolDefinition):
            out.append(item)
        elif callable(item):
            from .tools_schema import tool_from_function

            out.append(tool_from_function(item))
        else:
            raise TypeError(f"Unsupported tool type: {type(item).__name__}")
    return out


def _resolve_persona_text(
    persona: Persona | str | None,
    system_prompt: str | Persona | None,
) -> str | None:
    """Pick which textual system prompt to use (persona | system_prompt)."""
    if persona is not None and system_prompt is not None:
        raise ValueError("Provide either persona or system_prompt, not both.")
    if persona is not None:
        if isinstance(persona, Persona):
            return persona.render()
        if isinstance(persona, str):
            from .persona import get_persona

            resolved = get_persona(persona)
            if resolved is None:
                raise ValueError(f"Persona {persona!r} not in registry")
            return resolved.render()
    if system_prompt is not None:
        if isinstance(system_prompt, Persona):
            return system_prompt.render()
        return str(system_prompt)
    return None


def _upgrade_result(base: AgentResult, state: DeepAgentState) -> DeepAgentResult:
    """Wrap an :class:`AgentResult` in a :class:`DeepAgentResult` snapshot."""
    return DeepAgentResult(
        output=base.output,
        output_text=base.output_text,
        messages=base.messages,
        usage=base.usage,
        steps=base.steps,
        all_tool_calls=base.all_tool_calls,
        state=base.state,
        run_usage=base.run_usage,
        deep_state=state,
    )
