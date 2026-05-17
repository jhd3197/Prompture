"""Async counterpart of :class:`DeepAgent`.

Same configuration surface as :class:`DeepAgent` but built on top of
:class:`AsyncAgent`. The ``task`` tool dispatches to async sub-agents,
and the summariser uses an async driver.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ..drivers.async_base import AsyncDriver
from ..infra.budget import BudgetPolicy
from ..infra.callbacks import DriverCallbacks
from .async_agent import AsyncAgent
from .deep.planner import make_write_todos_tool
from .deep.subagents import make_async_task_tool
from .deep.summarizer import AsyncSummarizationMiddleware
from .deep.vfs import VFS_TOOL_NAMES, make_vfs_tools
from .deep_agent import _normalise_user_tools, _resolve_persona_text, _upgrade_result
from .deep_prompts import assemble_system_prompt, format_subagent_section
from .deep_state import DeepAgentResult, DeepAgentState, SubAgentSpec, Todo
from .persona import Persona
from .tools_schema import ToolDefinition, ToolRegistry
from .types import AgentCallbacks

logger = logging.getLogger("prompture.agents.async_deep_agent")


class AsyncDeepAgent(AsyncAgent):
    """An :class:`AsyncAgent` with planning, VFS, sub-agents, and summarization."""

    def __init__(
        self,
        model: str = "",
        *,
        driver: AsyncDriver | None = None,
        tools: list[ToolDefinition] | list[Callable[..., Any]] | ToolRegistry | None = None,
        subagents: list[SubAgentSpec] | None = None,
        persona: Persona | str | None = None,
        system_prompt: str | Persona | None = None,
        enable_planning: bool = True,
        enable_vfs: bool = True,
        enable_summarization: bool = True,
        summarize_at_tokens: int = 80_000,
        summarize_keep_last_n: int = 6,
        summarizer_model: str | AsyncDriver | None = None,
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
        self._user_tools: list[ToolDefinition] = _normalise_user_tools(tools)
        self._subagents: list[SubAgentSpec] = list(subagents or [])

        self.deep_state: DeepAgentState = DeepAgentState(files=dict(initial_files or {}))

        self._enable_planning = bool(enable_planning)
        self._enable_vfs = bool(enable_vfs)
        self._enable_summarization = bool(enable_summarization)

        builtin_names = self._enabled_builtin_names()
        collisions = {t.name for t in self._user_tools} & builtin_names
        if collisions:
            warnings.warn(
                f"User-supplied tool name(s) shadow AsyncDeepAgent built-ins: "
                f"{sorted(collisions)}. Built-ins take precedence.",
                stacklevel=2,
            )
            self._user_tools = [t for t in self._user_tools if t.name not in collisions]

        all_defs = list(self._user_tools) + self._build_builtin_tools(
            model_for_subagents=model,
            budget_policy=budget_policy,
            callbacks=callbacks,
            max_tool_result_length=max_tool_result_length,
        )
        tool_registry = ToolRegistry()
        for td in all_defs:
            tool_registry.add(td)

        persona_text = _resolve_persona_text(persona, system_prompt)
        subagent_section = format_subagent_section(self._subagents) if self._subagents else None
        composed_prompt = assemble_system_prompt(
            persona_text=persona_text,
            enable_planning=self._enable_planning,
            enable_vfs=self._enable_vfs,
            subagent_section=subagent_section,
        )

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

        if self._enable_summarization:
            summariser = self._resolve_summariser(summarizer_model, model, driver)
            self._summarizer: AsyncSummarizationMiddleware | None = AsyncSummarizationMiddleware(
                threshold_tokens=summarize_at_tokens,
                keep_last_n=summarize_keep_last_n,
                state=self.deep_state,
                summariser=summariser,
            )
            self._before_turn_hook: Callable[..., Any] | None = self._summarizer
        else:
            self._summarizer = None
            self._before_turn_hook = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def todos(self) -> list[Todo]:
        return self.deep_state.todos

    @property
    def files(self) -> dict[str, str]:
        return self.deep_state.files

    def reset(self) -> None:
        self.deep_state.reset()
        self.clear_history()

    async def run(self, prompt: str, *, deps: Any = None) -> DeepAgentResult:  # type: ignore[override]
        base = await super().run(prompt, deps=deps)
        return _upgrade_result(base, self.deep_state)

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
                make_async_task_tool(
                    specs=self._subagents,
                    state=self.deep_state,
                    parent_model=model_for_subagents,
                    parent_user_tools=self._user_tools,
                    parent_budget_policy=resolved_budget,
                    parent_callbacks=callbacks,
                    parent_max_tool_result_length=max_tool_result_length,
                )
            )
        return tools

    @staticmethod
    def _resolve_summariser(
        summarizer_model: str | AsyncDriver | None,
        agent_model: str,
        agent_driver: AsyncDriver | None,
    ) -> str | AsyncDriver:
        if isinstance(summarizer_model, AsyncDriver):
            return summarizer_model
        if isinstance(summarizer_model, str) and summarizer_model.strip():
            return summarizer_model
        if agent_driver is not None:
            return agent_driver
        return agent_model


def create_async_deep_agent(
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
    summarizer_model: str | AsyncDriver | None = None,
    initial_files: dict[str, str] | None = None,
    max_iterations: int = 50,
    max_tool_result_length: int | None = 10_000,
    budget_policy: BudgetPolicy | str | None = None,
    max_cost: float | None = None,
    max_tokens: int | None = None,
    callbacks: DriverCallbacks | None = None,
    agent_callbacks: AgentCallbacks | None = None,
    driver: AsyncDriver | None = None,
    options: dict[str, Any] | None = None,
    output_type: type[BaseModel] | None = None,
    name: str = "",
    description: str = "",
) -> AsyncDeepAgent:
    """Convenience factory for :class:`AsyncDeepAgent`."""
    return AsyncDeepAgent(
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
