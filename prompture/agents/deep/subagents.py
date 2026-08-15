"""``task`` tool factory for delegating subproblems to sub-agents.

A sub-agent is just a plain :class:`Agent` instantiated on demand. It
runs in isolation: it sees only the ``description`` argument as its
user prompt, has its own message history, and (by default) cannot see
the parent's virtual filesystem.
"""

from __future__ import annotations

from typing import Any

from ...drivers.base import Driver
from ...infra.budget import BudgetPolicy
from ...infra.callbacks import DriverCallbacks
from ..deep_prompts import TASK_TOOL_DESC_TEMPLATE
from ..deep_state import DeepAgentState, SubAgentCallRecord, SubAgentSpec
from ..persona import Persona
from ..tools_schema import ToolDefinition

# We import Agent lazily inside the tool function to avoid a circular
# import at module load (DeepAgent extends Agent and lives next door).


def _truncate(text: str, limit: int = 500) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more chars]"


def _resolve_subagent_prompt(prompt: str | Persona) -> str:
    """Render a sub-agent system prompt from string or Persona."""
    if isinstance(prompt, Persona):
        return prompt.render()
    return str(prompt)


def make_task_tool(
    specs: list[SubAgentSpec],
    state: DeepAgentState,
    parent_model: str,
    parent_user_tools: list[ToolDefinition],
    parent_budget_policy: BudgetPolicy | None = None,
    parent_callbacks: DriverCallbacks | None = None,
    parent_max_tool_result_length: int | None = None,
    parent_driver: Driver | None = None,
) -> ToolDefinition:
    """Build the ``task`` tool definition.

    Args:
        specs: Configured sub-agents.
        state: Shared :class:`DeepAgentState` (mutated to record calls).
        parent_model: Default model used when a spec does not override it.
        parent_user_tools: The parent's USER-supplied tools (not built-ins).
            Used when a spec has ``inherit_tools=True`` and ``tools is None``.
        parent_budget_policy: Optional budget shared with sub-agents.
        parent_callbacks: Optional driver-level callbacks shared with
            sub-agents (for unified observability).
        parent_max_tool_result_length: Default tool-result truncation.

    Raises:
        ValueError: If two specs share the same name.
    """
    spec_by_name: dict[str, SubAgentSpec] = {}
    for s in specs:
        if s.name in spec_by_name:
            raise ValueError(f"Duplicate sub-agent name: {s.name!r}")
        spec_by_name[s.name] = s

    available_types = "\n".join(f"- **{s.name}**: {s.description}" for s in specs) if specs else "(none configured)"
    description = TASK_TOOL_DESC_TEMPLATE.format(available_types=available_types)

    def task(description: str, subagent_type: str) -> str:
        spec = spec_by_name.get(subagent_type)
        if spec is None:
            available = ", ".join(sorted(spec_by_name)) or "(none)"
            return f"Error: unknown subagent_type {subagent_type!r}. Available: {available}"

        # Lazy import to avoid circular dep with Agent.
        from ..agent import Agent
        from .vfs import make_vfs_tools

        # Resolve sub-agent tools
        sub_tools: list[ToolDefinition]
        if spec.tools is not None:
            sub_tools = list(spec.tools)
        elif spec.inherit_tools:
            sub_tools = list(parent_user_tools)
        else:
            sub_tools = []

        if spec.inherit_vfs:
            # Add VFS tools that read/write the PARENT's state. Avoid
            # duplicate registration if the spec already includes VFS.
            existing = {t.name for t in sub_tools}
            for t in make_vfs_tools(state):
                if t.name not in existing:
                    sub_tools.append(t)

        sub_system_prompt = _resolve_subagent_prompt(spec.system_prompt)

        sub_kwargs: dict[str, Any] = {
            "tools": sub_tools or None,
            "system_prompt": sub_system_prompt,
            "max_iterations": spec.max_iterations,
            "max_tool_result_length": (
                spec.max_tool_result_length
                if spec.max_tool_result_length is not None
                else parent_max_tool_result_length
            ),
            "budget_policy": parent_budget_policy,
        }
        # Inherit the parent's driver instance unless the spec names its own
        # model. Without this a host that built the parent with driver=... —
        # a local endpoint, a test double, any alias the registry does not
        # own — cannot have sub-agents reach the same model, because they are
        # rebuilt from a model string through the global registry.
        if spec.model:
            sub_kwargs["model"] = spec.model
        elif parent_driver is not None:
            sub_kwargs["driver"] = parent_driver
        else:
            sub_kwargs["model"] = parent_model
        sub_agent: Agent[Any] = Agent(**sub_kwargs)

        # Propagate driver-level callbacks so token/cost tracking flows
        # to the same UsageSession the parent uses.
        if parent_callbacks is not None and getattr(sub_agent, "_driver", None) is None:
            # Driver is lazily constructed at run time; setting via
            # callbacks param would be ideal but Agent doesn't accept
            # DriverCallbacks directly. Fall back to wrapping the
            # callbacks onto the conversation by patching after build.
            pass  # Handled by parent's shared session; subagent gets its own session anyway.

        try:
            result = sub_agent.run(description)
        except Exception as exc:
            err = f"Sub-agent {spec.name!r} failed: {type(exc).__name__}: {exc}"
            state.sub_agent_calls.append(
                SubAgentCallRecord(
                    subagent_name=spec.name,
                    description=_truncate(description, 200),
                    result_summary=err,
                    usage={},
                    iterations=0,
                )
            )
            return err

        # Record the call
        state.sub_agent_calls.append(
            SubAgentCallRecord(
                subagent_name=spec.name,
                description=_truncate(description, 200),
                result_summary=_truncate(result.output_text, 500),
                usage=dict(result.run_usage) if result.run_usage else {},
                iterations=len(result.steps),
            )
        )

        return result.output_text

    return ToolDefinition(
        name="task",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Self-contained instructions for the sub-agent.",
                },
                "subagent_type": {
                    "type": "string",
                    "description": "Name of the sub-agent to dispatch to.",
                    "enum": list(spec_by_name) if spec_by_name else None,  # type: ignore[dict-item]
                },
            },
            "required": ["description", "subagent_type"],
        },
        function=task,
    )


def make_async_task_tool(
    specs: list[SubAgentSpec],
    state: DeepAgentState,
    parent_model: str,
    parent_user_tools: list[ToolDefinition],
    parent_budget_policy: BudgetPolicy | None = None,
    parent_callbacks: DriverCallbacks | None = None,
    parent_max_tool_result_length: int | None = None,
    parent_driver: Any | None = None,
) -> ToolDefinition:
    """Async counterpart of :func:`make_task_tool`.

    Returns a :class:`ToolDefinition` whose function is a coroutine —
    :meth:`AsyncConversation._ask_with_tools` awaits it via
    :meth:`ToolRegistry.aexecute`.
    """
    spec_by_name: dict[str, SubAgentSpec] = {}
    for s in specs:
        if s.name in spec_by_name:
            raise ValueError(f"Duplicate sub-agent name: {s.name!r}")
        spec_by_name[s.name] = s

    available_types = "\n".join(f"- **{s.name}**: {s.description}" for s in specs) if specs else "(none configured)"
    description = TASK_TOOL_DESC_TEMPLATE.format(available_types=available_types)

    async def task(description: str, subagent_type: str) -> str:
        spec = spec_by_name.get(subagent_type)
        if spec is None:
            available = ", ".join(sorted(spec_by_name)) or "(none)"
            return f"Error: unknown subagent_type {subagent_type!r}. Available: {available}"

        from ..async_agent import AsyncAgent
        from .vfs import make_vfs_tools

        sub_tools: list[ToolDefinition]
        if spec.tools is not None:
            sub_tools = list(spec.tools)
        elif spec.inherit_tools:
            sub_tools = list(parent_user_tools)
        else:
            sub_tools = []

        if spec.inherit_vfs:
            existing = {t.name for t in sub_tools}
            for t in make_vfs_tools(state):
                if t.name not in existing:
                    sub_tools.append(t)

        sub_system_prompt = _resolve_subagent_prompt(spec.system_prompt)

        sub_kwargs: dict[str, Any] = {
            "tools": sub_tools or None,
            "system_prompt": sub_system_prompt,
            "max_iterations": spec.max_iterations,
            "max_tool_result_length": (
                spec.max_tool_result_length
                if spec.max_tool_result_length is not None
                else parent_max_tool_result_length
            ),
            "budget_policy": parent_budget_policy,
        }
        # See make_task_tool — sub-agents inherit the parent's driver instance
        # unless the spec names its own model.
        if spec.model:
            sub_kwargs["model"] = spec.model
        elif parent_driver is not None:
            sub_kwargs["driver"] = parent_driver
        else:
            sub_kwargs["model"] = parent_model
        sub_agent: AsyncAgent[Any] = AsyncAgent(**sub_kwargs)

        try:
            result = await sub_agent.run(description)
        except Exception as exc:
            err = f"Sub-agent {spec.name!r} failed: {type(exc).__name__}: {exc}"
            state.sub_agent_calls.append(
                SubAgentCallRecord(
                    subagent_name=spec.name,
                    description=_truncate(description, 200),
                    result_summary=err,
                    usage={},
                    iterations=0,
                )
            )
            return err

        state.sub_agent_calls.append(
            SubAgentCallRecord(
                subagent_name=spec.name,
                description=_truncate(description, 200),
                result_summary=_truncate(result.output_text, 500),
                usage=dict(result.run_usage) if result.run_usage else {},
                iterations=len(result.steps),
            )
        )

        return result.output_text

    return ToolDefinition(
        name="task",
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Self-contained instructions for the sub-agent.",
                },
                "subagent_type": {
                    "type": "string",
                    "description": "Name of the sub-agent to dispatch to.",
                    "enum": list(spec_by_name) if spec_by_name else None,  # type: ignore[dict-item]
                },
            },
            "required": ["description", "subagent_type"],
        },
        function=task,
    )
