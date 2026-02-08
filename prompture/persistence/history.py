"""History export and filtering utilities for Agent results.

Provides functions to filter, search, and analyze agent execution history.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..agents.types import AgentResult, AgentStep, StepType


def filter_steps(
    steps: list[AgentStep],
    *,
    step_type: StepType | list[StepType] | None = None,
    tool_name: str | list[str] | None = None,
    after_timestamp: float | None = None,
    before_timestamp: float | None = None,
) -> list[AgentStep]:
    """Filter steps by type, tool name, or timestamp range.

    Args:
        steps: List of AgentStep objects to filter.
        step_type: Filter by step type (single or list).
        tool_name: Filter by tool name (single or list). Only applies
            to tool_call and tool_result step types.
        after_timestamp: Only include steps after this Unix timestamp.
        before_timestamp: Only include steps before this Unix timestamp.

    Returns:
        Filtered list of AgentStep objects.

    Example::

        from prompture.history import filter_steps
        from prompture import StepType

        # Get only tool call steps
        tool_calls = filter_steps(result.steps, step_type=StepType.tool_call)

        # Get steps for a specific tool
        search_steps = filter_steps(result.steps, tool_name="search")

        # Get steps in a time range
        recent = filter_steps(result.steps, after_timestamp=start_time)
    """
    result: list[AgentStep] = []

    # Normalize step_type to a set
    type_filter: set[StepType] | None = None
    if step_type is not None:
        if isinstance(step_type, list):
            type_filter = set(step_type)
        else:
            type_filter = {step_type}

    # Normalize tool_name to a set
    tool_filter: set[str] | None = None
    if tool_name is not None:
        if isinstance(tool_name, list):
            tool_filter = set(tool_name)
        else:
            tool_filter = {tool_name}

    for step in steps:
        # Filter by type
        if type_filter is not None and step.step_type not in type_filter:
            continue

        # Filter by tool name
        if tool_filter is not None and (step.tool_name is None or step.tool_name not in tool_filter):
            continue

        # Filter by timestamp range
        if after_timestamp is not None and step.timestamp <= after_timestamp:
            continue
        if before_timestamp is not None and step.timestamp >= before_timestamp:
            continue

        result.append(step)

    return result


def search_messages(
    messages: list[dict[str, Any]],
    *,
    role: str | list[str] | None = None,
    content_contains: str | None = None,
    has_tool_calls: bool | None = None,
) -> list[dict[str, Any]]:
    """Search messages by role or content.

    Args:
        messages: List of message dictionaries from AgentResult.messages.
        role: Filter by role (single or list, e.g., "user", "assistant", "tool").
        content_contains: Filter to messages whose content contains this substring
            (case-insensitive).
        has_tool_calls: If True, only return messages with tool_calls.
            If False, only return messages without tool_calls.

    Returns:
        Filtered list of message dictionaries.

    Example::

        from prompture.history import search_messages

        # Get all assistant messages
        assistant_msgs = search_messages(result.messages, role="assistant")

        # Search for messages mentioning "error"
        error_msgs = search_messages(result.messages, content_contains="error")

        # Get messages with tool calls
        tool_msgs = search_messages(result.messages, has_tool_calls=True)
    """
    result: list[dict[str, Any]] = []

    # Normalize role to a set
    role_filter: set[str] | None = None
    if role is not None:
        if isinstance(role, list):
            role_filter = set(role)
        else:
            role_filter = {role}

    for msg in messages:
        # Filter by role
        if role_filter is not None:
            msg_role = msg.get("role", "")
            if msg_role not in role_filter:
                continue

        # Filter by content
        if content_contains is not None:
            content = msg.get("content", "")
            if content is None:
                content = ""
            if content_contains.lower() not in content.lower():
                continue

        # Filter by tool calls
        if has_tool_calls is not None:
            msg_has_tools = bool(msg.get("tool_calls"))
            if msg_has_tools != has_tool_calls:
                continue

        result.append(msg)

    return result


def get_tool_call_summary(result: AgentResult) -> list[dict[str, Any]]:
    """Get a summary of all tool calls from an AgentResult.

    Returns a list of dictionaries, each containing:
    - name: The tool name
    - arguments: The arguments passed to the tool
    - result: The tool's return value (if available)
    - timestamp: When the call was made

    Args:
        result: AgentResult from an agent run.

    Returns:
        List of tool call summary dictionaries.

    Example::

        from prompture.history import get_tool_call_summary

        summary = get_tool_call_summary(result)
        for call in summary:
            print(f"{call['name']}: {call['arguments']} -> {call.get('result', 'N/A')}")
    """
    summaries: list[dict[str, Any]] = []

    # Pair up tool_call steps with their corresponding tool_result steps
    tool_calls = [s for s in result.steps if s.step_type == StepType.tool_call]
    tool_results = [s for s in result.steps if s.step_type == StepType.tool_result]

    for i, call_step in enumerate(tool_calls):
        summary: dict[str, Any] = {
            "name": call_step.tool_name,
            "arguments": call_step.tool_args or {},
            "timestamp": call_step.timestamp,
        }

        # Try to find the corresponding result
        if i < len(tool_results):
            result_step = tool_results[i]
            summary["result"] = result_step.content

        summaries.append(summary)

    return summaries


def calculate_cost_breakdown(run_usage: dict[str, Any]) -> dict[str, Any]:
    """Calculate a detailed cost breakdown from run_usage.

    Args:
        run_usage: The run_usage dictionary from AgentResult.

    Returns:
        Dictionary with cost breakdown:
        - prompt_tokens: Total prompt tokens used
        - completion_tokens: Total completion tokens used
        - total_tokens: Total tokens used
        - prompt_cost: Cost for prompt tokens
        - completion_cost: Cost for completion tokens
        - total_cost: Total cost
        - call_count: Number of API calls
        - error_count: Number of errors

    Example::

        from prompture.history import calculate_cost_breakdown

        breakdown = calculate_cost_breakdown(result.run_usage)
        print(f"Total cost: ${breakdown['total_cost']:.4f}")
        print(f"Calls: {breakdown['call_count']}")
    """
    return {
        "prompt_tokens": run_usage.get("prompt_tokens", 0),
        "completion_tokens": run_usage.get("completion_tokens", 0),
        "total_tokens": run_usage.get("total_tokens", 0),
        "prompt_cost": run_usage.get("prompt_cost", 0.0),
        "completion_cost": run_usage.get("completion_cost", 0.0),
        "total_cost": run_usage.get("total_cost", 0.0),
        "call_count": run_usage.get("call_count", 0),
        "error_count": run_usage.get("error_count", 0),
    }


def export_result_json(result: AgentResult, include_messages: bool = True) -> str:
    """Export an AgentResult to a JSON string.

    Args:
        result: AgentResult to export.
        include_messages: Whether to include the full message history.

    Returns:
        JSON string representation of the result.

    Example::

        from prompture.history import export_result_json

        json_str = export_result_json(result)
        with open("agent_history.json", "w") as f:
            f.write(json_str)
    """
    data = result_to_dict(result, include_messages=include_messages)
    return json.dumps(data, indent=2, default=str)


def result_to_dict(result: AgentResult, include_messages: bool = True) -> dict[str, Any]:
    """Convert an AgentResult to a dictionary.

    Args:
        result: AgentResult to convert.
        include_messages: Whether to include the full message history.

    Returns:
        Dictionary representation of the result.
    """
    data: dict[str, Any] = {
        "output": str(result.output) if result.output is not None else None,
        "output_text": result.output_text,
        "state": result.state.value if hasattr(result.state, "value") else str(result.state),
        "usage": result.usage,
        "run_usage": result.run_usage,
        "steps": [_step_to_dict(s) for s in result.steps],
        "all_tool_calls": result.all_tool_calls,
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    if include_messages:
        data["messages"] = result.messages

    return data


def _step_to_dict(step: AgentStep) -> dict[str, Any]:
    """Convert an AgentStep to a dictionary."""
    return {
        "step_type": step.step_type.value if hasattr(step.step_type, "value") else str(step.step_type),
        "timestamp": step.timestamp,
        "content": step.content,
        "tool_name": step.tool_name,
        "tool_args": step.tool_args,
        "tool_result": step.tool_result,
        "duration_ms": step.duration_ms,
    }
