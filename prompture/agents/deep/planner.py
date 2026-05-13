"""``write_todos`` tool factory for :class:`DeepAgent`."""

from __future__ import annotations

from typing import Any

from ..deep_prompts import WRITE_TODOS_DESC
from ..deep_state import DeepAgentState, Todo
from ..tools_schema import ToolDefinition

_VALID_STATUSES = {"pending", "in_progress", "completed"}


def make_write_todos_tool(state: DeepAgentState) -> ToolDefinition:
    """Return a :class:`ToolDefinition` that replaces ``state.todos``.

    The tool function mutates ``state`` in place. The new list is
    validated lightly — bad statuses are normalised to ``"pending"``
    rather than rejected, so the model can recover.
    """

    def write_todos(todos: list[dict[str, Any]]) -> str:
        """Replace the current todo list. See WRITE_TODOS_DESC for schema."""
        new_todos: list[Todo] = []
        for raw in todos or []:
            if not isinstance(raw, dict):
                continue
            content = str(raw.get("content", "")).strip()
            if not content:
                continue
            status = raw.get("status", "pending")
            if status not in _VALID_STATUSES:
                status = "pending"
            active_form = raw.get("active_form")
            new_todos.append(
                Todo(content=content, status=status, active_form=active_form)  # type: ignore[arg-type]
            )

        # Enforce at most one in_progress
        in_progress_seen = False
        for todo in new_todos:
            if todo.status == "in_progress":
                if in_progress_seen:
                    todo.status = "pending"
                else:
                    in_progress_seen = True

        state.todos = new_todos
        n_done = sum(1 for t in new_todos if t.status == "completed")
        n_active = sum(1 for t in new_todos if t.status == "in_progress")
        return (
            f"Todo list updated: {len(new_todos)} item(s), "
            f"{n_done} completed, {n_active} in progress."
        )

    return ToolDefinition(
        name="write_todos",
        description=WRITE_TODOS_DESC,
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Full replacement list of todo objects.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The action statement.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current status of the todo.",
                            },
                            "active_form": {
                                "type": "string",
                                "description": "Present-continuous label for UI display.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
        function=write_todos,
    )
