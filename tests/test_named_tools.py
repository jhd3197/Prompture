"""Tests for ``"namespace:name"`` tool specs accepted by agents."""

from __future__ import annotations

import pytest

from prompture.agents.tools_schema import ToolDefinition
from prompture.tools.named import (
    expand_tool_specs,
    register_tool_namespace,
    resolve_tool_spec,
    unregister_tool_namespace,
)


def _echo_def(name: str) -> ToolDefinition:
    def echo(text: str) -> str:
        return text

    return ToolDefinition(
        name=name,
        description="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        function=echo,
    )


@pytest.fixture
def demo_namespace():
    register_tool_namespace("demo", lambda name: [_echo_def(f"demo_{name}")])
    yield
    unregister_tool_namespace("demo")


def test_resolve_and_expand(demo_namespace):
    assert [t.name for t in resolve_tool_spec("demo:one")] == ["demo_one"]

    def plain(x: int) -> int:
        return x

    expanded = expand_tool_specs(["demo:two", plain])
    assert expanded[0].name == "demo_two"
    assert expanded[1] is plain


def test_unknown_namespace():
    with pytest.raises(ValueError, match="Unknown tool namespace"):
        resolve_tool_spec("nope:thing")
    with pytest.raises(ValueError):
        resolve_tool_spec("no-colon")


def test_agent_accepts_specs(demo_namespace):
    from prompture.agents.agent import Agent

    def plain(x: int) -> int:
        """Return x."""
        return x

    agent = Agent("openai/gpt-4o-mini", tools=["demo:three", plain])
    names = set(agent._tools.names) if hasattr(agent._tools, "names") else {t.name for t in agent._tools.definitions}
    assert {"demo_three", "plain"} <= names


def test_deep_agent_normaliser_accepts_specs(demo_namespace):
    from prompture.agents.deep_agent import _normalise_user_tools

    assert [t.name for t in _normalise_user_tools(["demo:four"])] == ["demo_four"]
