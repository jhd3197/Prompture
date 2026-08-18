"""Host annotations must survive Agent's tool wrapping.

``_wrap_tools_with_context`` rebuilds every :class:`ToolDefinition` around a
wrapper function.  Two things used to be silently dropped in that rebuild:

* ``ToolDefinition.metadata`` — free-form host annotations, emptied before any
  tool reached the loop, so a host could not gate execution on a flag it set
  itself.
* ``function.__skill__`` — read by :attr:`ToolDefinition.security_metadata`, so
  a tukuy-backed tool reported ``None`` for side effects after wrapping.

Both are advisory fields whose whole purpose is to be read by the host, which
is exactly why nothing failed loudly when they disappeared.
"""

from __future__ import annotations

from typing import Any

import pytest

from prompture.agents.agent import Agent, _tool_wants_context
from prompture.agents.async_agent import AsyncAgent
from prompture.agents.tools_schema import ToolRegistry
from prompture.agents.types import RunContext


class _Descriptor:
    name = "wipe"
    description = "Delete a path."
    side_effects = True
    requires_network = False


class _Skill:
    descriptor = _Descriptor()


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    def read_battery() -> str:
        """Read the battery."""
        return "100%"

    def wipe(path: str) -> str:
        """Delete a path."""
        return f"wiped {path}"

    wipe.__skill__ = _Skill()  # type: ignore[attr-defined]

    reg.register(read_battery, metadata={"is_write": False})
    reg.register(wipe, metadata={"is_write": True, "category": "fs"})
    return reg


def _ctx() -> RunContext[Any]:
    return RunContext(deps=None, model="test/model")


def _wrapped(agent: Agent | AsyncAgent) -> ToolRegistry:
    return agent._wrap_tools_with_context(_ctx())


@pytest.fixture(params=["sync", "async"])
def wrapped(request) -> ToolRegistry:
    reg = _registry()
    if request.param == "sync":
        return _wrapped(Agent(model="openai/gpt-4o-mini", tools=reg))
    return _wrapped(AsyncAgent(model="openai/gpt-4o-mini", tools=reg))


# ----------------------------------------------------------------------
# metadata
# ----------------------------------------------------------------------


def test_metadata_survives_the_wrap(wrapped):
    assert wrapped.get("wipe").metadata == {"is_write": True, "category": "fs"}
    assert wrapped.get("read_battery").metadata == {"is_write": False}


def test_metadata_still_supports_filtering_after_wrapping(wrapped):
    """The point of the field: a host gating execution on its own annotation."""
    read_only = wrapped.filter(lambda td: not td.metadata.get("is_write"))

    assert read_only.names == ["read_battery"]
    assert "wipe" not in read_only


# ----------------------------------------------------------------------
# __skill__ / security_metadata
# ----------------------------------------------------------------------


def test_security_metadata_survives_the_wrap(wrapped):
    meta = wrapped.get("wipe").security_metadata

    assert meta is not None, "__skill__ was stripped by the wrap"
    assert meta["name"] == "wipe"
    assert meta["side_effects"] is True


def test_native_tools_still_report_no_security_metadata(wrapped):
    """Only tukuy-backed tools have it; a plain tool must stay None."""
    assert wrapped.get("read_battery").security_metadata is None


# ----------------------------------------------------------------------
# The wrap itself must not regress
# ----------------------------------------------------------------------


def test_wrapper_is_not_mistaken_for_a_runcontext_tool(wrapped):
    """Guards against copying __wrapped__ (e.g. via functools.wraps), which
    would make inspect.signature report the inner signature."""
    assert _tool_wants_context(wrapped.get("wipe").function) is False


def test_schema_is_unchanged_by_the_wrap(wrapped):
    assert list(wrapped.get("wipe").parameters["properties"]) == ["path"]


def test_wrapped_tool_still_executes(wrapped):
    assert wrapped.execute("read_battery", {}) == "100%"
