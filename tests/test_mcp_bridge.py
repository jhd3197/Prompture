"""Tests for the MCP (Model Context Protocol) bridge.

All MCP objects (ClientSession, tools, content blocks) are mocked — no real
MCP server or ``mcp`` package dependency is required.
"""

import sys
from types import SimpleNamespace

import pytest

from prompture.agents.tools_schema import ToolRegistry
from prompture.integrations import mcp_bridge
from prompture.integrations.mcp_bridge import (
    _sanitize_name,
    register_mcp_tools,
    register_mcp_tools_sync,
)


def _tool(name, description=None, input_schema=None):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema,
    )


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _image_block():
    return SimpleNamespace(type="image", data="...", mimeType="image/png")


class FakeSession:
    """Mock of ``mcp.ClientSession`` with async list_tools/call_tool."""

    def __init__(self, tools=None, call_result=None, call_error=None):
        self._tools = tools or []
        self._call_result = call_result
        self._call_error = call_error
        self.calls = []

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._call_error is not None:
            raise self._call_error
        return self._call_result


WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City name"}},
    "required": ["city"],
}


class TestRegistration:
    async def test_registers_tools_with_schema_passthrough(self):
        session = FakeSession(tools=[_tool("get_weather", "Get weather", WEATHER_SCHEMA)])
        registry = ToolRegistry()

        names = await register_mcp_tools(registry, session)

        assert names == ["get_weather"]
        td = registry.get("get_weather")
        assert td is not None
        assert td.description == "Get weather"
        assert td.parameters == WEATHER_SCHEMA  # JSON Schema passed through unchanged

    async def test_prefix_applied_to_names(self):
        session = FakeSession(tools=[_tool("read_file", "Read a file")])
        registry = ToolRegistry()

        names = await register_mcp_tools(registry, session, prefix="fs")

        assert names == ["fs_read_file"]
        assert "fs_read_file" in registry

    async def test_missing_description_and_schema_get_defaults(self):
        session = FakeSession(tools=[_tool("ping")])
        registry = ToolRegistry()

        await register_mcp_tools(registry, session)

        td = registry.get("ping")
        assert td.description == "MCP tool ping"
        assert td.parameters == {"type": "object", "properties": {}}

    async def test_empty_tool_list(self):
        registry = ToolRegistry()
        names = await register_mcp_tools(registry, FakeSession())
        assert names == []
        assert len(registry) == 0


class TestNameSanitization:
    def test_valid_name_unchanged(self):
        assert _sanitize_name("get_weather-2") == "get_weather-2"

    def test_invalid_chars_replaced(self):
        assert _sanitize_name("get weather.now!") == "get_weather_now_"

    def test_long_name_truncated_to_64(self):
        name = _sanitize_name("a" * 100)
        assert len(name) == 64

    def test_prefix_counts_toward_limit(self):
        name = _sanitize_name("b" * 100, prefix="srv")
        assert len(name) == 64
        assert name.startswith("srv_")

    async def test_sanitization_applied_on_registration(self):
        session = FakeSession(tools=[_tool("my tool/v2")])
        registry = ToolRegistry()

        names = await register_mcp_tools(registry, session)

        assert names == ["my_tool_v2"]
        assert "my_tool_v2" in registry


class TestExecutionDispatch:
    async def test_aexecute_dispatches_to_session_call_tool(self):
        result = SimpleNamespace(content=[_text_block("sunny"), _text_block("25C")], isError=False)
        session = FakeSession(tools=[_tool("get_weather", "Get weather", WEATHER_SCHEMA)], call_result=result)
        registry = ToolRegistry()
        await register_mcp_tools(registry, session, prefix="w")

        output = await registry.aexecute("w_get_weather", {"city": "Paris"})

        # Original MCP name (not the sanitized/prefixed one) is used server-side.
        assert session.calls == [("get_weather", {"city": "Paris"})]
        assert output == "sunny\n25C"  # text blocks joined

    async def test_async_fn_hook_attached(self):
        session = FakeSession(tools=[_tool("ping")], call_result=SimpleNamespace(content=[], isError=False))
        registry = ToolRegistry()
        await register_mcp_tools(registry, session)

        td = registry.get("ping")
        assert getattr(td.function, "_async_fn", None) is not None

    def test_sync_execute_runs_outside_event_loop(self):
        result = SimpleNamespace(content=[_text_block("pong")], isError=False)
        session = FakeSession(tools=[_tool("ping")], call_result=result)
        registry = ToolRegistry()
        register_mcp_tools_sync(registry, session)

        assert registry.execute("ping", {}) == "pong"
        assert session.calls == [("ping", {})]

    async def test_sync_execute_inside_event_loop_raises_clear_error(self):
        session = FakeSession(tools=[_tool("ping")])
        registry = ToolRegistry()
        await register_mcp_tools(registry, session)

        with pytest.raises(RuntimeError, match="running event loop"):
            registry.execute("ping", {})


class TestErrorMapping:
    async def test_call_exception_becomes_error_string(self):
        session = FakeSession(
            tools=[_tool("boom")],
            call_error=ConnectionError("server unreachable"),
        )
        registry = ToolRegistry()
        await register_mcp_tools(registry, session)

        output = await registry.aexecute("boom", {})

        assert "Error calling MCP tool 'boom'" in output
        assert "server unreachable" in output

    async def test_is_error_result_becomes_error_string(self):
        result = SimpleNamespace(content=[_text_block("no such file")], isError=True)
        session = FakeSession(tools=[_tool("read_file")], call_result=result)
        registry = ToolRegistry()
        await register_mcp_tools(registry, session)

        output = await registry.aexecute("read_file", {})

        assert output == "Error from MCP tool 'read_file': no such file"


class TestContentSerialization:
    async def test_non_text_blocks_become_placeholder(self):
        result = SimpleNamespace(
            content=[_text_block("here you go"), _image_block()],
            isError=False,
        )
        session = FakeSession(tools=[_tool("screenshot")], call_result=result)
        registry = ToolRegistry()
        await register_mcp_tools(registry, session)

        output = await registry.aexecute("screenshot", {})

        assert "here you go" in output
        assert "[image content block]" in output
        assert "mimeType" not in output  # raw block not leaked

    async def test_empty_content_returns_empty_string(self):
        session = FakeSession(tools=[_tool("noop")], call_result=SimpleNamespace(content=[], isError=False))
        registry = ToolRegistry()
        await register_mcp_tools(registry, session)

        assert await registry.aexecute("noop", {}) == ""


class TestSyncRegistrationWrapper:
    def test_works_outside_event_loop(self):
        session = FakeSession(tools=[_tool("ping")])
        registry = ToolRegistry()

        names = register_mcp_tools_sync(registry, session)

        assert names == ["ping"]
        assert "ping" in registry

    async def test_raises_inside_running_loop(self):
        registry = ToolRegistry()
        with pytest.raises(RuntimeError, match="register_mcp_tools"):
            register_mcp_tools_sync(registry, FakeSession())


class TestImportGuard:
    async def test_clear_error_when_mcp_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "mcp", None)  # makes `import mcp` raise ImportError

        with pytest.raises(ImportError, match=r"pip install 'prompture\[mcp\]'"):
            async with mcp_bridge.mcp_session_from_stdio("some-server-command"):
                pass

    def test_bridge_module_importable_without_mcp(self):
        # The bridge module is already imported at the top of this file without
        # the ``mcp`` package needing to be installed (lazy import).
        assert mcp_bridge.register_mcp_tools is not None
