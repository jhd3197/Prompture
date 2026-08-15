"""Tests for the MCP runtime: tool catalog, server registration, client adaptation (G11)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from prompture.agents import ToolRegistry
from prompture.mcp import (
    build_mcp_server,
    fetch_mcp_tool_definitions,
    load_mcp_registry,
    mcp_tool_to_definition,
    prompture_estimate_cost,
    prompture_tool_definitions,
    register_tools,
)
from prompture.mcp.tools import _coerce_tool_definitions


class TestToolCatalog:
    def test_default_set(self):
        names = {d.name for d in prompture_tool_definitions()}
        assert {"generate_image", "generate_video", "generate_music", "generate_text", "estimate_cost"} <= names

    def test_generate_text_schema(self):
        gen = next(d for d in prompture_tool_definitions() if d.name == "generate_text")
        assert gen.parameters["required"] == ["model", "prompt"]

    def test_toggle_media_off(self):
        names = {d.name for d in prompture_tool_definitions(media=False)}
        assert "generate_image" not in names
        assert "generate_text" in names

    def test_extra_appended(self):
        from prompture.agents import ToolDefinition

        extra = ToolDefinition(
            name="custom", description="c", parameters={"type": "object", "properties": {}}, function=lambda: 1
        )
        names = {d.name for d in prompture_tool_definitions(extra=[extra])}
        assert "custom" in names

    def test_estimate_cost_tool_value(self):
        assert prompture_estimate_cost("muapi/nano-banana", n=2) == round(0.039 * 2, 6)

    def test_generate_text_tool(self):
        from prompture.mcp.tools import prompture_generate

        class _Drv:
            def generate(self, prompt, options):
                return {"text": f"out:{prompt}"}

        with patch("prompture.drivers.get_driver_for_model", return_value=_Drv()):
            assert prompture_generate("mock/m", "hi") == "out:hi"

    def test_coerce_from_registry(self):
        reg = ToolRegistry()
        reg.add(prompture_tool_definitions()[0])
        assert len(_coerce_tool_definitions(reg)) == 1
        assert _coerce_tool_definitions(None)  # defaults
        with pytest.raises(TypeError):
            _coerce_tool_definitions(42)


class _FakeFastMCP:
    def __init__(self, name, **kw):
        self.name = name
        self.registered: list[str] = []

    def add_tool(self, fn, name=None, description=None):
        self.registered.append(name)


class TestServer:
    def test_requires_mcp_package(self):
        # mcp SDK is not installed in the test env → clear error with install hint.
        with pytest.raises(RuntimeError, match=r"prompture\[mcp\]"):
            build_mcp_server()

    def test_build_with_fake_fastmcp_registers_all(self):
        with patch("prompture.mcp.server._require_fastmcp", return_value=_FakeFastMCP):
            srv = build_mcp_server("prompture-test")
        assert "generate_image" in srv.registered
        assert "generate_text" in srv.registered
        assert srv.name == "prompture-test"

    def test_register_tools_explicit_list(self):
        srv = _FakeFastMCP("x")
        register_tools(srv, prompture_tool_definitions(media=False))
        assert srv.registered == ["generate_text", "estimate_cost"]


class _FakeTool:
    def __init__(self, name, description, schema):
        self.name = name
        self.description = description
        self.inputSchema = schema


class _Content:
    def __init__(self, text):
        self.text = text


class _Result:
    def __init__(self, *texts):
        self.content = [_Content(t) for t in texts]


class _FakeSession:
    def __init__(self):
        self.calls = []

    async def list_tools(self):
        class _L:
            tools = [_FakeTool("remote_add", "Add", {"type": "object", "properties": {"a": {"type": "integer"}}})]

        return _L()

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return _Result(f"{name}:{args}")


class TestClient:
    def test_adapt_tool(self):
        sess = _FakeSession()
        td = mcp_tool_to_definition(_FakeTool("t", "desc", {"type": "object", "properties": {"x": {}}}), sess)
        assert td.name == "t"
        assert td.description == "desc"
        assert td.parameters["properties"] == {"x": {}}

    def test_remote_call_extracts_single_text(self):
        sess = _FakeSession()
        td = mcp_tool_to_definition(_FakeTool("remote_add", "d", {"type": "object", "properties": {}}), sess)
        out = asyncio.run(td.function(a=5))
        assert out == "remote_add:{'a': 5}"
        assert sess.calls == [("remote_add", {"a": 5})]

    def test_extract_content_variants(self):
        from prompture.mcp.client import _extract_content

        assert _extract_content(_Result("only")) == "only"
        assert _extract_content(_Result("a", "b")) == ["a", "b"]
        assert _extract_content("raw") == "raw"  # no .content

    def test_fetch_and_load_registry(self):
        sess = _FakeSession()
        defs = asyncio.run(fetch_mcp_tool_definitions(sess))
        assert [d.name for d in defs] == ["remote_add"]
        reg = asyncio.run(load_mcp_registry(sess))
        assert "remote_add" in reg

    def test_default_schema_when_missing(self):
        class _BareTool:
            name = "bare"
            description = None
            inputSchema = None

        td = mcp_tool_to_definition(_BareTool(), _FakeSession())
        assert td.parameters == {"type": "object", "properties": {}}
        assert td.description == "bare"
