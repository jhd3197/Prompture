"""Tests for prompt-based simulated tool calling."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prompture.driver import Driver
from prompture.simulated_tools import (
    build_tool_prompt,
    format_tool_result,
    parse_simulated_response,
)
from prompture.tools_schema import ToolDefinition, ToolRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_registry() -> ToolRegistry:
    """Create a simple tool registry for testing."""
    registry = ToolRegistry()

    def get_weather(city: str, units: str = "celsius") -> str:
        """Get the current weather for a city."""
        return f"Weather in {city}: 22 {units}"

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    registry.register(get_weather)
    registry.register(add_numbers)
    return registry


@pytest.fixture
def tools():
    return _make_registry()


class MockDriver(Driver):
    """A mock driver that returns pre-configured responses in sequence."""

    supports_tool_use = False
    supports_messages = True

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_count = 0
        self.callbacks = None

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        if self._call_count >= len(self._responses):
            raise RuntimeError("MockDriver ran out of responses")
        text = self._responses[self._call_count]
        self._call_count += 1
        return {
            "text": text,
            "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0},
        }


class NativeToolDriver(Driver):
    """A driver that claims native tool use support."""

    supports_tool_use = True
    supports_messages = True

    def __init__(self):
        self.callbacks = None

    def generate_messages(self, messages, options):
        return {
            "text": "native response",
            "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
        }

    def generate_messages_with_tools(self, messages, tools, options):
        return {
            "text": "native tool response",
            "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            "tool_calls": [],
            "stop_reason": "stop",
        }


# ---------------------------------------------------------------------------
# build_tool_prompt
# ---------------------------------------------------------------------------


class TestBuildToolPrompt:
    def test_includes_tool_names(self, tools):
        prompt = build_tool_prompt(tools)
        assert "get_weather" in prompt
        assert "add_numbers" in prompt

    def test_includes_descriptions(self, tools):
        prompt = build_tool_prompt(tools)
        assert "Get the current weather" in prompt
        assert "Add two numbers" in prompt

    def test_includes_parameters(self, tools):
        prompt = build_tool_prompt(tools)
        assert "city" in prompt
        assert "units" in prompt
        assert "required" in prompt
        assert "optional" in prompt

    def test_includes_format_examples(self, tools):
        prompt = build_tool_prompt(tools)
        assert "tool_call" in prompt
        assert "final_answer" in prompt
        assert "JSON" in prompt


# ---------------------------------------------------------------------------
# to_prompt_format
# ---------------------------------------------------------------------------


class TestToPromptFormat:
    def test_tool_definition_prompt_format(self):
        td = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "An integer"},
                    "y": {"type": "string"},
                },
                "required": ["x"],
            },
            function=lambda x, y="": None,
        )
        text = td.to_prompt_format()
        assert "Tool: test_tool" in text
        assert "Description: A test tool" in text
        assert "x (integer, required): An integer" in text
        assert "y (string, optional)" in text

    def test_tool_definition_no_params(self):
        td = ToolDefinition(
            name="no_params",
            description="No params tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: None,
        )
        text = td.to_prompt_format()
        assert "(none)" in text

    def test_registry_prompt_format(self, tools):
        text = tools.to_prompt_format()
        assert "get_weather" in text
        assert "add_numbers" in text
        # Two tools separated by blank line
        assert "\n\n" in text


# ---------------------------------------------------------------------------
# parse_simulated_response
# ---------------------------------------------------------------------------


class TestParseSimulatedResponse:
    def test_parse_tool_call(self, tools):
        text = json.dumps({"type": "tool_call", "name": "get_weather", "arguments": {"city": "London"}})
        result = parse_simulated_response(text, tools)
        assert result["type"] == "tool_call"
        assert result["name"] == "get_weather"
        assert result["arguments"] == {"city": "London"}

    def test_parse_final_answer(self, tools):
        text = json.dumps({"type": "final_answer", "content": "The weather is sunny."})
        result = parse_simulated_response(text, tools)
        assert result["type"] == "final_answer"
        assert result["content"] == "The weather is sunny."

    def test_parse_inferred_tool_call(self, tools):
        text = json.dumps({"name": "add_numbers", "arguments": {"a": 1, "b": 2}})
        result = parse_simulated_response(text, tools)
        assert result["type"] == "tool_call"
        assert result["name"] == "add_numbers"
        assert result["arguments"] == {"a": 1, "b": 2}

    def test_parse_inferred_final_answer(self, tools):
        text = json.dumps({"content": "Here is your answer."})
        result = parse_simulated_response(text, tools)
        assert result["type"] == "final_answer"
        assert result["content"] == "Here is your answer."

    def test_parse_plain_text_fallback(self, tools):
        text = "I don't know how to use tools, here's my answer."
        result = parse_simulated_response(text, tools)
        assert result["type"] == "final_answer"
        assert "I don't know" in result["content"]

    def test_parse_markdown_wrapped(self, tools):
        inner = json.dumps({"type": "tool_call", "name": "get_weather", "arguments": {"city": "Paris"}})
        text = f"```json\n{inner}\n```"
        result = parse_simulated_response(text, tools)
        assert result["type"] == "tool_call"
        assert result["name"] == "get_weather"
        assert result["arguments"]["city"] == "Paris"

    def test_parse_unknown_json_structure(self, tools):
        text = json.dumps({"foo": "bar"})
        result = parse_simulated_response(text, tools)
        assert result["type"] == "final_answer"


# ---------------------------------------------------------------------------
# format_tool_result
# ---------------------------------------------------------------------------


class TestFormatToolResult:
    def test_string_result(self):
        msg = format_tool_result("get_weather", "Sunny, 22C")
        assert "get_weather" in msg
        assert "Sunny, 22C" in msg
        assert "JSON format" in msg

    def test_dict_result(self):
        msg = format_tool_result("add_numbers", {"result": 42})
        assert "add_numbers" in msg
        assert "42" in msg


# ---------------------------------------------------------------------------
# Conversation integration (simulated tool loop)
# ---------------------------------------------------------------------------


class TestSimulatedToolLoop:
    def test_single_round(self, tools):
        """Mock driver returns tool call, then final answer."""
        from prompture.conversation import Conversation

        responses = [
            json.dumps({"type": "tool_call", "name": "get_weather", "arguments": {"city": "London"}}),
            json.dumps({"type": "final_answer", "content": "The weather in London is 22 celsius."}),
        ]
        driver = MockDriver(responses)
        conv = Conversation(driver=driver, system_prompt="You are helpful.", tools=tools, simulated_tools=True)
        result = conv.ask("What is the weather in London?")
        assert result == "The weather in London is 22 celsius."
        assert driver._call_count == 2

    def test_max_rounds_exceeded(self, tools):
        """Should raise RuntimeError when max rounds exceeded."""
        from prompture.conversation import Conversation

        # Always returns tool calls, never a final answer
        responses = [json.dumps({"type": "tool_call", "name": "get_weather", "arguments": {"city": "London"}})] * 5
        driver = MockDriver(responses)
        conv = Conversation(driver=driver, tools=tools, simulated_tools=True, max_tool_rounds=3)
        with pytest.raises(RuntimeError, match="exceeded 3 rounds"):
            conv.ask("What is the weather?")

    def test_tool_error_becomes_message(self, tools):
        """Tool execution errors are sent back as user messages."""
        from prompture.conversation import Conversation

        def bad_tool(x: str) -> str:
            """A tool that always fails."""
            raise ValueError("Something went wrong")

        tools.register(bad_tool)

        responses = [
            json.dumps({"type": "tool_call", "name": "bad_tool", "arguments": {"x": "test"}}),
            json.dumps({"type": "final_answer", "content": "The tool failed, sorry."}),
        ]
        driver = MockDriver(responses)
        conv = Conversation(driver=driver, tools=tools, simulated_tools=True)
        result = conv.ask("Try the bad tool")
        assert result == "The tool failed, sorry."

    def test_auto_mode_activates(self, tools):
        """simulated_tools='auto' with no native support → simulation path."""
        from prompture.conversation import Conversation

        responses = [
            json.dumps({"type": "final_answer", "content": "done"}),
        ]
        driver = MockDriver(responses)
        assert driver.supports_tool_use is False
        conv = Conversation(driver=driver, tools=tools, simulated_tools="auto")
        result = conv.ask("Hello")
        assert result == "done"
        # Verify the driver was called (simulation path used generate_messages_with_hooks)
        assert driver._call_count == 1

    def test_auto_mode_prefers_native(self, tools):
        """simulated_tools='auto' with native support → native tool path."""
        from prompture.conversation import Conversation

        driver = NativeToolDriver()
        conv = Conversation(driver=driver, tools=tools, simulated_tools="auto")
        result = conv.ask("Hello")
        assert result == "native tool response"

    def test_disabled_mode(self, tools):
        """simulated_tools=False → tools ignored, plain ask."""
        from prompture.conversation import Conversation

        responses = [
            json.dumps({"type": "final_answer", "content": "ignored"}),
        ]
        driver = MockDriver(responses)
        conv = Conversation(driver=driver, tools=tools, simulated_tools=False)
        result = conv.ask("Hello")
        # Should get raw text back, not parsed as final_answer
        assert "final_answer" in result  # raw JSON since tools are ignored

    def test_forced_mode(self, tools):
        """simulated_tools=True → simulation even with native support."""
        from prompture.conversation import Conversation

        # Use a driver that has supports_tool_use=True but we force simulation
        driver = NativeToolDriver()
        # Patch generate_messages to return tool call then final answer
        call_count = 0

        def mock_generate(messages, options):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return {
                    "text": json.dumps({"type": "tool_call", "name": "get_weather", "arguments": {"city": "Tokyo"}}),
                    "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
                }
            call_count += 1
            return {
                "text": json.dumps({"type": "final_answer", "content": "Tokyo weather is nice."}),
                "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            }

        driver.generate_messages = mock_generate
        conv = Conversation(driver=driver, tools=tools, simulated_tools=True)
        result = conv.ask("Weather in Tokyo?")
        assert result == "Tokyo weather is nice."
        assert call_count == 2

    def test_history_format(self, tools):
        """After simulation, all messages should use user/assistant roles."""
        from prompture.conversation import Conversation

        responses = [
            json.dumps({"type": "tool_call", "name": "get_weather", "arguments": {"city": "NYC"}}),
            json.dumps({"type": "final_answer", "content": "NYC is warm."}),
        ]
        driver = MockDriver(responses)
        conv = Conversation(driver=driver, tools=tools, simulated_tools=True)
        conv.ask("Weather in NYC?")

        for msg in conv.messages:
            assert msg["role"] in ("user", "assistant"), f"Unexpected role: {msg['role']}"
