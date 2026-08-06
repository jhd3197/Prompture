"""Tests for function calling / tool use support."""

from __future__ import annotations

from typing import Any

import pytest

from prompture.agents.conversation import Conversation
from prompture.agents.tools_schema import MAX_TOOL_DESCRIPTION_CHARS, ToolRegistry, tool_from_function
from prompture.drivers.base import Driver

# ---------------------------------------------------------------------------
# tool_from_function / ToolDefinition tests
# ---------------------------------------------------------------------------


class TestToolFromFunction:
    def test_basic_function(self):
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello {name}"

        td = tool_from_function(greet)
        assert td.name == "greet"
        assert td.description == "Say hello."
        assert td.parameters["type"] == "object"
        assert "name" in td.parameters["properties"]
        assert td.parameters["properties"]["name"]["type"] == "string"
        assert td.parameters["required"] == ["name"]

    def test_default_params_not_required(self):
        def add(a: int, b: int = 0) -> int:
            """Add two numbers."""
            return a + b

        td = tool_from_function(add)
        assert td.parameters["required"] == ["a"]
        assert "b" in td.parameters["properties"]

    def test_name_override(self):
        def foo():
            pass

        td = tool_from_function(foo, name="bar", description="Custom desc")
        assert td.name == "bar"
        assert td.description == "Custom desc"

    def test_openai_format(self):
        def f(x: int) -> str:
            """Do stuff."""
            return str(x)

        td = tool_from_function(f)
        fmt = td.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "f"
        assert "parameters" in fmt["function"]

    def test_anthropic_format(self):
        def f(x: int) -> str:
            """Do stuff."""
            return str(x)

        td = tool_from_function(f)
        fmt = td.to_anthropic_format()
        assert fmt["name"] == "f"
        assert "input_schema" in fmt


class TestDocstringDescription:
    def test_extended_description_is_kept(self):
        def search(query: str) -> str:
            """Search the internal wiki.

            Covers engineering runbooks and onboarding docs only.
            Does not reach public web pages.

            Args:
                query: Full-text search query.

            Returns:
                The top matching page as markdown.
            """
            return query

        td = tool_from_function(search)
        assert td.description.startswith("Search the internal wiki.")
        assert "engineering runbooks" in td.description
        assert "Does not reach public web pages." in td.description
        # Args live in the parameter schema, not the description
        assert "Full-text search query" not in td.description
        assert td.parameters["properties"]["query"]["description"] == "Full-text search query."
        # Returns is appended so the model knows what comes back
        assert "Returns: The top matching page as markdown." in td.description

    def test_summary_only_docstring_unchanged(self):
        def ping() -> str:
            """Check liveness."""
            return "ok"

        assert tool_from_function(ping).description == "Check liveness."

    def test_no_docstring_falls_back(self):
        def bare(x: int) -> int:
            return x

        assert tool_from_function(bare).description == "Call bare"

    def test_sections_after_args_are_dropped(self):
        def f(x: int) -> int:
            """Do a thing.

            Args:
                x: A number.

            Raises:
                ValueError: If x is negative.

            Examples:
                >>> f(1)
                1
            """
            return x

        td = tool_from_function(f)
        assert td.description == "Do a thing."
        assert td.parameters["properties"]["x"]["description"] == "A number."

    def test_description_truncated_to_limit(self):
        def f() -> None:
            """Summary line.

            {body}
            """

        f.__doc__ = "Summary line.\n\n" + ("word " * 500)
        td = tool_from_function(f, max_description_chars=200)
        assert len(td.description) <= 200
        assert td.description.startswith("Summary line.")
        assert td.description.endswith("…")

    def test_truncation_can_be_disabled(self):
        def f() -> None:
            pass

        f.__doc__ = "Summary line.\n\n" + ("word " * 500)
        td = tool_from_function(f, max_description_chars=None)
        assert len(td.description) > MAX_TOOL_DESCRIPTION_CHARS

    def test_prompt_format_indents_multiline_description(self):
        def f(x: int) -> int:
            """Do a thing.

            With a second paragraph.

            Args:
                x: A number.
            """
            return x

        text = tool_from_function(f).to_prompt_format()
        assert "  Description: Do a thing." in text
        assert "    With a second paragraph." in text
        # Parameters block still renders after the description
        assert text.index("  Parameters:") > text.index("With a second paragraph.")


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_lookup(self):
        reg = ToolRegistry()

        def my_tool(x: str) -> str:
            """A tool."""
            return x

        reg.register(my_tool)
        assert "my_tool" in reg
        assert len(reg) == 1
        assert reg.get("my_tool") is not None

    def test_decorator(self):
        reg = ToolRegistry()

        @reg.tool
        def decorated(a: int) -> int:
            """Decorated tool."""
            return a * 2

        assert "decorated" in reg
        # The original function is still callable
        assert decorated(3) == 6

    def test_execute(self):
        reg = ToolRegistry()

        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        reg.register(add)
        result = reg.execute("add", {"a": 3, "b": 4})
        assert result == 7

    def test_execute_missing_tool(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.execute("nope", {})

    def test_serialisation_formats(self):
        reg = ToolRegistry()

        def f(x: str) -> str:
            """Func."""
            return x

        reg.register(f)
        assert len(reg.to_openai_format()) == 1
        assert len(reg.to_anthropic_format()) == 1

    def test_bool(self):
        reg = ToolRegistry()
        assert not reg
        reg.register(lambda: None, name="test", description="test")
        assert reg


# ---------------------------------------------------------------------------
# MockDriver for tool use
# ---------------------------------------------------------------------------


class MockToolDriver(Driver):
    supports_messages = True
    supports_tool_use = True

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self._call_idx = 0

    def generate(self, prompt, options):
        return self._get_next()

    def generate_messages(self, messages, options):
        return self._get_next()

    def generate_messages_with_tools(self, messages, tools, options):
        return self._get_next()

    def _get_next(self):
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp


# ---------------------------------------------------------------------------
# Conversation tool use tests
# ---------------------------------------------------------------------------


class TestConversationToolUse:
    def test_ask_with_tools_single_round(self):
        """LLM calls a tool, then returns a final answer."""

        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        # Round 1: LLM requests tool call
        # Round 2: LLM returns final answer
        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                "tool_calls": [{"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "The weather in Paris is sunny.",
                "meta": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.002},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        reg = ToolRegistry()
        reg.register(get_weather)

        driver = MockToolDriver(responses)
        conv = Conversation(driver=driver, tools=reg)
        result = conv.ask("What's the weather in Paris?")

        assert result == "The weather in Paris is sunny."
        # Should have user, assistant (with tool_calls), tool result, assistant (final)
        assert len(conv.messages) >= 3

    def test_ask_without_tools_skips_loop(self):
        """When no tools are registered, use normal ask path."""

        class SimpleDriver(Driver):
            supports_messages = True

            def generate(self, prompt, options):
                return self.generate_messages([], options)

            def generate_messages(self, messages, options):
                return {
                    "text": "Hello!",
                    "meta": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "cost": 0.0},
                }

        conv = Conversation(driver=SimpleDriver())
        result = conv.ask("Hi")
        assert result == "Hello!"

    def test_max_tool_rounds_graceful_final_answer(self):
        """Exhausting max_tool_rounds yields one final no-tools answer instead of raising."""

        def noop() -> str:
            """Do nothing."""
            return "ok"

        # Every loop response has tool_calls; the final graceful call (no
        # tools) returns plain text.
        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
                "tool_calls": [{"id": f"call_{i}", "name": "noop", "arguments": {}}],
                "stop_reason": "tool_use",
            }
            for i in range(3)
        ]
        responses.append(
            {
                "text": "Final answer without tools.",
                "meta": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
            }
        )

        reg = ToolRegistry()
        reg.register(noop)

        driver = MockToolDriver(responses)
        conv = Conversation(driver=driver, tools=reg, max_tool_rounds=3)

        result = conv.ask("Do something")
        assert result == "Final answer without tools."
        assert conv.max_rounds_reached is True

    def test_register_tool_method(self):
        """Conversation.register_tool convenience method works."""

        class SimpleDriver(Driver):
            supports_messages = True
            supports_tool_use = True

            def generate(self, prompt, options):
                return self.generate_messages([], options)

            def generate_messages(self, messages, options):
                return {
                    "text": "ok",
                    "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
                }

            def generate_messages_with_tools(self, messages, tools, options):
                return {
                    "text": "done",
                    "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
                    "tool_calls": [],
                    "stop_reason": "end_turn",
                }

        conv = Conversation(driver=SimpleDriver())
        conv.register_tool(lambda x: x, name="echo", description="Echo")
        assert "echo" in conv._tools

    def test_tool_execution_error_handled(self):
        """Tool execution errors are returned as error strings."""

        def fail_tool() -> str:
            """Always fails."""
            raise ValueError("broken")

        responses = [
            {
                "text": "",
                "meta": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
                "tool_calls": [{"id": "call_err", "name": "fail_tool", "arguments": {}}],
                "stop_reason": "tool_use",
            },
            {
                "text": "Tool failed, sorry.",
                "meta": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": 0.0},
                "tool_calls": [],
                "stop_reason": "end_turn",
            },
        ]

        reg = ToolRegistry()
        reg.register(fail_tool)

        driver = MockToolDriver(responses)
        conv = Conversation(driver=driver, tools=reg)
        result = conv.ask("Call the tool")

        assert result == "Tool failed, sorry."
        # Check the tool result message contains the error
        tool_msgs = [m for m in conv.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert "Error" in tool_msgs[0]["content"]
