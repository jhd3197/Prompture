"""Strict tool use, and refusing to run a tool on arguments that did not parse."""

from __future__ import annotations

from typing import Any

from prompture.agents.conversation import Conversation
from prompture.agents.live_events import MessageStop, TextDelta, ToolUseStart, ToolUseStop
from prompture.agents.tools_schema import ToolRegistry
from prompture.drivers._anthropic_wire import strict_input_schema, to_anthropic_tools
from prompture.drivers._openai_chat import ChatProfile, get_chat_profile, prepare_tools, provider_tool_choice
from prompture.drivers.base import Driver

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "default": "c"},
                "days": {"type": "integer", "minimum": 1, "maximum": 7},
            },
            "required": ["city"],
        },
    },
}


class TestStrictTools:
    def test_openai_strict_requires_everything_and_makes_optionals_nullable(self):
        tool = prepare_tools([WEATHER], get_chat_profile("openai"), {"strict_tools": True})[0]["function"]

        assert tool["strict"] is True
        params = tool["parameters"]
        assert params["additionalProperties"] is False
        assert set(params["required"]) == {"city", "units", "days"}
        assert params["properties"]["city"]["type"] == "string"
        assert "null" in params["properties"]["units"]["type"]

    def test_strict_is_opt_in(self):
        tool = prepare_tools([WEATHER], get_chat_profile("openai"), {})[0]
        assert tool == WEATHER

    def test_providers_without_strict_mode_get_the_plain_schema(self):
        tool = prepare_tools([WEATHER], get_chat_profile("zai"), {"strict_tools": True})[0]
        assert "strict" not in tool["function"]

    def test_the_callers_tools_are_not_mutated(self):
        before = repr(WEATHER)
        prepare_tools([WEATHER], get_chat_profile("openai"), {"strict_tools": True})
        prepare_tools([WEATHER], get_chat_profile("moonshot"), {})
        assert repr(WEATHER) == before

    def test_anthropic_strict_closes_objects_and_moves_unsupported_constraints(self):
        tool = to_anthropic_tools([WEATHER], strict=True)[0]

        assert tool["strict"] is True
        schema = tool["input_schema"]
        assert schema["additionalProperties"] is False
        days = schema["properties"]["days"]
        assert "minimum" not in days and "maximum" not in days
        assert "minimum=1" in days["description"] and "maximum=7" in days["description"]

    def test_anthropic_strict_reaches_nested_objects(self):
        schema = strict_input_schema(
            {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "minItems": 3, "items": {"type": "object", "properties": {}}}
                },
            }
        )
        assert schema["properties"]["items"]["minItems"] == 1
        assert schema["properties"]["items"]["items"]["additionalProperties"] is False


class TestProfiles:
    def test_required_tool_choice_is_named_per_provider(self):
        assert provider_tool_choice("required", get_chat_profile("openai")) == "required"
        assert provider_tool_choice("required", get_chat_profile("mistral")) == "any"
        assert provider_tool_choice("required", get_chat_profile("moonshot")) == "auto"

    def test_a_provider_that_cannot_name_a_tool_falls_back(self):
        profile = ChatProfile("x", named_tool_choice=False, required_tool_choice="any")
        assert provider_tool_choice({"name": "get_weather"}, profile) == "any"

    def test_unknown_provider_behaves_like_openai(self):
        assert get_chat_profile("some-new-host").required_tool_choice == "required"


class TestNullOptionals:
    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        @registry.tool
        def get_weather(city: str, units: str = "c") -> str:
            """Look up the weather."""
            return f"{city}:{units}"

        return registry

    def test_null_for_an_optional_parameter_uses_the_default(self):
        assert self._registry().execute("get_weather", {"city": "Paris", "units": None}) == "Paris:c"

    def test_a_given_value_is_kept(self):
        assert self._registry().execute("get_weather", {"city": "Paris", "units": "f"}) == "Paris:f"


class _MalformedThenAnswer(Driver):
    """First turn: a tool call whose arguments did not parse. Second: an answer."""

    supports_messages = True
    supports_tool_use = True
    supports_streaming_tool_use = True
    model = "fake"

    def __init__(self) -> None:
        self.turns: list[list[dict[str, Any]]] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("not used")

    def generate_messages_with_tools(self, messages, tools, options):  # pragma: no cover
        raise AssertionError("ask_live streams")

    def generate_messages_with_tools_stream(self, messages, tools, options):
        self.turns.append(list(messages))
        if len(self.turns) == 1:
            yield ToolUseStart(id="call_1", name="get_weather")
            yield ToolUseStop(
                id="call_1",
                name="get_weather",
                input={},
                arguments_error="Failed to parse tool arguments for get_weather as JSON",
            )
            yield MessageStop(stop_reason="tool_use", usage={})
        else:
            yield TextDelta(text="done")
            yield MessageStop(stop_reason="end_turn", usage={})


def test_streamed_malformed_arguments_are_not_executed():
    calls: list[dict[str, Any]] = []
    registry = ToolRegistry()

    @registry.tool
    def get_weather(city: str = "nowhere") -> str:
        """Look up the weather."""
        calls.append({"city": city})
        return "sunny"

    driver = _MalformedThenAnswer()
    conv = Conversation(driver=driver, tools=registry)
    events = list(conv.ask_live("Weather in Paris?"))

    assert calls == [], "the tool ran on arguments that never parsed"
    result = next(e for e in events if e.event_type == "tool_result")
    assert result.is_error
    assert "could not be parsed" in result.output
    # The error went back to the model, which then answered.
    assert len(driver.turns) == 2
    assert driver.turns[1][-1]["role"] == "tool"
