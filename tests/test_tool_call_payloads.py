"""Tool-call requests carry images and tool schemas every provider accepts."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from typing import Any

import pytest

from prompture.agents.tools_schema import ToolDefinition
from prompture.infra.tool_schema import flatten_tool_definitions, flatten_tool_schema
from prompture.media.image import make_image

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

ONE_OF = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
    "oneOf": [
        {"title": "From text", "properties": {"duration": {"type": "integer"}}, "required": ["prompt", "duration"]},
        {"title": "From image", "properties": {"image_url": {"type": "string"}}, "required": ["prompt", "image_url"]},
    ],
}


class TestFlattenToolSchema:
    def test_plain_object_is_returned_unchanged(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        assert flatten_tool_schema(schema) is schema

    def test_top_level_one_of_becomes_one_object(self):
        flat = flatten_tool_schema(ONE_OF)
        assert flat["type"] == "object"
        assert not {"oneOf", "anyOf", "allOf"} & flat.keys()
        assert set(flat["properties"]) == {"prompt", "duration", "image_url"}
        # Only what every alternative needs stays required.
        assert flat["required"] == ["prompt"]
        assert "From text (prompt, duration)" in flat["description"]
        assert "From image (prompt, image_url)" in flat["description"]

    def test_input_is_not_mutated_and_result_is_stable(self):
        before = json.dumps(ONE_OF, sort_keys=True)
        flat = flatten_tool_schema(ONE_OF)
        assert json.dumps(ONE_OF, sort_keys=True) == before
        assert flatten_tool_schema(flat) is flat

    def test_all_of_keeps_every_required_field(self):
        schema = {
            "allOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
                {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]},
            ]
        }
        flat = flatten_tool_schema(schema)
        assert flat["type"] == "object"
        assert set(flat["properties"]) == {"a", "b"}
        assert flat["required"] == ["a", "b"]

    def test_any_of_branches_by_reference(self):
        schema = {
            "anyOf": [{"$ref": "#/$defs/ByUrl"}, {"$ref": "#/$defs/ById"}],
            "$defs": {
                "ByUrl": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                "ById": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            },
        }
        flat = flatten_tool_schema(schema)
        assert set(flat["properties"]) == {"url", "id"}
        assert "required" not in flat
        assert "$defs" in flat  # nested references keep resolving

    def test_required_only_branches_are_described(self):
        schema = {
            "type": "object",
            "properties": {"url": {"type": "string"}, "file_id": {"type": "string"}},
            "oneOf": [{"required": ["url"]}, {"required": ["file_id"]}],
        }
        flat = flatten_tool_schema(schema)
        assert "required" not in flat
        assert "(url)" in flat["description"] and "(file_id)" in flat["description"]

    def test_missing_type_and_non_objects(self):
        assert flatten_tool_schema({"properties": {"x": {"type": "string"}}})["type"] == "object"
        assert flatten_tool_schema({"type": "string"}) == {"type": "object", "properties": {}}
        assert flatten_tool_schema(None) == {"type": "object", "properties": {}}

    def test_nested_combinators_are_left_alone(self):
        schema = {
            "type": "object",
            "properties": {"size": {"oneOf": [{"type": "integer"}, {"type": "string"}]}},
        }
        assert flatten_tool_schema(schema) is schema

    def test_tool_definition_serialisers_flatten(self):
        td = ToolDefinition(name="make_video", description="Make a video", parameters=ONE_OF, function=lambda **_: "")
        params = td.to_openai_format()["function"]["parameters"]
        assert "oneOf" not in params and params["type"] == "object"
        strict = td.to_openai_format(strict=True)["function"]["parameters"]
        assert "oneOf" not in strict and strict["additionalProperties"] is False
        assert "oneOf" not in td.to_anthropic_format()["input_schema"]

    @pytest.mark.parametrize(
        "module,func",
        [
            ("prompture.drivers.claude_driver", "_convert_tools_to_anthropic"),
            ("prompture.drivers.bedrock_driver", "_convert_tools_to_anthropic_bedrock"),
        ],
    )
    def test_anthropic_converters_flatten_raw_tools(self, module, func):
        convert = getattr(importlib.import_module(module), func)
        openai_shape = {"type": "function", "function": {"name": "a", "description": "", "parameters": ONE_OF}}
        anthropic_shape = {"name": "b", "description": "", "input_schema": ONE_OF}
        for tool in convert([openai_shape, anthropic_shape]):
            assert "oneOf" not in tool["input_schema"]

    def test_flatten_tool_definitions_handles_both_shapes(self):
        tools = flatten_tool_definitions(
            [
                {"type": "function", "function": {"name": "a", "parameters": ONE_OF}},
                {"name": "b", "input_schema": ONE_OF},
                {"type": "web_search"},
            ]
        )
        assert "oneOf" not in tools[0]["function"]["parameters"]
        assert "oneOf" not in tools[1]["input_schema"]
        assert tools[2] == {"type": "web_search"}


# ---------------------------------------------------------------------------
# Images on tool-call turns
# ---------------------------------------------------------------------------

#: Every driver that converts attached images in ``_prepare_messages``.
DRIVERS = [
    ("openai_driver", "OpenAIDriver"),
    ("async_openai_driver", "AsyncOpenAIDriver"),
    ("openrouter_driver", "OpenRouterDriver"),
    ("async_openrouter_driver", "AsyncOpenRouterDriver"),
    ("groq_driver", "GroqDriver"),
    ("async_groq_driver", "AsyncGroqDriver"),
    ("grok_driver", "GrokDriver"),
    ("async_grok_driver", "AsyncGrokDriver"),
    ("moonshot_driver", "MoonshotDriver"),
    ("async_moonshot_driver", "AsyncMoonshotDriver"),
    ("zai_driver", "ZaiDriver"),
    ("async_zai_driver", "AsyncZaiDriver"),
    ("mistral_driver", "MistralDriver"),
    ("async_mistral_driver", "AsyncMistralDriver"),
    ("azure_driver", "AzureDriver"),
    ("async_azure_driver", "AsyncAzureDriver"),
    ("ollama_driver", "OllamaDriver"),
    ("async_ollama_driver", "AsyncOllamaDriver"),
    ("cachibot_driver", "CachiBotDriver"),
    ("async_cachibot_driver", "AsyncCachiBotDriver"),
    ("claude_driver", "ClaudeDriver"),
    ("async_claude_driver", "AsyncClaudeDriver"),
]


class _Prepared(Exception):
    """Raised by the stubbed ``_prepare_messages``."""


class _Trap:
    """A provider client that fails the test if a request goes out unprepared."""

    def __getattr__(self, name: str) -> _Trap:
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("request sent before its messages were prepared")


def _drive(result: Any) -> None:
    """Run whatever a tools method returned until it does real work."""
    if inspect.isawaitable(result):
        asyncio.run(result)
    elif inspect.isasyncgen(result):

        async def _first() -> None:
            async for _ in result:
                break

        asyncio.run(_first())
    elif inspect.isgenerator(result):
        next(result)


@pytest.mark.parametrize("method", ["generate_messages_with_tools", "generate_messages_with_tools_stream"])
@pytest.mark.parametrize("module,cls_name", DRIVERS)
def test_tool_calls_prepare_attached_images(module, cls_name, method):
    """A tool-call turn converts attached images before the request is sent."""
    try:
        cls = getattr(importlib.import_module(f"prompture.drivers.{module}"), cls_name)
    except ImportError as exc:  # pragma: no cover — optional provider SDK missing
        pytest.skip(str(exc))
    if method not in vars(cls):
        pytest.skip(f"{cls_name} inherits {method}")

    driver = object.__new__(cls)
    driver.model = "test-model"
    driver.client = _Trap()
    driver.api_key = "test"

    def _stub(_messages):
        raise _Prepared

    driver._prepare_messages = _stub
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    with pytest.raises(_Prepared):
        _drive(getattr(driver, method)(messages, [], {"model": "test-model"}))


def test_openai_tool_call_with_an_image_serialises():
    """The Chat Completions tool path sends image_url blocks, not ImageContent."""
    from prompture.drivers.async_openai_driver import AsyncOpenAIDriver

    sent: dict[str, Any] = {}

    class _Completions:
        async def create(self, **kwargs):
            sent.update(kwargs)
            raise _Prepared

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    driver = AsyncOpenAIDriver(api_key="test", model="gpt-4o-mini")
    driver.client = _Client()
    image = make_image("data:image/png;base64,iVBORw0KGgo=")
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image", "source": image}]}]
    tools = [{"type": "function", "function": {"name": "a", "description": "", "parameters": {"type": "object"}}}]

    with pytest.raises(_Prepared):
        asyncio.run(driver.generate_messages_with_tools(messages, tools, {"api": "chat_completions"}))

    json.dumps(sent["messages"])  # would raise on a raw ImageContent
    assert sent["messages"][0]["content"][1]["type"] == "image_url"
