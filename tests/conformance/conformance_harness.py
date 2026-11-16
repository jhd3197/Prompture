"""Fake transports and canned replies for the driver conformance suite.

The suite runs one set of scenarios against every tool-capable driver, so a
provider quirk that one driver handles and its sibling forgets shows up as a
failing test instead of a bug report. Nothing here touches the network: each
transport a driver can use (``requests``, ``httpx``, an OpenAI-style SDK
client, an Anthropic SDK client, boto3) is replaced by a fake that records the
outgoing request and serves a canned reply.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import io
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Canned inputs
# ---------------------------------------------------------------------------

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

ONE_OF_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "make_video",
        "description": "Make a video",
        "parameters": {
            "oneOf": [
                {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
                {"type": "object", "properties": {"image_url": {"type": "string"}}, "required": ["image_url"]},
            ]
        },
    },
}

#: A conversation that already went one tool round, in the canonical
#: (OpenAI chat) shape the agent loop and the gateway both produce.
HISTORY: list[dict[str, Any]] = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_prev",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_prev", "content": "18C and sunny"},
    {"role": "user", "content": "And in Rome?"},
]

SIMPLE: list[dict[str, Any]] = [{"role": "user", "content": "Weather in Paris?"}]


def image_messages() -> list[dict[str, Any]]:
    from prompture.media.image import make_image

    image = make_image("data:image/png;base64,iVBORw0KGgo=")
    return [
        {"role": "user", "content": [{"type": "text", "text": "What is this?"}, {"type": "image", "source": image}]}
    ]


# ---------------------------------------------------------------------------
# Canned replies
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    """One assistant turn, independent of wire format."""

    text: str = ""
    #: ``(id, name, arguments)``; arguments is a JSON string or a dict.
    tool_calls: list[tuple[str | None, str, Any]] = field(default_factory=list)
    finish: str | None = None
    reasoning: str | None = None
    prompt_tokens: int = 11
    completion_tokens: int = 7

    def openai_finish(self) -> str:
        return self.finish or ("tool_calls" if self.tool_calls else "stop")

    def to_openai(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.reasoning is not None:
            message["reasoning_content"] = self.reasoning
        if self.tool_calls:
            calls = []
            for call_id, name, args in self.tool_calls:
                call: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": args}}
                if call_id is not None:
                    call["id"] = call_id
                calls.append(call)
            message["tool_calls"] = calls
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [{"index": 0, "message": message, "finish_reason": self.openai_finish()}],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        }

    def to_openai_chunks(self) -> list[dict[str, Any]]:
        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        chunks: list[dict[str, Any]] = []
        if self.reasoning:
            chunks.append(chunk({"reasoning_content": self.reasoning}))
        if self.text:
            chunks.append(chunk({"content": self.text}))
        for index, (call_id, name, args) in enumerate(self.tool_calls):
            raw = args if isinstance(args, str) else json.dumps(args)
            head: dict[str, Any] = {"index": index, "type": "function", "function": {"name": name, "arguments": ""}}
            if call_id is not None:
                head["id"] = call_id
            chunks.append(chunk({"tool_calls": [head]}))
            half = max(1, len(raw) // 2)
            for piece in (raw[:half], raw[half:]):
                if piece:
                    chunks.append(chunk({"tool_calls": [{"index": index, "function": {"arguments": piece}}]}))
        chunks.append(chunk({}, self.openai_finish()))
        chunks.append(
            {
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.prompt_tokens + self.completion_tokens,
                },
            }
        )
        return chunks

    def to_anthropic(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if self.reasoning:
            content.append({"type": "thinking", "thinking": self.reasoning})
        if self.text:
            content.append({"type": "text", "text": self.text})
        for call_id, name, args in self.tool_calls:
            content.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": content,
            "stop_reason": self.finish or ("tool_use" if self.tool_calls else "end_turn"),
            "usage": {"input_tokens": self.prompt_tokens, "output_tokens": self.completion_tokens},
        }


class Obj:
    """An SDK response object: attribute access, absent fields read as None."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        for key, value in data.items():
            setattr(self, key, _wrap(value))

    def __getattr__(self, name: str) -> Any:
        # ``keys`` stays absent so ``dict(obj)`` iterates pairs, as pydantic does.
        if name.startswith("_") or name == "keys":
            raise AttributeError(name)
        return None

    def __iter__(self):
        return iter(self._data.items())

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self._data

    def to_dict(self, **_: Any) -> dict[str, Any]:
        return self._data


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Obj(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# The wire: records requests, serves the reply
# ---------------------------------------------------------------------------


class Wire:
    def __init__(self, reply: Reply):
        self.reply = reply
        self.requests: list[dict[str, Any]] = []

    @property
    def payload(self) -> dict[str, Any]:
        assert self.requests, "the driver sent no request"
        return self.requests[-1]["payload"]

    def record(self, payload: dict[str, Any], **extra: Any) -> None:
        # Round-trip through JSON: a payload the transport could not
        # serialize (a raw ImageContent, say) fails here, as it would live.
        self.requests.append({"payload": json.loads(json.dumps(payload)), **extra})

    def sse_lines(self) -> list[str]:
        return [f"data: {json.dumps(c)}" for c in self.reply.to_openai_chunks()] + ["data: [DONE]"]


class _HTTPResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, wire: Wire, stream: bool):
        self._wire = wire
        self._stream = stream
        self.text = "" if stream else json.dumps(wire.reply.to_openai())

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._wire.reply.to_openai()

    def iter_lines(self, decode_unicode: bool = False, **_: Any):
        yield from self._wire.sse_lines()

    async def aiter_lines(self):
        for line in self._wire.sse_lines():
            yield line

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


def install_requests(monkeypatch: pytest.MonkeyPatch, wire: Wire) -> None:
    import requests

    def post(url: str, headers: Any = None, json: Any = None, stream: bool = False, **kwargs: Any):
        wire.record(json, url=url, headers=headers, stream=bool(stream or (json or {}).get("stream")))
        return _HTTPResponse(wire, stream=bool(stream))

    monkeypatch.setattr(requests, "post", post)


def install_httpx(monkeypatch: pytest.MonkeyPatch, wire: Wire) -> None:
    import httpx

    class Client:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def aclose(self) -> None:
            return None

        async def post(self, url: str, headers: Any = None, json: Any = None, **kwargs: Any):
            wire.record(json, url=url, headers=headers, stream=bool((json or {}).get("stream")))
            return _HTTPResponse(wire, stream=False)

        def stream(self, method: str, url: str, headers: Any = None, json: Any = None, **kwargs: Any):
            wire.record(json, url=url, headers=headers, stream=True)
            return _HTTPResponse(wire, stream=True)

    monkeypatch.setattr(httpx, "AsyncClient", Client)


class _SyncStream:
    response = None

    def __init__(self, chunks: list[dict[str, Any]]):
        self._chunks = [Obj(c) for c in chunks]

    def __iter__(self):
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _AsyncStream:
    response = None

    def __init__(self, chunks: list[dict[str, Any]]):
        self._chunks = [Obj(c) for c in chunks]

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield chunk


def openai_sdk_client(wire: Wire, *, is_async: bool) -> Any:
    """A stand-in for ``OpenAI()`` / ``AsyncOpenAI()`` / ``Groq()``."""

    def create(**kwargs: Any) -> Any:
        wire.record(kwargs, stream=bool(kwargs.get("stream")))
        if kwargs.get("stream"):
            chunks = wire.reply.to_openai_chunks()
            return _AsyncStream(chunks) if is_async else _SyncStream(chunks)
        return Obj(wire.reply.to_openai())

    async def acreate(**kwargs: Any) -> Any:
        return create(**kwargs)

    completions = type("Completions", (), {"create": staticmethod(acreate if is_async else create)})()
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


def anthropic_sdk_client(wire: Wire, *, is_async: bool) -> Any:
    """A stand-in for ``anthropic.Anthropic()`` and friends."""

    def create(**kwargs: Any) -> Any:
        wire.record(kwargs, stream=False)
        return Obj(wire.reply.to_anthropic())

    async def acreate(**kwargs: Any) -> Any:
        return create(**kwargs)

    messages = type("Messages", (), {"create": staticmethod(acreate if is_async else create)})()
    return type("Client", (), {"messages": messages})()


def bedrock_client(wire: Wire) -> Any:
    def invoke_model(modelId: str, body: str, **kwargs: Any) -> dict[str, Any]:
        wire.record(json.loads(body), model=modelId, stream=False)
        return {"body": io.BytesIO(json.dumps(wire.reply.to_anthropic()).encode()), "ResponseMetadata": {}}

    return type("Bedrock", (), {"invoke_model": staticmethod(invoke_model)})()


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass
class Target:
    id: str
    module: str
    cls: str
    transport: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    #: What a normalized ``tool_choice="required"`` becomes on this provider.
    required_choice: Any = "required"

    @property
    def is_async(self) -> bool:
        return self.cls.startswith("Async")

    def load(self) -> type:
        try:
            return getattr(importlib.import_module(f"prompture.drivers.{self.module}"), self.cls)
        except ImportError as exc:  # pragma: no cover — optional SDK missing
            pytest.skip(str(exc))

    def build(self, monkeypatch: pytest.MonkeyPatch, wire: Wire) -> Any:
        cls = self.load()
        # Module-level transports are replaced before the driver exists, so a
        # client built in ``__init__`` is already the fake.
        if self.transport == "requests":
            install_requests(monkeypatch, wire)
        elif self.transport == "httpx":
            install_httpx(monkeypatch, wire)
        elif self.transport == "anthropic_sdk":
            import anthropic

            fake = anthropic_sdk_client(wire, is_async=self.is_async)
            monkeypatch.setattr(anthropic, "AsyncAnthropic" if self.is_async else "Anthropic", lambda **kw: fake)
        try:
            driver = cls(**self.kwargs)
        except (ImportError, RuntimeError) as exc:  # pragma: no cover — optional SDK missing
            pytest.skip(f"{self.cls} cannot be constructed here: {exc}")
        if self.transport == "openai_sdk":
            driver.client = openai_sdk_client(wire, is_async=self.is_async)
        elif self.transport == "azure_openai":
            client = openai_sdk_client(wire, is_async=self.is_async)
            driver._get_openai_client = lambda config: client
        elif self.transport == "azure_claude":
            client = anthropic_sdk_client(wire, is_async=self.is_async)
            driver._get_anthropic_client = lambda config: client
        elif self.transport == "vertex_claude":
            driver._claude_client = anthropic_sdk_client(wire, is_async=self.is_async)
        elif self.transport == "bedrock":
            (driver._sync if self.is_async else driver)._client = bedrock_client(wire)
        elif self.transport not in ("requests", "httpx", "anthropic_sdk"):  # pragma: no cover
            raise AssertionError(f"unknown transport {self.transport}")
        return driver


def _pair(name: str, cls: str, transport: tuple[str, str], **kw: Any) -> list[Target]:
    return [
        Target(id=name, module=f"{name}_driver", cls=cls, transport=transport[0], **kw),
        Target(id=f"{name}-async", module=f"async_{name}_driver", cls=f"Async{cls}", transport=transport[1], **kw),
    ]


_HTTP = ("requests", "httpx")
_SDK = ("openai_sdk", "openai_sdk")
_KEY = {"api_key": "test-key"}

#: Every driver that speaks the OpenAI chat-completions wire format.
OPENAI_WIRE: list[Target] = [
    *_pair(
        "openai", "OpenAIDriver", _SDK, kwargs={**_KEY, "model": "gpt-4o-mini"}, options={"api": "chat_completions"}
    ),
    *_pair("groq", "GroqDriver", _SDK, kwargs=_KEY),
    *_pair("grok", "GrokDriver", _HTTP, kwargs=_KEY),
    *_pair("openrouter", "OpenRouterDriver", _HTTP, kwargs=_KEY),
    *_pair("mistral", "MistralDriver", _HTTP, kwargs=_KEY, required_choice="any"),
    *_pair("moonshot", "MoonshotDriver", _HTTP, kwargs=_KEY, required_choice="auto"),
    *_pair("zai", "ZaiDriver", _HTTP, kwargs=_KEY),
    *_pair("modelscope", "ModelScopeDriver", _HTTP, kwargs=_KEY),
    *_pair("cachibot", "CachiBotDriver", _HTTP, kwargs=_KEY),
    *_pair("minimax", "MiniMaxDriver", _HTTP, kwargs=_KEY),
    *_pair(
        "openai_compatible",
        "OpenAICompatibleDriver",
        _HTTP,
        kwargs={**_KEY, "endpoint": "http://compat.test/v1", "model": "test-model"},
    ),
    *_pair(
        "azure",
        "AzureDriver",
        ("azure_openai", "azure_openai"),
        kwargs={**_KEY, "endpoint": "https://azure.test", "deployment_id": "dep", "model": "gpt-4o-mini"},
    ),
]

#: Every driver that speaks the Anthropic Messages wire format.
ANTHROPIC_WIRE: list[Target] = [
    *_pair("claude", "ClaudeDriver", ("anthropic_sdk", "anthropic_sdk"), kwargs=_KEY),
    Target(
        id="azure-claude",
        module="azure_driver",
        cls="AzureDriver",
        transport="azure_claude",
        kwargs={**_KEY, "endpoint": "https://azure.test", "model": "claude-sonnet-4-5"},
    ),
    Target(
        id="azure-claude-async",
        module="async_azure_driver",
        cls="AsyncAzureDriver",
        transport="azure_claude",
        kwargs={**_KEY, "endpoint": "https://azure.test", "model": "claude-sonnet-4-5"},
    ),
    Target(
        id="vertex-claude",
        module="google_vertexai_driver",
        cls="GoogleVertexAIDriver",
        transport="vertex_claude",
        kwargs={"project_id": "proj", "model": "claude-sonnet-4-5"},
    ),
    Target(
        id="vertex-claude-async",
        module="async_google_vertexai_driver",
        cls="AsyncGoogleVertexAIDriver",
        transport="vertex_claude",
        kwargs={"project_id": "proj", "model": "claude-sonnet-4-5"},
    ),
    Target(
        id="bedrock",
        module="bedrock_driver",
        cls="BedrockDriver",
        transport="bedrock",
        kwargs={"aws_access_key_id": "k", "aws_secret_access_key": "s", "aws_region": "us-east-1"},
    ),
    Target(
        id="bedrock-async",
        module="async_bedrock_driver",
        cls="AsyncBedrockDriver",
        transport="bedrock",
        kwargs={"aws_access_key_id": "k", "aws_secret_access_key": "s", "aws_region": "us-east-1"},
    ),
]


# ---------------------------------------------------------------------------
# Running a driver method to completion, sync or async
# ---------------------------------------------------------------------------


def run(value: Any) -> Any:
    """Resolve whatever a driver method returned: a value, or a list of events."""
    if inspect.isawaitable(value):
        return asyncio.run(_await(value))
    if inspect.isasyncgen(value):
        return asyncio.run(_collect(value))
    if inspect.isgenerator(value):
        return list(value)
    return value


async def _await(value: Any) -> Any:
    return await value


async def _collect(agen: Any) -> list[Any]:
    return [item async for item in agen]


def call_tools(target: Target, driver: Any, messages: Any, tools: Any, options: dict[str, Any] | None = None) -> Any:
    return run(driver.generate_messages_with_tools(messages, tools, {**target.options, **(options or {})}))


def stream_tools(target: Target, driver: Any, messages: Any, tools: Any, options: dict[str, Any] | None = None) -> Any:
    return run(driver.generate_messages_with_tools_stream(messages, tools, {**target.options, **(options or {})}))
