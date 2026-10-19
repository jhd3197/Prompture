"""Long Claude requests travel over the streaming API.

A non-streaming ``messages.create()`` sends no bytes until the whole reply is ready. A long
generation (a stylesheet, a full HTML page, a long JSON extraction) can take minutes, and a
network that closes a silent connection after 60 s drops it: the SDK retries, each attempt dies
at the same 60 s, and the caller gets ``APIConnectionError`` after three minutes. The SDK also
refuses non-streaming requests whose ``max_tokens`` could run past ten minutes.

Above ``STREAM_TRANSPORT_ABOVE_MAX_TOKENS`` the drivers call ``messages.stream()`` and collect
``get_final_message()``, the same ``Message`` object ``create()`` returns, so usage, content
and stop reason are read exactly as before. Short requests keep ``create()``; the
``stream_transport`` option forces either.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from prompture.drivers.claude_driver import (
    STREAM_TRANSPORT_ABOVE_MAX_TOKENS,
    _acreate_message,
    _create_message,
    _stream_transport,
)


class _Usage:
    input_tokens = 10
    output_tokens = 5
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _Block:
    type = "text"
    text = "ok"


class _Message:
    def __init__(self) -> None:
        self.usage = _Usage()
        self.content = [_Block()]
        self.stop_reason = "end_turn"

    def keys(self) -> list[str]:
        return []

    def __getitem__(self, _: str) -> Any:
        raise KeyError


class _Stream:
    def __init__(self, sink: dict) -> None:
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def get_final_message(self) -> _Message:
        self.sink["final"] = True
        return _Message()


class _AsyncStream(_Stream):
    async def get_final_message(self) -> _Message:  # type: ignore[override]
        self.sink["final"] = True
        return _Message()


class _Messages:
    def __init__(self, *, is_async: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.sink: dict = {}
        self._async = is_async

    def create(self, **kwargs: Any):
        self.calls.append(("create", kwargs))
        if self._async:
            async def _done():
                return _Message()
            return _done()
        return _Message()

    def stream(self, **kwargs: Any):
        assert "stream" not in kwargs  # stream() takes create()'s kwargs minus `stream`
        self.calls.append(("stream", kwargs))
        return _AsyncStream(self.sink) if self._async else _Stream(self.sink)


class _Client:
    def __init__(self, *, is_async: bool = False) -> None:
        self.messages = _Messages(is_async=is_async)


def test_the_threshold_and_the_override() -> None:
    long = {"max_tokens": STREAM_TRANSPORT_ABOVE_MAX_TOKENS + 1}
    short = {"max_tokens": STREAM_TRANSPORT_ABOVE_MAX_TOKENS}
    assert _stream_transport(long, {}) and not _stream_transport(short, {})
    assert not _stream_transport(long, {"stream_transport": False})
    assert _stream_transport(short, {"stream_transport": True})


def test_a_short_request_keeps_create() -> None:
    client = _Client()
    _create_message(client, {"model": "m", "max_tokens": 512, "messages": []}, {})
    assert [c[0] for c in client.messages.calls] == ["create"]


def test_a_long_request_streams_and_returns_the_final_message() -> None:
    client = _Client()
    msg = _create_message(client, {"model": "m", "max_tokens": 16000, "messages": []}, {})
    assert [c[0] for c in client.messages.calls] == ["stream"]
    assert client.messages.sink["final"] and msg.content[0].text == "ok"


def test_the_async_helper_does_the_same() -> None:
    client = _Client(is_async=True)
    asyncio.run(_acreate_message(client, {"model": "m", "max_tokens": 512, "messages": []}, {}))
    msg = asyncio.run(_acreate_message(client, {"model": "m", "max_tokens": 16000, "messages": []}, {}))
    assert [c[0] for c in client.messages.calls] == ["create", "stream"]
    assert msg.stop_reason == "end_turn"


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch):
    from prompture.drivers import claude_driver as cd

    client = _Client()

    class _StubAnthropic:
        def __new__(cls, *_: Any, **__: Any) -> _Client:  # type: ignore[misc]
            return client

    monkeypatch.setattr(cd, "anthropic", type("M", (), {"Anthropic": _StubAnthropic}))
    return cd.ClaudeDriver(api_key="test-key", model="claude-sonnet-4-6"), client


def test_the_driver_streams_a_long_generation(driver) -> None:
    d, client = driver
    out = d.generate_messages([{"role": "user", "content": "a long page"}], options={"max_tokens": 16000})
    assert out["text"] == "ok"
    kind, kwargs = client.messages.calls[-1]
    assert kind == "stream" and kwargs["max_tokens"] == 16000


def test_the_driver_keeps_create_for_a_short_one(driver) -> None:
    d, client = driver
    d.generate_messages([{"role": "user", "content": "hi"}], options={"max_tokens": 256})
    assert client.messages.calls[-1][0] == "create"
