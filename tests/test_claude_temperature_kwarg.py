"""`temperature` reaches Anthropic through the body, never as a kwarg.

``anthropic`` 1.0.0 (2026-08-20) dropped ``temperature`` / ``top_p`` / ``top_k``
from ``messages.create()``. Every driver call that passed one died with

    TypeError: AsyncMessages.create() got an unexpected keyword argument 'temperature'

*before* the HTTP request — so a host that let the SDK float took its whole
background AI lane down on the next rebuild and, because the exception fired
ahead of any usage recording, left no trace of it anywhere.

The stub client below is the 1.x contract: it refuses the kwarg the way the real
SDK does. These tests fail on the old code path and pass on ``extra_body``.
"""

from __future__ import annotations

from typing import Any

import pytest

from prompture.drivers.claude_driver import _apply_temperature


class _StubUsage:
    input_tokens = 10
    output_tokens = 5
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _StubBlock:
    def __init__(self) -> None:
        self.type = "text"
        self.text = "ok"


class _StubResponse:
    def __init__(self) -> None:
        self.usage = _StubUsage()
        self.content = [_StubBlock()]
        self.stop_reason = "end_turn"

    def keys(self) -> list[str]:
        return []

    def __getitem__(self, _: str) -> Any:
        raise KeyError


class _StrictMessages:
    """``messages.create`` as anthropic 1.x defines it: no sampling kwargs."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _StubResponse:
        for dropped in ("temperature", "top_p", "top_k"):
            if dropped in kwargs:
                raise TypeError(
                    f"Messages.create() got an unexpected keyword argument '{dropped}'"
                )
        self.last_kwargs = kwargs
        return _StubResponse()


class _StrictClient:
    def __init__(self) -> None:
        self.messages = _StrictMessages()


@pytest.fixture
def strict_driver(monkeypatch: pytest.MonkeyPatch):
    """A ClaudeDriver talking to a stub that behaves like anthropic 1.x."""
    from prompture.drivers import claude_driver as cd

    client = _StrictClient()

    class _StubAnthropic:
        def __new__(cls, *_: Any, **__: Any) -> _StrictClient:  # type: ignore[misc]
            return client

    monkeypatch.setattr(cd, "anthropic", type("M", (), {"Anthropic": _StubAnthropic}))
    return cd.ClaudeDriver(api_key="test-key", model="claude-sonnet-4-6"), client


def test_a_temperature_model_still_gets_one_on_anthropic_1x(strict_driver) -> None:
    driver, client = strict_driver
    driver.generate_messages([{"role": "user", "content": "hi"}], options={"temperature": 0.4})

    kw = client.messages.last_kwargs
    assert "temperature" not in kw
    assert kw["extra_body"]["temperature"] == 0.4


def test_the_default_temperature_travels_the_same_way(strict_driver) -> None:
    driver, client = strict_driver
    driver.generate_messages([{"role": "user", "content": "hi"}], options={})
    assert client.messages.last_kwargs["extra_body"]["temperature"] == 0.0


def test_json_mode_takes_the_same_path(strict_driver) -> None:
    driver, client = strict_driver
    driver.generate_messages(
        [{"role": "user", "content": "hi"}],
        options={
            "temperature": 0.2,
            "json_mode": True,
            "json_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
        },
    )
    assert client.messages.last_kwargs["extra_body"]["temperature"] == 0.2


def test_the_helper_keeps_whatever_else_is_in_extra_body() -> None:
    kwargs: dict[str, Any] = {"extra_body": {"thinking": {"type": "enabled"}}}
    _apply_temperature(kwargs, 0.7)
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}, "temperature": 0.7}


def test_the_helper_creates_the_body_when_there_is_none() -> None:
    kwargs: dict[str, Any] = {}
    _apply_temperature(kwargs, 0.0)
    assert kwargs == {"extra_body": {"temperature": 0.0}}
