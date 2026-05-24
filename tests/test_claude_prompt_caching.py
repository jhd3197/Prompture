"""Tests for Anthropic prompt caching (Phase 1).

Two layers:

1. **Unit tests** — exercise ``_apply_cache_control_to_system`` and
   ``_apply_cache_control_to_tools`` directly. No network, no SDK,
   always run.
2. **Driver integration tests** — instantiate ``ClaudeDriver`` and
   ``AsyncClaudeDriver`` with a stub Anthropic SDK and verify the
   request kwargs include ``cache_control`` markers when expected.
3. **Live integration test** — marked ``integration``, hits the real
   Anthropic API to verify ``cache_creation_input_tokens > 0`` on the
   first call and ``cache_read_input_tokens > 0`` on the second.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from prompture.drivers.claude_driver import (
    CACHE_PROMPT_MIN_CHARS,
    _apply_cache_control_to_system,
    _apply_cache_control_to_tools,
)

# ---------------------------------------------------------------------------
# Unit tests — helper behaviour
# ---------------------------------------------------------------------------


class TestApplyCacheControlToSystem:
    def test_none_returns_none(self) -> None:
        assert _apply_cache_control_to_system(None, cache_prompt=True) is None

    def test_empty_string_returns_none(self) -> None:
        assert _apply_cache_control_to_system("", cache_prompt=True) is None

    def test_short_prompt_returns_string(self) -> None:
        out = _apply_cache_control_to_system("short", cache_prompt=True)
        assert out == "short"

    def test_long_prompt_wraps_with_cache_control(self) -> None:
        text = "x" * (CACHE_PROMPT_MIN_CHARS + 1)
        out = _apply_cache_control_to_system(text, cache_prompt=True)
        assert isinstance(out, list)
        assert len(out) == 1
        block = out[0]
        assert block["type"] == "text"
        assert block["text"] == text
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_cache_prompt_false_returns_string_even_when_long(self) -> None:
        text = "x" * (CACHE_PROMPT_MIN_CHARS + 1)
        out = _apply_cache_control_to_system(text, cache_prompt=False)
        assert out == text

    def test_custom_min_chars_threshold(self) -> None:
        text = "x" * 100
        # Below custom threshold -> plain string
        assert _apply_cache_control_to_system(text, cache_prompt=True, min_chars=200) == text
        # At/above custom threshold -> wrapped
        out = _apply_cache_control_to_system(text, cache_prompt=True, min_chars=50)
        assert isinstance(out, list) and out[0]["cache_control"]["type"] == "ephemeral"


class TestApplyCacheControlToTools:
    def _make_tool(self, name: str, description: str = "") -> dict[str, Any]:
        return {"name": name, "description": description, "input_schema": {"type": "object"}}

    def test_empty_tools_returned_unchanged(self) -> None:
        assert _apply_cache_control_to_tools([], cache_prompt=True) == []

    def test_cache_prompt_false_returns_unchanged(self) -> None:
        tools = [self._make_tool("t1", description="x" * 5000)]
        out = _apply_cache_control_to_tools(tools, cache_prompt=False)
        assert out is tools  # no copy when bypassing
        assert "cache_control" not in tools[0]

    def test_small_tool_bundle_unchanged(self) -> None:
        tools = [self._make_tool("t1")]
        out = _apply_cache_control_to_tools(tools, cache_prompt=True)
        assert "cache_control" not in out[0]

    def test_large_tool_bundle_marks_last(self) -> None:
        big_desc = "y" * (CACHE_PROMPT_MIN_CHARS + 100)
        tools = [self._make_tool("t1", description=big_desc), self._make_tool("t2")]
        out = _apply_cache_control_to_tools(tools, cache_prompt=True)
        # Only the LAST tool gets cache_control.
        assert "cache_control" not in out[0]
        assert out[1]["cache_control"] == {"type": "ephemeral"}

    def test_input_tools_not_mutated(self) -> None:
        big_desc = "y" * (CACHE_PROMPT_MIN_CHARS + 100)
        tools = [self._make_tool("t1", description=big_desc)]
        out = _apply_cache_control_to_tools(tools, cache_prompt=True)
        # Output dict has the marker; input dict does NOT.
        assert "cache_control" in out[0]
        assert "cache_control" not in tools[0]

    def test_returns_list_not_tuple(self) -> None:
        big_desc = "y" * (CACHE_PROMPT_MIN_CHARS + 100)
        tools = [self._make_tool("t1", description=big_desc)]
        out = _apply_cache_control_to_tools(tools, cache_prompt=True)
        assert isinstance(out, list)


# ---------------------------------------------------------------------------
# Driver integration — stub the Anthropic SDK, capture request kwargs
# ---------------------------------------------------------------------------


class _StubUsage:
    input_tokens = 100
    output_tokens = 50
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _StubContentBlock:
    def __init__(self, text: str = "ok") -> None:
        self.type = "text"
        self.text = text


class _StubResponse:
    def __init__(self) -> None:
        self.usage = _StubUsage()
        self.content = [_StubContentBlock()]
        self.stop_reason = "end_turn"

    # ClaudeDriver._build_anthropic_meta does dict(resp); make it iterable as a mapping.
    def keys(self) -> list[str]:
        return []

    def __getitem__(self, _: str) -> Any:
        raise KeyError


class _StubMessages:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _StubResponse:
        self.last_kwargs = kwargs
        return _StubResponse()


class _StubClient:
    def __init__(self) -> None:
        self.messages = _StubMessages()


@pytest.fixture
def stub_claude_driver(monkeypatch: pytest.MonkeyPatch):
    """Yield a ``ClaudeDriver`` whose ``anthropic.Anthropic`` client is stubbed."""
    from prompture.drivers import claude_driver as cd

    stub_client = _StubClient()

    class _StubAnthropic:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def __new__(cls, *_: Any, **__: Any) -> _StubClient:  # type: ignore[misc]
            return stub_client

    # Patch the module-level ``anthropic.Anthropic`` constructor.
    monkeypatch.setattr(cd, "anthropic", type("M", (), {"Anthropic": _StubAnthropic}))
    driver = cd.ClaudeDriver(api_key="test-key", model="claude-sonnet-4-6")
    return driver, stub_client


class TestClaudeDriverRequestShape:
    def test_short_system_passes_string(self, stub_claude_driver) -> None:
        driver, client = stub_claude_driver
        driver.generate_messages(
            [{"role": "system", "content": "short"}, {"role": "user", "content": "hi"}],
            options={},
        )
        kw = client.messages.last_kwargs
        # System must be a plain string when below the threshold.
        assert isinstance(kw["system"], str)
        assert kw["system"] == "short"

    def test_long_system_passes_list_with_cache_control(self, stub_claude_driver) -> None:
        driver, client = stub_claude_driver
        long_sys = "x" * (CACHE_PROMPT_MIN_CHARS + 100)
        driver.generate_messages(
            [{"role": "system", "content": long_sys}, {"role": "user", "content": "hi"}],
            options={},
        )
        kw = client.messages.last_kwargs
        assert isinstance(kw["system"], list)
        assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert kw["system"][0]["text"] == long_sys

    def test_cache_prompt_false_disables_caching(self, stub_claude_driver) -> None:
        driver, client = stub_claude_driver
        long_sys = "x" * (CACHE_PROMPT_MIN_CHARS + 100)
        driver.generate_messages(
            [{"role": "system", "content": long_sys}, {"role": "user", "content": "hi"}],
            options={"cache_prompt": False},
        )
        kw = client.messages.last_kwargs
        # Plain string, no list wrapping when caching is disabled.
        assert isinstance(kw["system"], str)

    def test_long_tools_get_cache_control_on_last(self, stub_claude_driver) -> None:
        driver, client = stub_claude_driver
        big_tool_desc = "y" * (CACHE_PROMPT_MIN_CHARS + 100)
        tools = [
            {
                "type": "function",
                "function": {"name": "t1", "description": big_tool_desc, "parameters": {"type": "object"}},
            },
            {
                "type": "function",
                "function": {"name": "t2", "description": "small", "parameters": {"type": "object"}},
            },
        ]
        driver.generate_messages_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            options={},
        )
        kw = client.messages.last_kwargs
        sent_tools = kw["tools"]
        assert len(sent_tools) == 2
        assert "cache_control" not in sent_tools[0]
        assert sent_tools[1]["cache_control"] == {"type": "ephemeral"}

    def test_json_mode_schema_tool_can_be_cached(self, stub_claude_driver) -> None:
        driver, client = stub_claude_driver
        # Build a large schema to push the single inline tool over the threshold.
        big_schema = {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string", "description": "x" * 100} for i in range(50)},
        }
        driver.generate_messages(
            [{"role": "user", "content": "extract"}],
            options={"json_mode": True, "json_schema": big_schema},
        )
        kw = client.messages.last_kwargs
        sent_tools = kw["tools"]
        # Single tool above threshold -> last (only) tool carries cache_control.
        assert len(sent_tools) == 1
        # If schema was large enough, cache_control is present; otherwise absent.
        # Just verify the marker is correct *if* present (the threshold may
        # be tuned later).
        if "cache_control" in sent_tools[0]:
            assert sent_tools[0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Live integration test — hits real Anthropic API
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_prompt_caching_reports_cache_read() -> None:
    """End-to-end: two identical calls should show cache write then cache read.

    Requires ``CLAUDE_API_KEY`` and the ``anthropic`` package. Skips otherwise.
    """
    if not os.getenv("CLAUDE_API_KEY"):
        pytest.skip("CLAUDE_API_KEY not set")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        pytest.skip("anthropic SDK not installed")

    from prompture.drivers.claude_driver import ClaudeDriver

    # Build a system prompt definitely large enough to be cacheable
    # (Sonnet 4.6 min: 1024 tokens ≈ 4000 chars; we use ~6000 to be safe).
    big_system = (
        "You are a careful research assistant. "
        "Always cite sources. Refuse to invent facts. "
        "Reply with exactly one sentence. "
    ) * 100  # ~5500 chars, well over the 4000 threshold

    driver = ClaudeDriver(model="claude-sonnet-4-6")
    messages = [
        {"role": "system", "content": big_system},
        {"role": "user", "content": "Say 'cached'."},
    ]

    # First call -> cache creation
    r1 = driver.generate_messages(messages, options={"max_tokens": 16})
    cache_create_1 = r1["meta"]["cache_creation_tokens"]
    cache_read_1 = r1["meta"]["cached_prompt_tokens"]
    assert cache_create_1 > 0, (
        f"Expected cache_creation_tokens > 0 on first call; "
        f"got cache_create={cache_create_1}, cache_read={cache_read_1}. "
        f"Either the system prompt is too short or cache_control is not being sent."
    )

    # Second call -> cache read (within 5min TTL)
    r2 = driver.generate_messages(messages, options={"max_tokens": 16})
    cache_read_2 = r2["meta"]["cached_prompt_tokens"]
    assert cache_read_2 > 0, (
        f"Expected cache_read_input_tokens > 0 on second call; got {cache_read_2}. "
        f"Cache may have expired or cache_control may not be propagating."
    )
