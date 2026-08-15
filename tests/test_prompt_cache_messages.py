"""Tests for message-level prompt caching, per-model minimums and TTL.

The system/tools breakpoints are covered by ``test_claude_prompt_caching``.
This file covers the pieces added to lift real-world cache hit rates:

1. **Message breakpoints** — the rolling + anchor placement that keeps a
   growing agentic conversation cacheable.
2. **Per-model minimums** — Anthropic's minimum cacheable prefix varies
   8x across models, so a flat char threshold both skipped cacheable
   prompts and marked blocks the API ignores.
3. **TTL** — the 1-hour cache window and its 2x write pricing.
4. **OpenAI routing** — ``prompt_cache_key`` derivation.
"""

from __future__ import annotations

import json
from typing import Any

from prompture.drivers._prompt_cache import (
    MAX_CACHE_BREAKPOINTS,
    apply_cache_control_to_messages,
    apply_cache_control_to_system,
    breakpoint_budget,
    cache_control,
    cache_write_multiplier,
    derive_prompt_cache_key,
    min_cache_chars,
    min_cache_tokens,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "content": text}


def _conversation(turns: int, *, chars: int = 1500) -> list[dict[str, Any]]:
    """A conversation guaranteed to clear any model's cache minimum."""
    out: list[dict[str, Any]] = []
    for i in range(turns):
        role = "user" if i % 2 == 0 else "assistant"
        out.append(_msg(role, f"turn-{i} " + "x" * chars))
    return out


def _markers(messages: list[dict[str, Any]]) -> list[int]:
    """Indices of messages carrying a cache_control marker."""
    found: list[int] = []
    for idx, m in enumerate(messages):
        content = m.get("content")
        if isinstance(content, list) and any(isinstance(b, dict) and "cache_control" in b for b in content):
            found.append(idx)
    return found


# ---------------------------------------------------------------------------
# Per-model minimums
# ---------------------------------------------------------------------------


class TestModelMinimums:
    def test_opus_5_uses_512_token_minimum(self) -> None:
        assert min_cache_tokens("claude-opus-5") == 512

    def test_sonnet_5_uses_1024_token_minimum(self) -> None:
        assert min_cache_tokens("claude-sonnet-5") == 1024

    def test_haiku_45_uses_4096_token_minimum(self) -> None:
        assert min_cache_tokens("claude-haiku-4-5") == 4096

    def test_dated_snapshot_resolves_via_prefix(self) -> None:
        assert min_cache_tokens("claude-haiku-4-5-20251001") == 4096

    def test_bedrock_prefixed_id_resolves(self) -> None:
        assert min_cache_tokens("anthropic.claude-opus-5") == 512

    def test_unknown_model_falls_back_to_default(self) -> None:
        assert min_cache_tokens("some-future-model") == 1024

    def test_none_model_falls_back_to_default(self) -> None:
        assert min_cache_tokens(None) == 1024

    def test_longest_prefix_wins(self) -> None:
        # "claude-opus-4-6" must not be shadowed by a shorter "claude-opus-4".
        assert min_cache_tokens("claude-opus-4-6") == 4096

    def test_explicit_min_chars_overrides_model(self) -> None:
        assert min_cache_chars("claude-haiku-4-5", 50) == 50

    def test_opus_5_prompt_below_old_flat_threshold_now_caches(self) -> None:
        """The regression the flat 4000-char threshold caused.

        A ~2500-char system prompt clears Opus 5's 512-token minimum but
        sat under the old flat threshold, so it was silently never cached.
        """
        text = "x" * 2500
        out = apply_cache_control_to_system(text, cache_prompt=True, model="claude-opus-5")
        assert isinstance(out, list)
        assert out[0]["cache_control"] == {"type": "ephemeral"}

    def test_haiku_45_prompt_below_its_minimum_is_not_marked(self) -> None:
        """Inverse case: don't send markers Haiku 4.5 will ignore."""
        text = "x" * 5000
        out = apply_cache_control_to_system(text, cache_prompt=True, model="claude-haiku-4-5")
        assert out == text


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


class TestCacheTTL:
    def test_default_ttl_omits_key_for_wire_compatibility(self) -> None:
        assert cache_control("5m") == {"type": "ephemeral"}

    def test_one_hour_ttl_sets_key(self) -> None:
        assert cache_control("1h") == {"type": "ephemeral", "ttl": "1h"}

    def test_unknown_ttl_falls_back_to_default(self) -> None:
        assert cache_control("7d") == {"type": "ephemeral"}
        assert cache_control(None) == {"type": "ephemeral"}

    def test_write_multiplier_is_neutral_for_5m(self) -> None:
        assert cache_write_multiplier("5m") == 1.0

    def test_write_multiplier_scales_1h_to_2x_input(self) -> None:
        # Rate tables store the 5m write price (1.25x input); 1h is 2x.
        assert cache_write_multiplier("1h") * 1.25 == 2.0

    def test_ttl_propagates_into_system_block(self) -> None:
        out = apply_cache_control_to_system("x" * 5000, cache_prompt=True, model="claude-sonnet-5", ttl="1h")
        assert isinstance(out, list)
        assert out[0]["cache_control"]["ttl"] == "1h"


# ---------------------------------------------------------------------------
# Message breakpoints — the actual fix
# ---------------------------------------------------------------------------


class TestMessageBreakpoints:
    def test_disabled_returns_input_unchanged(self) -> None:
        msgs = _conversation(4)
        assert apply_cache_control_to_messages(msgs, cache_prompt=False) is msgs

    def test_empty_messages_unchanged(self) -> None:
        assert apply_cache_control_to_messages([], cache_prompt=True) == []

    def test_zero_budget_returns_unchanged(self) -> None:
        msgs = _conversation(4)
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, max_breakpoints=0)
        assert out is msgs

    def test_short_conversation_not_marked(self) -> None:
        msgs = [_msg("user", "hi")]
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5")
        assert _markers(out) == []

    def test_last_message_gets_rolling_breakpoint(self) -> None:
        msgs = _conversation(4)
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5")
        assert len(out) - 1 in _markers(out)

    def test_string_content_promoted_to_text_block(self) -> None:
        msgs = _conversation(4)
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5")
        last = out[-1]["content"]
        assert isinstance(last, list)
        assert last[0]["type"] == "text"
        assert last[0]["text"] == msgs[-1]["content"]

    def test_input_messages_not_mutated(self) -> None:
        msgs = _conversation(4)
        original = json.dumps(msgs, sort_keys=True)
        apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5")
        assert json.dumps(msgs, sort_keys=True) == original

    def test_marker_lands_on_last_block_of_list_content(self) -> None:
        msgs = _conversation(3)
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "x" * 600},
                    {"type": "tool_result", "tool_use_id": "b", "content": "y" * 600},
                ],
            }
        )
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5")
        blocks = out[-1]["content"]
        assert "cache_control" not in blocks[0]
        assert "cache_control" in blocks[1]

    def test_thinking_blocks_are_never_marked(self) -> None:
        """Anthropic rejects cache_control on reasoning blocks."""
        msgs = _conversation(3)
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "answer " + "z" * 600},
                    {"type": "thinking", "thinking": "secret"},
                ],
            }
        )
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5")
        blocks = out[-1]["content"]
        assert "cache_control" in blocks[0]
        assert "cache_control" not in blocks[1]

    def test_empty_trailing_message_is_skipped(self) -> None:
        msgs = _conversation(4)
        msgs.append({"role": "assistant", "content": ""})
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5")
        marked = _markers(out)
        assert marked, "expected the walk-back to find an earlier markable message"
        assert len(out) - 1 not in marked

    def test_long_conversation_gets_anchor_plus_rolling(self) -> None:
        msgs = _conversation(45)
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5")
        marked = _markers(out)
        assert len(marked) == 2, f"expected anchor + rolling, got {marked}"
        assert marked[-1] == len(out) - 1

    def test_anchor_is_stable_as_conversation_grows(self) -> None:
        """The anchor must not drift, or its cache entry is never re-read.

        Counting from the front is what makes it stable: appending more
        turns leaves the anchor on the same message, so the entry written
        there stays readable on later requests.
        """
        base = _conversation(45)
        first = _markers(apply_cache_control_to_messages(base, cache_prompt=True, model="claude-sonnet-5"))[0]
        grown = base + _conversation(4)
        second = _markers(apply_cache_control_to_messages(grown, cache_prompt=True, model="claude-sonnet-5"))[0]
        assert first == second

    def test_anchor_moves_within_lookback_window_when_it_shifts(self) -> None:
        """When the anchor does advance, it stays inside the 20-block window."""
        a = _markers(apply_cache_control_to_messages(_conversation(41), cache_prompt=True, model="claude-sonnet-5"))[0]
        b = _markers(apply_cache_control_to_messages(_conversation(61), cache_prompt=True, model="claude-sonnet-5"))[0]
        assert 0 < b - a <= 20

    def test_budget_of_one_places_only_rolling(self) -> None:
        msgs = _conversation(45)
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5", max_breakpoints=1)
        assert _markers(out) == [len(out) - 1]

    def test_ttl_propagates_into_message_marker(self) -> None:
        msgs = _conversation(4)
        out = apply_cache_control_to_messages(msgs, cache_prompt=True, model="claude-sonnet-5", ttl="1h")
        last = out[-1]["content"][-1]
        assert last["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


# ---------------------------------------------------------------------------
# Breakpoint budget — Anthropic hard-errors on a 5th marker
# ---------------------------------------------------------------------------


class TestBreakpointBudget:
    def test_nothing_marked_leaves_full_budget(self) -> None:
        assert breakpoint_budget(None, None) == MAX_CACHE_BREAKPOINTS

    def test_plain_string_system_costs_nothing(self) -> None:
        assert breakpoint_budget("a plain system prompt") == MAX_CACHE_BREAKPOINTS

    def test_marked_system_and_tools_each_cost_one(self) -> None:
        system = [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]
        tools = [{"name": "t", "cache_control": {"type": "ephemeral"}}]
        assert breakpoint_budget(system, tools) == MAX_CACHE_BREAKPOINTS - 2

    def test_unmarked_tools_cost_nothing(self) -> None:
        assert breakpoint_budget(None, [{"name": "t"}]) == MAX_CACHE_BREAKPOINTS

    def test_total_markers_never_exceed_api_limit(self) -> None:
        system = [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]
        tools = [{"name": "t", "cache_control": {"type": "ephemeral"}}]
        msgs = apply_cache_control_to_messages(
            _conversation(60),
            cache_prompt=True,
            model="claude-sonnet-5",
            max_breakpoints=breakpoint_budget(system, tools),
        )
        total = 1 + 1 + len(_markers(msgs))
        assert total <= MAX_CACHE_BREAKPOINTS


# ---------------------------------------------------------------------------
# OpenAI — automatic caching, routed by prompt_cache_key
# ---------------------------------------------------------------------------


class TestPromptCacheKey:
    def test_no_prefix_returns_none(self) -> None:
        assert derive_prompt_cache_key() is None

    def test_explicit_key_wins(self) -> None:
        assert derive_prompt_cache_key(system="x", explicit="mine") == "mine"

    def test_same_prefix_yields_same_key(self) -> None:
        a = derive_prompt_cache_key(system="shared prompt", tools=[{"name": "t"}])
        b = derive_prompt_cache_key(system="shared prompt", tools=[{"name": "t"}])
        assert a == b and a is not None

    def test_different_prefix_yields_different_key(self) -> None:
        a = derive_prompt_cache_key(system="prompt A")
        b = derive_prompt_cache_key(system="prompt B")
        assert a != b

    def test_tool_ordering_does_not_change_key(self) -> None:
        a = derive_prompt_cache_key(system="s", tools=[{"a": 1, "b": 2}])
        b = derive_prompt_cache_key(system="s", tools=[{"b": 2, "a": 1}])
        assert a == b

    def test_key_is_namespaced_and_bounded(self) -> None:
        key = derive_prompt_cache_key(system="x")
        assert key is not None
        assert key.startswith("prompture-")
        assert len(key) <= 64


class TestOpenAIDriverCacheKey:
    def test_derived_from_system_not_user_turn(self) -> None:
        """The key must ignore the volatile tail, or every request is unique."""
        from prompture.drivers.openai_driver import _openai_prompt_cache_key

        base = [{"role": "system", "content": "stable"}]
        a = _openai_prompt_cache_key([*base, {"role": "user", "content": "q1"}], {})
        b = _openai_prompt_cache_key([*base, {"role": "user", "content": "q2"}], {})
        assert a == b and a is not None

    def test_cache_prompt_false_disables_key(self) -> None:
        from prompture.drivers.openai_driver import _openai_prompt_cache_key

        msgs = [{"role": "system", "content": "stable"}]
        assert _openai_prompt_cache_key(msgs, {"cache_prompt": False}) is None

    def test_key_omitted_from_kwargs_when_none(self) -> None:
        """Compat endpoints (Grok/DeepSeek/Moonshot) reject unknown fields."""
        from prompture.drivers.openai_driver import _build_openai_base_kwargs

        kwargs = _build_openai_base_kwargs("m", [], {}, "max_tokens", False, 512)
        assert "prompt_cache_key" not in kwargs

    def test_key_included_when_supplied(self) -> None:
        from prompture.drivers.openai_driver import _build_openai_base_kwargs

        kwargs = _build_openai_base_kwargs("m", [], {}, "max_tokens", False, 512, prompt_cache_key="k")
        assert kwargs["prompt_cache_key"] == "k"
