"""Provider-agnostic prompt-cache helpers.

Prompt caching cuts input cost by ~90% on the repeated prefix of a
request, but providers expose it in two very different ways:

**1. Explicit breakpoints — Anthropic (Claude API, Bedrock, Vertex).**
The caller marks blocks with ``cache_control`` and the API caches
everything up to and including the marked block. Nothing is cached
unless you ask. This module builds those markers.

**2. Automatic prefix caching — OpenAI, xAI/Grok, DeepSeek, Moonshot.**
There is no marker to send; the provider hashes the request prefix and
caches it for you. The only levers are (a) keeping the prefix
byte-stable across requests and (b) on OpenAI, passing
``prompt_cache_key`` so requests sharing a prefix land on the same
cache node. :func:`derive_prompt_cache_key` builds that key.

Either way the win comes from the same discipline: **stable content
first, volatile content last**. Caching is a prefix match, so a single
changed byte at position N invalidates every cache entry at or after N.

Why message-level breakpoints matter
------------------------------------
Marking only ``system`` and ``tools`` caches a fixed-size prefix. In an
agentic tool loop the conversation is what actually grows — ten rounds
of tool results can dwarf the system prompt — and re-sending it
unmarked means it is billed at full input price every round. The
rolling breakpoint from :func:`apply_cache_control_to_messages` is what
turns a single-digit cache hit rate into a high one.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Rough bytes-per-token used to translate Anthropic's token-denominated
# cache minimums into a cheap ``len(str)`` check. Deliberately
# conservative: undershooting means we skip a cacheable block (a missed
# saving), overshooting means we send a marker the API silently ignores
# (no error, no charge, just noise).
CHARS_PER_TOKEN = 4

# Anthropic's minimum cacheable prefix, in tokens. Blocks below this are
# silently ignored — no error, no cache entry. The minimum is NOT
# monotonic across generations, so this has to be a per-model lookup
# rather than a single constant: Opus 4.6 needs 8x the prefix that
# Opus 5 does.
#
# Keys are matched as prefixes of the model id, longest first, so
# "claude-opus-4-6-20251101" resolves via "claude-opus-4-6".
ANTHROPIC_CACHE_MIN_TOKENS: dict[str, int] = {
    # 512-token tier
    "claude-opus-5": 512,
    "claude-fable-5": 512,
    "claude-mythos-5": 512,
    # 1024-token tier
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-4-5": 1024,
    "claude-sonnet-4": 1024,
    "claude-opus-4-1": 1024,
    "claude-opus-4-0": 1024,
    "claude-opus-4": 1024,
    # 2048-token tier
    "claude-opus-4-7": 2048,
    "claude-mythos-preview": 2048,
    "claude-3-5-haiku": 2048,
    # 4096-token tier
    "claude-opus-4-6": 4096,
    "claude-opus-4-5": 4096,
    "claude-haiku-4-5": 4096,
}

# Used when the model id doesn't match any known prefix. 1024 tokens is
# the most common tier and the safest middle ground: too low and we send
# ignored markers, too high and we skip real savings.
DEFAULT_CACHE_MIN_TOKENS = 1024

# Backwards-compatible flat threshold (1024 tokens x 4 chars/token).
# Retained so existing callers and tests that import this constant keep
# working; new code should use :func:`min_cache_chars`.
CACHE_PROMPT_MIN_CHARS = 4000

# Anthropic allows at most 4 ``cache_control`` breakpoints per request;
# a 5th is a hard API error, so the budget is split explicitly between
# system, tools and messages by the caller.
MAX_CACHE_BREAKPOINTS = 4

# Anthropic walks back at most 20 content blocks from a breakpoint
# looking for an existing cache entry. Anchoring the second message
# breakpoint on this stride keeps a readable entry inside that window
# even when a single tool round appends many messages.
ANCHOR_STRIDE = 20

# Block types that must never carry a cache_control marker — the API
# rejects markers on reasoning blocks.
_UNMARKABLE_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})

_VALID_TTLS = ("5m", "1h")


def normalize_ttl(ttl: str | None) -> str:
    """Coerce a caller-supplied TTL to one Anthropic accepts.

    Anything unrecognised falls back to the 5-minute default rather than
    raising — a bad TTL should cost a cache hit, not break the request.
    """
    if ttl in _VALID_TTLS:
        return str(ttl)
    return "5m"


def cache_control(ttl: str | None = "5m") -> dict[str, Any]:
    """Build a ``cache_control`` marker.

    The 5-minute form is emitted without an explicit ``ttl`` key so the
    payload stays byte-identical to what earlier versions sent (and so
    it keeps working against older API versions).
    """
    normalized = normalize_ttl(ttl)
    if normalized == "5m":
        return {"type": "ephemeral"}
    return {"type": "ephemeral", "ttl": normalized}


def cache_write_multiplier(ttl: str | None = "5m") -> float:
    """Cost multiplier to apply to the configured ``cache_write`` rate.

    Rate tables store the 5-minute write price (1.25x input). A 1-hour
    write costs 2x input, so it needs an extra 2.0 / 1.25 = 1.6x on top.
    """
    return 1.6 if normalize_ttl(ttl) == "1h" else 1.0


def _lookup_min_tokens(model: str | None) -> int | None:
    """Resolve ``model`` to a known minimum, or ``None`` if unrecognised."""
    if not model:
        return None
    normalized = str(model).lower()
    # Strip a provider prefix such as "anthropic/" or "claude/".
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]
    # Bedrock ids look like "anthropic.claude-opus-5" or carry a
    # region/version suffix; a longest-prefix match handles both.
    best: int | None = None
    best_len = -1
    for prefix, minimum in ANTHROPIC_CACHE_MIN_TOKENS.items():
        if prefix in normalized and len(prefix) > best_len:
            best, best_len = minimum, len(prefix)
    return best


def min_cache_tokens(model: str | None) -> int:
    """Return the minimum cacheable prefix for ``model``, in tokens."""
    resolved = _lookup_min_tokens(model)
    return resolved if resolved is not None else DEFAULT_CACHE_MIN_TOKENS


def min_cache_chars(model: str | None = None, min_chars: int | None = None) -> int:
    """Return the minimum cacheable prefix for ``model``, in characters.

    An explicit ``min_chars`` always wins so callers (and tests) can
    override the model-derived value. An unrecognised (or absent) model
    keeps the legacy flat threshold rather than the slightly higher
    tokens-based default, so behaviour is unchanged for callers that
    never pass a model.
    """
    if min_chars is not None:
        return min_chars
    resolved = _lookup_min_tokens(model)
    if resolved is None:
        return CACHE_PROMPT_MIN_CHARS
    return resolved * CHARS_PER_TOKEN


# ----------------------------------------------------------------------
# Anthropic — explicit cache_control breakpoints
# ----------------------------------------------------------------------


def apply_cache_control_to_system(
    system_text: str | None,
    *,
    cache_prompt: bool,
    model: str | None = None,
    ttl: str | None = "5m",
    min_chars: int | None = None,
) -> str | list[dict[str, Any]] | None:
    """Wrap the system prompt as a cacheable text block when worthwhile.

    Returns the original string unchanged for short prompts (caching
    would be silently dropped by the API). For longer prompts, returns
    a single-element list ``[{"type": "text", "text": ..., "cache_control": ...}]``
    that the Anthropic Messages API accepts in place of a plain string.

    Returns ``None`` for an empty / missing system prompt so callers
    can simply omit the ``system=`` kwarg.
    """
    if not system_text:
        return None
    if not cache_prompt or len(system_text) < min_cache_chars(model, min_chars):
        return system_text
    return [
        {
            "type": "text",
            "text": system_text,
            "cache_control": cache_control(ttl),
        }
    ]


def apply_cache_control_to_tools(
    tools: list[dict[str, Any]],
    *,
    cache_prompt: bool,
    model: str | None = None,
    ttl: str | None = "5m",
    min_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Tag the last tool dict with cache_control when the bundle is large.

    Anthropic only honours one cache_control marker per logical section.
    Putting it on the *last* tool extends the cached prefix through the
    entire tools block. The cost of trying-and-being-ignored is tiny
    (a few bytes per request), so we apply it whenever the combined
    JSON for the tools meets the model's minimum.

    Returns a new list (and shallow-copies the last tool dict) so the
    caller's tool definitions aren't mutated.
    """
    if not cache_prompt or not tools:
        return tools
    combined_len = sum(len(json.dumps(t, default=str)) for t in tools)
    if combined_len < min_cache_chars(model, min_chars):
        return tools
    result = list(tools)
    last = dict(result[-1])
    last["cache_control"] = cache_control(ttl)
    result[-1] = last
    return result


def _mark_message(message: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any] | None:
    """Return a copy of ``message`` with ``marker`` on its last block.

    String content is promoted to a single text block, since
    ``cache_control`` can only live on a structured block. Returns
    ``None`` when the message has nothing markable (empty content, or
    only reasoning blocks) so the caller can walk further back.
    """
    content = message.get("content")

    if isinstance(content, str):
        if not content.strip():
            return None
        new_msg = dict(message)
        new_msg["content"] = [{"type": "text", "text": content, "cache_control": marker}]
        return new_msg

    if isinstance(content, list) and content:
        # Walk backwards to the last block that can carry a marker.
        for idx in range(len(content) - 1, -1, -1):
            block = content[idx]
            if not isinstance(block, dict):
                continue
            if block.get("type") in _UNMARKABLE_BLOCK_TYPES:
                continue
            new_blocks = list(content)
            new_block = dict(block)
            new_block["cache_control"] = marker
            new_blocks[idx] = new_block
            new_msg = dict(message)
            new_msg["content"] = new_blocks
            return new_msg

    return None


def _anchor_index(count: int, stride: int = ANCHOR_STRIDE) -> int | None:
    """Pick a stable, front-counted index for the second breakpoint.

    Counting from the front is what makes the anchor *stable*: the
    message array grows append-only, so index 40 refers to the same
    message on every subsequent request and the cache entry written
    there stays readable. A rolling breakpoint alone would drift out of
    the 20-block lookback window during fat tool rounds.

    Returns ``None`` when the conversation is too short to need one.
    """
    if count <= stride:
        return None
    idx = ((count - 1) // stride) * stride
    if idx <= 0 or idx >= count - 1:
        return None
    return idx


def apply_cache_control_to_messages(
    messages: list[dict[str, Any]],
    *,
    cache_prompt: bool,
    model: str | None = None,
    ttl: str | None = "5m",
    min_chars: int | None = None,
    max_breakpoints: int = 2,
) -> list[dict[str, Any]]:
    """Place rolling cache breakpoints on the conversation itself.

    This is the breakpoint that matters for agentic runs. ``system`` and
    ``tools`` are a fixed-size prefix; the message array is what grows,
    and without a marker every tool round re-sends the whole accumulated
    history at full input price.

    Two breakpoints are placed when budget allows:

    * a **rolling** one on the last message, which writes a cache entry
      covering everything sent this round;
    * an **anchor** one at a stride-aligned index counted from the front,
      which stays put as the array grows and guarantees a readable entry
      inside Anthropic's 20-block lookback window.

    Returns a new list; input messages are never mutated.
    """
    if not cache_prompt or not messages or max_breakpoints <= 0:
        return messages

    combined_len = sum(len(json.dumps(m, default=str)) for m in messages)
    if combined_len < min_cache_chars(model, min_chars):
        return messages

    marker = cache_control(ttl)
    result = list(messages)
    marked = 0

    # Rolling breakpoint: last markable message, walking backwards past
    # anything empty or reasoning-only.
    rolling_idx: int | None = None
    for idx in range(len(result) - 1, -1, -1):
        updated = _mark_message(result[idx], marker)
        if updated is not None:
            result[idx] = updated
            rolling_idx = idx
            marked += 1
            break

    if rolling_idx is None:
        return messages

    # Anchor breakpoint, only if we have budget and the array is long
    # enough for the rolling one to drift out of lookback range.
    if marked < max_breakpoints:
        anchor_idx = _anchor_index(len(result))
        if anchor_idx is not None and anchor_idx != rolling_idx:
            updated = _mark_message(result[anchor_idx], marker)
            if updated is not None:
                result[anchor_idx] = updated

    return result


def breakpoint_budget(*sections: Any) -> int:
    """Remaining cache_control budget after the given sections are marked.

    Anthropic hard-errors on a 5th breakpoint, so callers pass whatever
    they already marked (a truthy value counts as one used breakpoint)
    and get back how many are left for the messages array.
    """
    used = sum(1 for section in sections if _section_is_marked(section))
    return max(0, MAX_CACHE_BREAKPOINTS - used)


def _section_is_marked(section: Any) -> bool:
    """True when ``section`` already carries a cache_control marker."""
    if isinstance(section, list):
        return any(isinstance(item, dict) and "cache_control" in item for item in section)
    return False


# ----------------------------------------------------------------------
# OpenAI-family — automatic caching, routed by prompt_cache_key
# ----------------------------------------------------------------------


def derive_prompt_cache_key(
    *,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    explicit: str | None = None,
    namespace: str = "prompture",
) -> str | None:
    """Build an OpenAI ``prompt_cache_key`` from the stable prefix.

    OpenAI caches prefixes automatically, but routes requests across
    machines; two requests sharing a prefix only share a cache if they
    land on the same node. ``prompt_cache_key`` is the routing hint, so
    hashing exactly the content that forms the stable prefix (system
    prompt + tool definitions) sends every request with that prefix to
    the same place.

    Deliberately excludes the message array — that is the volatile part,
    and including it would produce a unique key per request, which is
    worse than sending no key at all.

    Returns ``None`` when there is no stable prefix worth routing on.
    """
    if explicit:
        return explicit
    if not system and not tools:
        return None
    digest = hashlib.sha256()
    if system:
        digest.update(system.encode("utf-8", errors="replace"))
    if tools:
        digest.update(json.dumps(tools, sort_keys=True, default=str).encode("utf-8", errors="replace"))
    return f"{namespace}-{digest.hexdigest()[:32]}"
