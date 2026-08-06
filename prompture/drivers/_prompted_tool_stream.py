"""Prompted tool-call streaming parser and driver mixin.

This is the engine behind Prompture's *Tier 2* live streaming-tool support
for any text-only driver. It turns a plain token stream from a model that
doesn't have native function-calling (HuggingFace inference, raw local
HTTP, older Ollama models, AirLLM, …) into the same interleaved
:mod:`~prompture.agents.live_events` stream that native streaming-tool
drivers (Claude, OpenAI, …) produce.

How it works
------------

1. The mixin's :meth:`PromptedToolStreamMixin._stream_via_prompted_emulation`
   injects tool schemas into the system prompt using a chosen
   :class:`~prompture.agents.tool_grammars.ToolGrammar`.
2. It then calls the driver's own :meth:`generate_messages_stream` to get
   raw text chunks.
3. Each chunk is fed to :class:`PromptedToolStreamParser`, which runs a
   small state machine: ``NARRATION → TOOL_ARGS → narration``. Text outside
   tool calls becomes :class:`~prompture.agents.live_events.TextDelta`.
   Tool calls become :class:`~prompture.agents.live_events.ToolUseStart`
   (fired the moment the opening tag is matched, so the user sees
   ``calling search(...)`` before the args finish streaming),
   :class:`~prompture.agents.live_events.ToolInputDelta` per inner-JSON
   chunk, and :class:`~prompture.agents.live_events.ToolUseStop` with the
   parsed argument dict at the closing tag.

Partial-delimiter handling
~~~~~~~~~~~~~~~~~~~~~~~~~~

Token boundaries don't respect delimiters — ``<tool_`` may arrive in one
chunk and ``call name="x">`` in the next. The parser keeps a small
*holdback* in each state: when in narration, it never emits the trailing
characters of pending text that could grow into an opening tag; when
inside a tool call, it holds back the trailing characters that could
grow into the close marker.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..agents.live_events import (
    AssistantTurnStart,
    LiveEvent,
    MessageStop,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)
from ..agents.tool_grammars import ToolGrammar, get_grammar, inject_tool_instructions

logger = logging.getLogger(__name__)


class _ParserState(Enum):
    """Parser state machine vertices."""

    NARRATION = "narration"
    """Emitting :class:`~prompture.agents.live_events.TextDelta` events for
    plain text — watching for the grammar's ``open_prefix`` to start a tool
    call."""

    TOOL_ARGS = "tool_args"
    """Inside a tool-call block — emitting
    :class:`~prompture.agents.live_events.ToolInputDelta` events and
    watching for ``grammar.close_marker`` to finish the call."""


@dataclass
class _ToolInProgress:
    """Mutable per-tool-call state while the parser is inside a block."""

    id: str
    name: str
    args_buf: str = field(default_factory=str)
    """Args text not yet emitted as a fragment (holdback for close marker)."""

    full_args: str = field(default_factory=str)
    """Complete args text seen so far — used to parse the final JSON at
    close time, since :attr:`args_buf` gets sliced as fragments are
    emitted."""


class PromptedToolStreamParser:
    """Incremental parser turning raw text chunks into
    :mod:`~prompture.agents.live_events` events.

    Usage::

        parser = PromptedToolStreamParser(get_grammar("xml_tags"))
        for chunk in text_stream:
            yield from parser.feed(chunk)
        yield from parser.finish()
    """

    def __init__(self, grammar: ToolGrammar) -> None:
        self.grammar = grammar
        self._state = _ParserState.NARRATION
        self._narration_buf = ""
        self._current_tool: _ToolInProgress | None = None
        self._open_holdback = max(0, len(grammar.open_prefix) - 1)
        self._close_holdback = max(0, len(grammar.close_marker) - 1)

    def feed(self, chunk: str) -> Iterator[LiveEvent]:
        """Consume one text chunk and yield any events it produced.

        Safe to call with empty strings (no-op). Multi-tool-call chunks
        are handled — the parser loops until no further progress can be
        made on the current buffer.
        """
        if not chunk:
            return
        if self._state is _ParserState.NARRATION:
            self._narration_buf += chunk
            yield from self._drain_narration()
        else:
            assert self._current_tool is not None
            self._current_tool.args_buf += chunk
            self._current_tool.full_args += chunk
            yield from self._drain_tool_args()

    def finish(self) -> Iterator[LiveEvent]:
        """Flush trailing state at end-of-stream.

        - Pending narration is emitted as one final
          :class:`~prompture.agents.live_events.TextDelta`.
        - An unclosed tool call (model stopped mid-block) emits
          ``ToolUseStop`` with whatever args parsed cleanly, logged as
          a warning.
        """
        if self._state is _ParserState.NARRATION:
            if self._narration_buf:
                yield TextDelta(text=self._narration_buf)
                self._narration_buf = ""
            return

        assert self._current_tool is not None
        logger.warning(
            "Tool call %s (%s) was not closed before end-of-stream",
            self._current_tool.name,
            self._current_tool.id,
        )
        # Emit any held-back args as a final fragment for visibility.
        if self._current_tool.args_buf:
            yield ToolInputDelta(
                id=self._current_tool.id,
                fragment=self._current_tool.args_buf,
            )
        parsed = self._parse_tool_args(self._current_tool.full_args)
        yield ToolUseStop(
            id=self._current_tool.id,
            name=self._current_tool.name,
            input=parsed,
        )
        self._current_tool = None
        self._state = _ParserState.NARRATION

    # ------------------------------------------------------------------
    # Internal: state-machine drivers
    # ------------------------------------------------------------------

    def _drain_narration(self) -> Iterator[LiveEvent]:
        """In NARRATION: emit safe-prefix text; on open tag, transition."""
        while True:
            match = self.grammar.open_regex.search(self._narration_buf)
            if match is None:
                # No opening tag yet. Emit everything except a tail that
                # could still grow into the grammar's opening delimiter:
                # either a *complete* ``open_prefix`` whose tag terminator
                # hasn't arrived yet (worst case it's prose mentioning the
                # prefix — emission is merely delayed until finish()), or a
                # partial prefix at the buffer tail.
                prefix = self.grammar.open_prefix
                hold_from = self._narration_buf.rfind(prefix) if prefix else -1
                if hold_from != -1:
                    safe_len = hold_from
                elif self._buffer_could_become_open_tag():
                    safe_len = max(0, len(self._narration_buf) - self._open_holdback)
                else:
                    safe_len = len(self._narration_buf)
                if safe_len > 0:
                    yield TextDelta(text=self._narration_buf[:safe_len])
                    self._narration_buf = self._narration_buf[safe_len:]
                return

            # Opening tag found. Emit any text before it, then transition.
            pre = self._narration_buf[: match.start()]
            if pre:
                yield TextDelta(text=pre)

            name, tag_id = self.grammar.parse_open_tag(match)
            tool_id = tag_id or f"call_{uuid.uuid4().hex[:24]}"
            self._current_tool = _ToolInProgress(id=tool_id, name=name)
            yield ToolUseStart(id=tool_id, name=name)

            leftover = self._narration_buf[match.end() :]
            self._narration_buf = ""
            self._state = _ParserState.TOOL_ARGS
            if leftover:
                self._current_tool.args_buf = leftover
                self._current_tool.full_args = leftover
                yield from self._drain_tool_args()
            # If still in TOOL_ARGS we're waiting for more chunks; if back
            # in NARRATION (close marker was already in the leftover), the
            # while-loop re-scans the rebuilt narration buffer.
            if self._state is _ParserState.TOOL_ARGS:
                return

    def _drain_tool_args(self) -> Iterator[LiveEvent]:
        """In TOOL_ARGS: emit input deltas; on close marker, finalize."""
        assert self._current_tool is not None
        close = self.grammar.close_marker
        buf = self._current_tool.args_buf

        idx = buf.find(close)
        if idx == -1:
            # Hold back the trailing characters that could grow into the
            # close marker.
            safe_len = max(0, len(buf) - self._close_holdback)
            if safe_len > 0:
                fragment = buf[:safe_len]
                yield ToolInputDelta(id=self._current_tool.id, fragment=fragment)
                self._current_tool.args_buf = buf[safe_len:]
            return

        # Close marker found. Emit the final args fragment (text before
        # the close marker), then finalize and reset to narration.
        final_fragment = buf[:idx]
        if final_fragment:
            yield ToolInputDelta(id=self._current_tool.id, fragment=final_fragment)

        # Strip the close marker from full_args to get the JSON for parsing.
        # Use `find` (first occurrence), not `rfind` — when two tool calls
        # arrive in the same chunk, full_args will contain both close
        # markers and rfind would pick up the wrong one.
        full = self._current_tool.full_args
        close_offset = full.find(close)
        args_text = full[:close_offset] if close_offset != -1 else full
        parsed = self._parse_tool_args(args_text)
        yield ToolUseStop(
            id=self._current_tool.id,
            name=self._current_tool.name,
            input=parsed,
        )

        leftover = buf[idx + len(close) :]
        self._current_tool = None
        self._state = _ParserState.NARRATION
        self._narration_buf = leftover
        if self._narration_buf:
            yield from self._drain_narration()

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _buffer_could_become_open_tag(self) -> bool:
        """True iff the tail of :attr:`_narration_buf` could grow into an
        opening tag with more characters appended.

        Cheap O(holdback) check that avoids unnecessary text holdback
        when the buffer's tail clearly can't be the start of the grammar's
        ``open_prefix``.
        """
        prefix = self.grammar.open_prefix
        if self._open_holdback <= 0 or not prefix:
            return False
        tail = self._narration_buf[-self._open_holdback :]
        if not tail:
            return False
        start = tail.rfind(prefix[0])
        if start == -1:
            return False
        candidate = tail[start:]
        return prefix.startswith(candidate)

    def _parse_tool_args(self, raw: str) -> dict[str, Any]:
        """Parse the accumulated args JSON, with graceful fallback."""
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse tool args JSON (%s). Raw: %r", exc, text[:200])
            return {}
        if isinstance(parsed, dict):
            return parsed
        logger.warning(
            "Tool args parsed as %s, not dict; ignoring. Raw: %r",
            type(parsed).__name__,
            text[:200],
        )
        return {}


# ---------------------------------------------------------------------------
# Driver mixin
# ---------------------------------------------------------------------------


class PromptedToolStreamMixin:
    """Mixin that gives any text-streaming driver Tier 2 emulated
    streaming tool use.

    The host driver must provide :meth:`generate_messages_stream` (or
    :meth:`generate_messages` as a fallback) yielding chunks of the
    shape::

        {"type": "delta", "text": "..."}                # token chunks
        {"type": "done",  "text": "...", "meta": {...}} # final

    Set the class attributes::

        supports_streaming_tool_use = True
        prompted_tool_grammar = "xml_tags"  # or another registered grammar

    and either:

    - leave :meth:`generate_messages_with_tools_stream` unimplemented to
      always emulate via :meth:`_stream_via_prompted_emulation`, or
    - implement it to try a native path first and call the emulation as
      fallback.

    The mixin yields :class:`~prompture.agents.live_events.MessageStop`
    at the end with whatever usage the underlying ``generate_messages_stream``
    reports in its final ``"done"`` chunk (or zeros if absent).
    """

    prompted_tool_grammar: str = "xml_tags"

    def _stream_via_prompted_emulation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[LiveEvent]:
        """Synchronous Tier 2 emulation. Yields LiveEvents."""
        grammar = get_grammar(options.get("prompted_tool_grammar") or self.prompted_tool_grammar)
        augmented = inject_tool_instructions(messages, tools, grammar)
        parser = PromptedToolStreamParser(grammar)

        usage: dict[str, Any] = {}
        saw_delta = False
        full_text = ""
        for chunk in self.generate_messages_stream(augmented, options):  # type: ignore[attr-defined]
            ctype = chunk.get("type")
            if ctype == "delta":
                saw_delta = True
                text = chunk.get("text", "")
                full_text += text
                yield from parser.feed(text)
            elif ctype == "done":
                usage = dict(chunk.get("meta", {}) or {})
                # Some drivers only return the full text in "done" with
                # no per-token deltas — feed it through the parser now.
                if not saw_delta:
                    tail = chunk.get("text", "")
                    if tail:
                        yield from parser.feed(tail)

        yield from parser.finish()
        yield MessageStop(stop_reason="end_turn", usage=usage)

    async def _astream_via_prompted_emulation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[LiveEvent]:
        """Async sibling of :meth:`_stream_via_prompted_emulation`."""
        grammar = get_grammar(options.get("prompted_tool_grammar") or self.prompted_tool_grammar)
        augmented = inject_tool_instructions(messages, tools, grammar)
        parser = PromptedToolStreamParser(grammar)

        usage: dict[str, Any] = {}
        saw_delta = False
        async for chunk in self.generate_messages_stream(augmented, options):  # type: ignore[attr-defined]
            ctype = chunk.get("type")
            if ctype == "delta":
                saw_delta = True
                text = chunk.get("text", "")
                for ev in parser.feed(text):
                    yield ev
            elif ctype == "done":
                usage = dict(chunk.get("meta", {}) or {})
                if not saw_delta:
                    tail = chunk.get("text", "")
                    if tail:
                        for ev in parser.feed(tail):
                            yield ev

        for ev in parser.finish():
            yield ev
        yield MessageStop(stop_reason="end_turn", usage=usage)


def emit_turn_start(turn_index: int = 0) -> AssistantTurnStart:
    """Convenience constructor for the ``AssistantTurnStart`` event drivers
    emit at the top of a streaming-tool turn."""
    return AssistantTurnStart(turn_index=turn_index)


__all__ = [
    "PromptedToolStreamMixin",
    "PromptedToolStreamParser",
    "emit_turn_start",
]
