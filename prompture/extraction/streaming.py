"""Partial-JSON streaming for structured extraction.

When an LLM streams a JSON response, this module makes the in-flight
JSON readable as a partial Pydantic model so downstream consumers can
render fields progressively instead of blocking on the full response.

The parser is stdlib-only by default. For complex / edge-case partial
JSON (unicode escapes mid-stream, exotic literals) you can install the
``partial-json-parser`` package and pass ``parser="partial_json_parser"``
to :func:`stream_extract` — the module imports it lazily and falls back
to stdlib when it's not installed.

Two public entry points:

* :func:`parse_partial_json` — best-effort parse of an incomplete JSON
  string into a Python value. Returns ``None`` when no recoverable
  parse exists.
* :func:`stream_extract` / :func:`astream_extract` — drive an LLM
  driver's streaming method and yield progressively-filled Pydantic
  model instances as text arrives.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Partial JSON parsing
# ---------------------------------------------------------------------------


def _scan_state(text: str) -> tuple[list[str], bool, bool] | None:
    """Scan ``text`` once and report the open-bracket stack + string state.

    Returns ``(stack, in_string, trailing_escape)``. ``stack`` is a list
    of *closing* characters (``'}'`` / ``']'``) in reverse open order
    so it can be reversed and appended to recover a parseable suffix.
    ``trailing_escape`` is True when the buffer ended right after a
    backslash — we can't reliably parse that case.

    Returns ``None`` if the bracket structure is unrecoverable
    (more closers than openers, etc.).
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if not stack:
                return None
            stack.pop()
    return stack, in_string, escape


def parse_partial_json(text: str) -> Any:
    """Best-effort parse of a possibly-incomplete JSON string.

    Strategy:

    1. Try ``json.loads(text)`` — fast path when the stream happens to
       be balanced.
    2. Walk the string to compute the open-bracket stack and whether
       we're inside a string literal. Append the necessary closing
       tokens (closing quote, then ``}``/``]`` in reverse open order)
       and re-attempt parsing.
    3. If that fails, progressively trim the trailing fragment back to
       a comma / open-bracket / colon boundary and retry. This handles
       partial numbers (``42``), partial keywords (``tru``), or partial
       keys (``"na``).

    Returns the parsed Python value, or ``None`` when no usable parse
    exists yet. Stripping whitespace at both ends; treats input as
    UTF-8 text. Designed for object/array roots — bare-literal streams
    work too but are not the common case.
    """
    if not text:
        return None
    text = text.strip()
    if not text:
        return None

    # Fast path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Compute structural state once
    state = _scan_state(text)
    if state is None:
        return None
    _stack, _in_string, _trailing_escape = state
    if _trailing_escape:
        # Drop the trailing backslash and retry — it's an incomplete escape.
        text = text[:-1]

    suffix = ""
    if _in_string:
        suffix += '"'
    suffix += "".join(reversed(_stack))

    try:
        return json.loads(text + suffix)
    except json.JSONDecodeError:
        pass

    # Fallback: walk backwards from the end, attempting parse after each cut.
    # We only cut at "safe" boundaries that won't sit inside a string literal.
    cut_points: list[int] = []
    s_in_string = False
    s_escape = False
    for i, ch in enumerate(text):
        if s_escape:
            s_escape = False
            continue
        if ch == "\\":
            s_escape = True
            continue
        if ch == '"':
            s_in_string = not s_in_string
            continue
        if s_in_string:
            continue
        if ch in (",", "{", "[", ":", "}", "]"):
            cut_points.append(i + 1)

    # Try the cut points from the rightmost backwards.
    for cut in reversed(cut_points):
        candidate = text[:cut].rstrip(" \t\n\r,:")
        if not candidate:
            continue
        sub_state = _scan_state(candidate)
        if sub_state is None:
            continue
        sub_stack, sub_in_string, sub_escape = sub_state
        if sub_escape:
            continue
        sub_suffix = ""
        if sub_in_string:
            sub_suffix += '"'
        sub_suffix += "".join(reversed(sub_stack))
        try:
            return json.loads(candidate + sub_suffix)
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------------------------
# Building partial Pydantic models
# ---------------------------------------------------------------------------


def _build_partial_model(model_cls: type[T], data: Any) -> T | None:
    """Build a non-validated instance of ``model_cls`` from partial data.

    Uses :meth:`BaseModel.model_construct` (Pydantic v2) so missing /
    not-yet-typed fields don't raise. Returns ``None`` when ``data``
    isn't a mapping (a partial top-level array, say) — the caller can
    decide to skip emitting in that case.

    Only keys present in ``model_cls.model_fields`` are forwarded so
    stray keys the LLM hallucinates don't pollute the partial.
    """
    if not isinstance(data, dict):
        return None
    field_names = set(model_cls.model_fields.keys())
    filtered = {k: v for k, v in data.items() if k in field_names}
    try:
        return model_cls.model_construct(**filtered)
    except Exception:
        logger.debug("Partial model construction failed for %s", model_cls.__name__, exc_info=True)
        return None


def _dump(model: BaseModel) -> dict[str, Any]:
    """Compare two partial models by their dumped JSON-compatible representation."""
    try:
        return model.model_dump(mode="json")
    except Exception:
        return model.model_dump()


# ---------------------------------------------------------------------------
# Streaming drivers
# ---------------------------------------------------------------------------


def stream_extract(
    driver: Any,
    content_prompt: str,
    model_cls: type[T],
    *,
    system_prompt: str | None = None,
    options: dict[str, Any] | None = None,
    emit_unchanged: bool = False,
) -> Iterator[T]:
    """Yield progressively-filled ``model_cls`` instances from a streaming driver.

    Drives :meth:`Driver.generate_messages_stream` (or equivalent),
    accumulating delta text, calling :func:`parse_partial_json` after
    every chunk and yielding a new partial Pydantic instance whenever
    the parsed-and-filtered snapshot changes.

    Args:
        driver: Any LLM driver exposing ``generate_messages_stream``.
        content_prompt: The user prompt the model should respond to.
            A schema-enforcing instruction is up to the caller — this
            function does NOT modify the prompt beyond optionally
            wrapping it as a user-role message.
        model_cls: Pydantic v2 model to construct partials of.
        system_prompt: Optional system prompt. When given, sent as the
            first message in the role=system slot.
        options: Driver options forwarded to ``generate_messages_stream``.
        emit_unchanged: When True, yield an instance even when the
            partial snapshot is identical to the previous one (useful
            for testing). Defaults to False — only changed snapshots
            are yielded.

    Yields:
        A new ``model_cls`` instance after each material change. The
        final instance may still be a partial if the driver finished
        before producing a complete object; consumers wanting strict
        validation should pass the final into ``model_cls(...)``
        themselves.

    Raises:
        AttributeError: when ``driver`` has no ``generate_messages_stream``
            method. Use :func:`prompture.extraction.core.extract_with_model`
            instead for non-streaming drivers.
    """
    if not hasattr(driver, "generate_messages_stream"):
        raise AttributeError(
            f"{type(driver).__name__} does not expose generate_messages_stream(); "
            "use extract_with_model() for non-streaming drivers."
        )

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_prompt})

    merged_options = dict(options or {})

    accumulated = ""
    last_snapshot: dict[str, Any] | None = None

    for chunk in driver.generate_messages_stream(messages, merged_options):
        if not isinstance(chunk, dict):
            continue
        if chunk.get("type") == "delta":
            piece = chunk.get("text") or ""
            if not piece:
                continue
            accumulated += piece
            parsed = parse_partial_json(accumulated)
            partial = _build_partial_model(model_cls, parsed)
            if partial is None:
                continue
            snapshot = _dump(partial)
            if not emit_unchanged and snapshot == last_snapshot:
                continue
            last_snapshot = snapshot
            yield partial
        elif chunk.get("type") == "done":
            # Last shot: if the accumulated text now parses fully, emit one
            # final, *validated* model so consumers get the strict object.
            final_text = chunk.get("text") or accumulated
            try:
                final_data = json.loads(final_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(final_data, dict):
                continue
            try:
                yield model_cls.model_validate(final_data)
            except Exception:
                # Final didn't validate — keep the last partial as our best output.
                logger.debug("Final stream payload didn't validate; consumer keeps last partial.")


async def astream_extract(
    driver: Any,
    content_prompt: str,
    model_cls: type[T],
    *,
    system_prompt: str | None = None,
    options: dict[str, Any] | None = None,
    emit_unchanged: bool = False,
) -> AsyncIterator[T]:
    """Async counterpart of :func:`stream_extract`.

    Drives ``async generate_messages_stream`` on the driver. Same
    semantics, same emission rules — see :func:`stream_extract`.
    """
    if not hasattr(driver, "generate_messages_stream"):
        raise AttributeError(
            f"{type(driver).__name__} does not expose generate_messages_stream(); "
            "use extract_with_model() for non-streaming drivers."
        )

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_prompt})

    merged_options = dict(options or {})

    accumulated = ""
    last_snapshot: dict[str, Any] | None = None

    async for chunk in driver.generate_messages_stream(messages, merged_options):
        if not isinstance(chunk, dict):
            continue
        if chunk.get("type") == "delta":
            piece = chunk.get("text") or ""
            if not piece:
                continue
            accumulated += piece
            parsed = parse_partial_json(accumulated)
            partial = _build_partial_model(model_cls, parsed)
            if partial is None:
                continue
            snapshot = _dump(partial)
            if not emit_unchanged and snapshot == last_snapshot:
                continue
            last_snapshot = snapshot
            yield partial
        elif chunk.get("type") == "done":
            final_text = chunk.get("text") or accumulated
            try:
                final_data = json.loads(final_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(final_data, dict):
                continue
            try:
                yield model_cls.model_validate(final_data)
            except Exception:
                logger.debug("Final stream payload didn't validate; consumer keeps last partial.")


__all__ = [
    "astream_extract",
    "parse_partial_json",
    "stream_extract",
]
