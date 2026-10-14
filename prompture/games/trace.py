"""Trace sinks: where turn records go. The caller chooses storage and retention.

A sink receives plain JSON-safe dicts (``{"type": ..., ...}``) built from
allowlisted record fields, already redacted and size-bounded. Two sinks ship:

* :class:`MemoryTraceSink` keeps the last ``maxlen`` records in memory.
* :class:`JsonlTraceSink` appends one JSON line per record to a file. Each
  record is one ``write`` of one line, so an interrupted run leaves complete
  earlier lines plus at most one truncated final line, which
  :func:`read_jsonl_trace` skips and reports.

A harness writes ``turn_started`` before its first call, one ``attempt`` per
call as it finishes, then ``turn_finished``. An interrupted turn therefore
leaves its started record and every finished attempt behind.

Sink failures follow the harness's ``trace_errors`` policy: ``"warn"`` (the
default) logs and counts the failure and the turn continues; ``"raise"`` raises
:class:`TraceWriteError` from the turn before the move is returned, so the host
never applies a move whose trace it required but did not get.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from ..exceptions import PromptureError
from .records import redact_text

logger = logging.getLogger("prompture.games.trace")

DEFAULT_MAX_RECORD_BYTES = 512_000


class TraceWriteError(PromptureError, RuntimeError):
    """A trace sink failed and the harness was told to treat that as an error."""


class TraceSink(Protocol):
    def write(self, record: dict[str, Any]) -> None: ...


class MemoryTraceSink:
    """Keeps the most recent ``maxlen`` records (oldest dropped first)."""

    def __init__(self, maxlen: int = 10_000) -> None:
        self.records: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.records.append(record)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.records if r.get("type") == kind]


class JsonlTraceSink:
    """Appends records to a JSON-lines file.

    ``max_record_bytes`` bounds one line: a larger record has its attempts'
    prompts and responses dropped and is marked ``"truncated": true``.
    ``max_file_bytes`` (optional) stops writing once the file is that large
    and raises, so retention stays the caller's decision. ``fsync`` forces each
    line to disk for crash safety at some speed cost.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
        max_file_bytes: int | None = None,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_record_bytes = max_record_bytes
        self.max_file_bytes = max_file_bytes
        self.fsync = fsync
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = encode_record(record, self.max_record_bytes)
        with self._lock:
            if self.max_file_bytes is not None:
                size = self.path.stat().st_size if self.path.exists() else 0
                if size + len(line) > self.max_file_bytes:
                    raise TraceWriteError(f"trace file {self.path} reached its {self.max_file_bytes}-byte limit")
            with self.path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())


def encode_record(record: dict[str, Any], max_bytes: int = DEFAULT_MAX_RECORD_BYTES) -> bytes:
    """One JSON line, redacted again as a last line of defense and bounded in size."""
    text = redact_text(json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")))
    data = (text + "\n").encode("utf-8")
    if len(data) <= max_bytes:
        return data
    slim = dict(record)
    slim["truncated"] = True
    if isinstance(slim.get("record"), dict):
        inner = dict(slim["record"])
        inner["attempts"] = [
            {k: v for k, v in a.items() if k not in ("prompt", "response_text", "schema")}
            for a in inner.get("attempts", [])
        ]
        slim["record"] = inner
    for key in ("prompt", "response_text", "schema"):
        slim.pop(key, None)
    text = redact_text(json.dumps(slim, ensure_ascii=False, default=str, separators=(",", ":")))
    data = (text + "\n").encode("utf-8")
    if len(data) <= max_bytes:
        return data
    minimal = {k: record.get(k) for k in ("type", "run_id", "game_id", "turn_id", "seq") if k in record}
    minimal["truncated"] = True
    return (json.dumps(minimal, default=str) + "\n").encode("utf-8")


def read_jsonl_trace(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """``(records, skipped_lines)`` from a JSONL trace; a torn final line is skipped, not fatal."""
    records: list[dict[str, Any]] = []
    skipped = 0
    for line in _lines(Path(path)):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    return records, skipped


def _lines(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                yield line


class TraceWriter:
    """Writes to a sink under a failure policy; counts what it could not write."""

    def __init__(self, sink: TraceSink | None, policy: str = "warn") -> None:
        if policy not in ("warn", "raise"):
            raise ValueError(f"trace_errors must be 'warn' or 'raise', got {policy!r}")
        self.sink = sink
        self.policy = policy
        self.failures = 0
        self.pending_error: Exception | None = None

    def write(self, record: dict[str, Any]) -> None:
        if self.sink is None:
            return
        try:
            self.sink.write(record)
        except Exception as exc:
            self.failures += 1
            logger.warning("Trace sink failed (%s); policy=%s", exc, self.policy)
            if self.policy == "raise" and self.pending_error is None:
                self.pending_error = exc

    def raise_pending(self) -> None:
        """Raise a deferred sink failure (``raise`` policy) and clear it."""
        if self.pending_error is not None:
            exc, self.pending_error = self.pending_error, None
            raise TraceWriteError(f"trace sink failed: {exc}") from exc
