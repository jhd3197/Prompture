"""In-process live event bus behind a companion's ``GET /v1/live``.

Producers publish an event when a call starts, produces its first token,
changes activity and finishes. Events carry metadata only: ids, model,
project, status, tokens, cost, timings — never prompt or completion text.

The bus keeps the last :data:`RING_SIZE` events, so a client that reconnects
with ``Last-Event-ID`` gets what it missed, and the set of in-flight calls, so
a client that connects mid-request still sees it running. Publishing is
thread-safe. Subscribers are either asyncio queues (FastAPI servers such as
prompture-hub) or plain thread queues (the standard-library local server).
"""

from __future__ import annotations

import asyncio
import itertools
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

RING_SIZE = 1000
SUBSCRIBER_QUEUE = 500
#: In-flight entries older than this are assumed abandoned (a client that
#: dropped a stream before the upstream call finished).
IN_FLIGHT_TTL = 30 * 60


@dataclass(eq=False)  # compared and hashed by identity
class Subscriber:
    """One stream reader. ``loop`` is set for asyncio readers, ``None`` for threads."""

    queue: Any
    loop: asyncio.AbstractEventLoop | None = None
    overflowed: bool = False


@dataclass
class LiveBus:
    ring: deque = field(default_factory=lambda: deque(maxlen=RING_SIZE))
    in_flight: dict[str, dict[str, Any]] = field(default_factory=dict)
    _subscribers: set = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _seq: itertools.count = field(default_factory=lambda: itertools.count(1))

    def publish(self, type_: str, data: dict[str, Any]) -> dict[str, Any]:
        event = {"type": type_, "ts": data.get("ts") or datetime.now(timezone.utc).isoformat(), **data}
        with self._lock:
            event["id"] = next(self._seq)
            self.ring.append(event)
            request_id = data.get("request_id")
            if type_ == "request.started" and request_id:
                self.in_flight[request_id] = {**event, "_mono": time.monotonic()}
            elif request_id in self.in_flight and type_ in ("request.first_token", "request.activity"):
                self.in_flight[request_id].update({k: v for k, v in data.items() if k != "request_id"})
            elif type_ == "request.finished" and request_id:
                self.in_flight.pop(request_id, None)
            subscribers = list(self._subscribers)
        for sub in subscribers:
            if sub.loop is None:
                self._deliver(sub, event)
                continue
            try:
                sub.loop.call_soon_threadsafe(self._deliver, sub, event)
            except RuntimeError:  # loop closed; the subscriber is gone
                self.unsubscribe(sub)
        return event

    @staticmethod
    def _deliver(sub: Subscriber, event: dict[str, Any]) -> None:
        try:
            sub.queue.put_nowait(event)
        except (asyncio.QueueFull, queue.Full):
            sub.overflowed = True

    def subscribe(self) -> Subscriber:
        """Subscribe from a running asyncio loop."""
        sub = Subscriber(asyncio.Queue(maxsize=SUBSCRIBER_QUEUE), asyncio.get_running_loop())
        with self._lock:
            self._subscribers.add(sub)
        return sub

    def subscribe_thread(self) -> Subscriber:
        """Subscribe from a plain thread; read with ``sub.queue.get(timeout=...)``."""
        sub = Subscriber(queue.Queue(maxsize=SUBSCRIBER_QUEUE))
        with self._lock:
            self._subscribers.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        with self._lock:
            self._subscribers.discard(sub)

    def replay(self, after_id: int) -> list[dict[str, Any]]:
        """Buffered events newer than *after_id* (oldest first)."""
        with self._lock:
            return [e for e in self.ring if e["id"] > after_id]

    def running(self) -> list[dict[str, Any]]:
        """Calls that started and have not finished, dropping abandoned ones."""
        cutoff = time.monotonic() - IN_FLIGHT_TTL
        with self._lock:
            for rid in [rid for rid, e in self.in_flight.items() if e["_mono"] < cutoff]:
                del self.in_flight[rid]
            return [{k: v for k, v in e.items() if k != "_mono"} for e in self.in_flight.values()]

    def last_id(self) -> int:
        with self._lock:
            return self.ring[-1]["id"] if self.ring else 0

    def reset(self) -> None:
        with self._lock:
            self.ring.clear()
            self.in_flight.clear()
            self._subscribers.clear()
            self._seq = itertools.count(1)


_bus = LiveBus()


def get_bus() -> LiveBus:
    """The process-wide bus."""
    return _bus


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:20]}"


def visible(event: dict[str, Any], key_ids: Iterable[int] | None) -> bool:
    """Whether a caller limited to *key_ids* (``None`` = all) may see *event*."""
    if key_ids is None:
        return True
    key_id = event.get("key_id")
    return key_id is None or key_id in key_ids


def sse_event(event: dict[str, Any]) -> str:
    """Format one event as a Server-Sent Events frame."""
    import json

    lines = []
    if "id" in event:
        lines.append(f"id: {event['id']}")
    lines.append(f"event: {event['type']}")
    lines.append("data: " + json.dumps(event, separators=(",", ":"), default=str))
    return "\n".join(lines) + "\n\n"
