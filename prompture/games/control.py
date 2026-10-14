"""Turn control for the game harness: deadlines, cancellation, retries, budgets and events.

* :class:`CancelToken` — a thread-safe stop signal. Tokens can be linked (a
  batch token cancels every game token made from it).
* :class:`RetryPolicy` — a separate allowance for transient failures (rate
  limits, upstream errors, timeouts, broken connections) with bounded
  exponential backoff, jitter and the provider's ``Retry-After`` hint.
* :class:`GameBudget` — call, token, cost and wall-time limits, shareable
  across harnesses and threads. Each request reserves before it is sent and
  settles after; an unknown price stops spending unless another bound (calls or
  tokens) still limits it.
* :class:`TurnEvent` / :class:`EventBuffer` — sequenced progress events and a
  bounded buffer that coalesces output deltas for slow consumers.

What cancellation really does depends on the driver. A streaming driver's
request is closed between chunks (the stream generator is closed, releasing
its connection). A blocking call cannot be interrupted from Python: the
harness stops waiting, marks the attempt ``abandoned``, ignores any late
result and never sends another request on its behalf, but the provider may
still finish (and bill) that request. Nothing here closes a driver or its
client, so one cancelled game cannot break another game sharing a driver.

Stream-close audit of the sync drivers (closing the stream generator releases
the HTTP response): Claude (context-managed SDK stream), OpenAI chat
completions and Ollama (closed in ``finally``), and the OpenAI-compatible raw
HTTP path (context-managed ``requests`` response) do. Other drivers' streams
are not verified to release early; even a closed request may already have been
billed for tokens the provider generated. Final invoices are the provider's.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .records import BudgetExhausted

logger = logging.getLogger("prompture.games.control")


class CancelToken:
    """A stop signal shared by the code that wants to stop and the turn that should."""

    def __init__(self, parent: CancelToken | None = None) -> None:
        self._event = threading.Event()
        self._parent = parent
        self.reason = ""

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    @property
    def cancelled(self) -> bool:
        if self._event.is_set():
            return True
        if self._parent is not None and self._parent.cancelled:
            self.reason = self.reason or self._parent.reason
            return True
        return False

    def child(self) -> CancelToken:
        """A token cancelled by this one, which can also be cancelled on its own."""
        return CancelToken(self)

    def wait(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds; return early (True) if cancelled."""
        end = time.monotonic() + max(0.0, timeout)
        while True:
            if self.cancelled:
                return True
            left = end - time.monotonic()
            if left <= 0:
                return False
            self._event.wait(min(left, 0.05))


@dataclass(frozen=True)
class RetryPolicy:
    """How transient failures are retried, separately from invalid-output retries.

    Delay for retry ``n`` (1-based) is ``min(max_delay, base_delay * 2**(n-1))``
    plus up to ``jitter`` of itself, or the provider's ``Retry-After`` when it
    is longer (capped at ``max_retry_after``). A turn deadline caps every wait.
    """

    transient_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.25
    respect_retry_after: bool = True
    max_retry_after: float = 30.0

    def delay(self, retry: int, retry_after: float | None = None, rng: random.Random | None = None) -> float:
        backoff = min(self.max_delay, self.base_delay * (2 ** max(0, retry - 1)))
        if self.jitter:
            backoff += (rng or random).uniform(0, self.jitter * backoff)  # nosec B311 - jitter only
        if self.respect_retry_after and retry_after is not None:
            backoff = max(backoff, min(retry_after, self.max_retry_after))
        return backoff

    def to_dict(self) -> dict[str, Any]:
        return {
            "transient_retries": self.transient_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "jitter": self.jitter,
            "respect_retry_after": self.respect_retry_after,
            "max_retry_after": self.max_retry_after,
        }


@dataclass
class Reservation:
    """Budget held for one request until it settles."""

    tokens: float
    cost: float
    settled: bool = False


class GameBudget:
    """Call, token, cost and wall-time limits for one harness or a whole batch.

    Every request first :meth:`reserve` holds one call plus an estimate of its
    tokens (and cost, when ``cost_per_token`` is given), so concurrent games
    cannot all start the last affordable request. :meth:`settle` replaces the
    estimate with what was observed. ``max_seconds`` counts from creation.

    An observed call whose price is unknown makes ``max_cost`` unenforceable.
    With ``strict=True`` that stops further requests; otherwise requests go on
    only while ``max_calls`` or ``max_tokens`` still bounds spending.
    """

    def __init__(
        self,
        *,
        max_calls: int | None = None,
        max_tokens: float | None = None,
        max_cost: float | None = None,
        max_seconds: float | None = None,
        cost_per_token: float | None = None,
        strict: bool = False,
        output_reserve: int = 1024,
    ) -> None:
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.max_seconds = max_seconds
        self.cost_per_token = cost_per_token
        self.strict = strict
        self.output_reserve = output_reserve
        self.calls = 0
        self.tokens = 0.0
        self.cost = 0.0
        self.unknown_cost_calls = 0
        self._reserved_calls = 0
        self._reserved_tokens = 0.0
        self._reserved_cost = 0.0
        self._started = time.monotonic()
        self._lock = threading.Lock()

    def reserve(self, estimated_tokens: float) -> Reservation:
        """Hold one call; raise :class:`~prompture.games.records.BudgetExhausted` if it does not fit."""
        cost = estimated_tokens * self.cost_per_token if self.cost_per_token else 0.0
        with self._lock:
            reason = self._refusal(estimated_tokens, cost)
            if reason:
                raise BudgetExhausted(f"budget exhausted: {reason}")
            self._reserved_calls += 1
            self._reserved_tokens += estimated_tokens
            self._reserved_cost += cost
            return Reservation(estimated_tokens, cost)

    def settle(self, reservation: Reservation, usage: dict[str, float], cost_status: str) -> None:
        """Replace a reservation with observed usage (a sent request always counts as a call)."""
        with self._lock:
            if reservation.settled:
                return
            reservation.settled = True
            self._reserved_calls -= 1
            self._reserved_tokens -= reservation.tokens
            self._reserved_cost -= reservation.cost
            self.calls += 1
            self.tokens += float(usage.get("total_tokens") or 0)
            self.cost += float(usage.get("cost") or 0)
            if cost_status in ("unknown", "partial"):
                self.unknown_cost_calls += 1

    def release(self, reservation: Reservation) -> None:
        """Give back a reservation for a request that was never sent."""
        with self._lock:
            if reservation.settled:
                return
            reservation.settled = True
            self._reserved_calls -= 1
            self._reserved_tokens -= reservation.tokens
            self._reserved_cost -= reservation.cost

    def check(self) -> str | None:
        """Why no further request may start, or ``None``."""
        with self._lock:
            return self._refusal(0.0, 0.0)

    def _refusal(self, tokens: float, cost: float) -> str | None:
        if self.max_seconds is not None and time.monotonic() - self._started >= self.max_seconds:
            return f"{self.max_seconds:g}s time budget used"
        if self.max_calls is not None and self.calls + self._reserved_calls + 1 > self.max_calls:
            return f"{self.max_calls} call limit reached"
        if self.max_tokens is not None and self.tokens + self._reserved_tokens + tokens > self.max_tokens:
            return f"{self.max_tokens:g} token limit would be exceeded"
        if self.max_cost is not None:
            if self.cost + self._reserved_cost + cost >= self.max_cost:
                return f"${self.max_cost:g} cost limit reached"
            if self.unknown_cost_calls and (self.strict or (self.max_calls is None and self.max_tokens is None)):
                return "a previous call's price is unknown, so the cost limit cannot be enforced"
        return None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "tokens": self.tokens,
                "cost": self.cost,
                "unknown_cost_calls": self.unknown_cost_calls,
                "reserved_calls": self._reserved_calls,
                "elapsed_s": round(time.monotonic() - self._started, 3),
                "limits": {
                    "max_calls": self.max_calls,
                    "max_tokens": self.max_tokens,
                    "max_cost": self.max_cost,
                    "max_seconds": self.max_seconds,
                    "strict": self.strict,
                },
            }


# -- events ---------------------------------------------------------------------

EVENT_KINDS = (
    "turn_started",
    "attempt_started",
    "output_delta",
    "attempt_finished",
    "retry_scheduled",
    "turn_completed",
    "turn_failed",
    "turn_cancelled",
    "late_response",
)


@dataclass(frozen=True)
class TurnEvent:
    """One progress event. ``seq`` increases per harness; gaps mean dropped events."""

    seq: int
    kind: str
    run_id: str
    game_id: str
    turn_id: str
    side: str
    time: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "run_id": self.run_id,
            "game_id": self.game_id,
            "turn_id": self.turn_id,
            "side": self.side,
            "time": self.time,
            "data": dict(self.data),
        }


class EventBuffer:
    """A bounded, thread-safe queue of :class:`TurnEvent` objects for a slow consumer.

    Pass ``buffer.put`` as a harness's ``on_event``. When full, a new output
    delta is merged into a queued delta from the same attempt; otherwise the
    oldest queued delta (or, failing that, the oldest event) is dropped and
    counted in ``dropped``. Memory stays bounded whatever the consumer does.
    """

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize < 2:
            raise ValueError("maxsize must be at least 2")
        self.maxsize = maxsize
        self.dropped = 0
        self._items: deque[TurnEvent] = deque()
        self._cond = threading.Condition()

    def put(self, event: TurnEvent) -> None:
        with self._cond:
            if len(self._items) >= self.maxsize:
                if event.kind == "output_delta" and self._merge(event):
                    return
                self._drop_one()
            self._items.append(event)
            self._cond.notify()

    def _merge(self, event: TurnEvent) -> bool:
        for index in range(len(self._items) - 1, -1, -1):
            queued = self._items[index]
            if queued.kind == "output_delta" and queued.data.get("attempt_id") == event.data.get("attempt_id"):
                text = str(queued.data.get("text", "")) + str(event.data.get("text", ""))
                merged = TurnEvent(
                    event.seq,
                    queued.kind,
                    queued.run_id,
                    queued.game_id,
                    queued.turn_id,
                    queued.side,
                    event.time,
                    {**queued.data, "text": text, "coalesced": int(queued.data.get("coalesced", 1)) + 1},
                )
                self._items[index] = merged
                return True
        return False

    def _drop_one(self) -> None:
        self.dropped += 1
        for index, queued in enumerate(self._items):
            if queued.kind == "output_delta":
                del self._items[index]
                return
        self._items.popleft()

    def get(self, timeout: float | None = None) -> TurnEvent | None:
        """The next event, waiting up to ``timeout`` seconds (``None`` if none arrived)."""
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            return self._items.popleft() if self._items else None

    def drain(self) -> list[TurnEvent]:
        with self._cond:
            items = list(self._items)
            self._items.clear()
            return items

    def __iter__(self) -> Iterator[TurnEvent]:
        return iter(self.drain())

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)


EventHandler = Callable[[TurnEvent], None]
