"""Shared world state for a swarm.

The :class:`Environment` is the swarm's blackboard: an append-only
event log plus a mutable world-state dict.  Agents perceive the world
by reading filtered slices of the event log (and, optionally, world
state); they act by emitting events back into it.

Visibility model
----------------
Every event carries a ``targets`` tuple.  An empty tuple means
*broadcast* — every agent sees it.  A non-empty tuple addresses the
event to those agents only.  The emitting agent always sees their own
event regardless of targets.  Topics provide a parallel filtering axis
agents can subscribe to.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from .types import Event

ENV_AGENT_ID = "__env__"


class Environment:
    """Append-only event log + mutable world state shared by all swarm agents.

    The environment is the only mutable surface every agent shares.
    Reads (``state``, ``events_for``) return defensive copies so callers
    cannot mutate the log in place.

    Args:
        initial_state: Optional starting values for the world-state dict.
        on_event: Optional callback fired for every event written to the
            log.  Receives the :class:`Event` instance.  Useful for
            piping the timeline to a UI without subscribing through
            :class:`~prompture.swarm.types.SwarmCallbacks`.
        on_state_update: Optional callback ``(key, value)`` fired
            whenever :meth:`set` mutates the world-state dict.
    """

    def __init__(
        self,
        *,
        initial_state: dict[str, Any] | None = None,
        on_event: Callable[[Event], None] | None = None,
        on_state_update: Callable[[str, Any], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = dict(initial_state) if initial_state else {}
        self._events: list[Event] = []
        self._on_event = on_event
        self._on_state_update = on_state_update
        self._tick: int = 0

    # ------------------------------------------------------------------
    # Tick bookkeeping
    # ------------------------------------------------------------------

    @property
    def tick(self) -> int:
        """The current round index, controlled by :class:`Swarm`."""
        return self._tick

    def advance_tick(self) -> int:
        """Move to the next round.  Returns the new tick index.

        Called by :class:`~prompture.swarm.Swarm` at the start of every
        round.  Direct callers normally do not invoke this.
        """
        with self._lock:
            self._tick += 1
            return self._tick

    # ------------------------------------------------------------------
    # World state
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict[str, Any]:
        """Return a shallow copy of the current world-state dict."""
        with self._lock:
            return dict(self._state)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a single world-state value."""
        with self._lock:
            return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Write a world-state value and fire the update callback."""
        with self._lock:
            self._state[key] = value
        if self._on_state_update is not None:
            self._on_state_update(key, value)

    def update(self, values: dict[str, Any]) -> None:
        """Merge *values* into the world state.

        Fires ``on_state_update`` once per key.
        """
        for k, v in values.items():
            self.set(k, v)

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def emit(
        self,
        agent_id: str,
        content: str,
        *,
        kind: str = "say",
        targets: Iterable[str] = (),
        topic: str | None = None,
        metadata: dict[str, Any] | None = None,
        tick: int | None = None,
        ts: float | None = None,
    ) -> Event:
        """Append a new :class:`Event` to the log.

        Args:
            agent_id: Author of the event.  Use :data:`ENV_AGENT_ID`
                for events authored by the environment itself.
            content: Free-form text payload.
            kind: Event classification (see :class:`EventKind`).
            targets: Iterable of agent IDs the event is addressed to.
                Empty means broadcast.
            topic: Optional topic tag.
            metadata: Optional structured metadata.
            tick: Override the recorded tick (defaults to the current).
            ts: Override the timestamp (defaults to ``time.time()``).
        """
        event = Event(
            tick=tick if tick is not None else self._tick,
            ts=ts if ts is not None else time.time(),
            agent_id=agent_id,
            kind=kind,
            content=content,
            targets=tuple(targets),
            topic=topic,
            metadata=dict(metadata) if metadata else {},
        )
        with self._lock:
            self._events.append(event)
        if self._on_event is not None:
            self._on_event(event)
        return event

    def seed(self, content: str, *, topic: str | None = None, metadata: dict[str, Any] | None = None) -> Event:
        """Broadcast a seed event from the environment itself.

        Convenience wrapper for the common case of injecting an external
        signal (breaking news, a financial signal, a story prompt) that
        every agent should see in round 0.
        """
        return self.emit(
            ENV_AGENT_ID,
            content,
            kind="seed",
            targets=(),
            topic=topic,
            metadata=metadata,
        )

    @property
    def events(self) -> list[Event]:
        """Return a shallow copy of the full event log."""
        with self._lock:
            return list(self._events)

    def events_for(
        self,
        agent_id: str,
        *,
        since_tick: int | None = None,
        last_n: int | None = None,
        kinds: Iterable[str] | None = None,
        topics: Iterable[str] | None = None,
        include_own: bool = True,
    ) -> list[Event]:
        """Return events visible to *agent_id*, filtered.

        Visibility rules:
        - The agent sees every event addressed to it (``agent_id`` in
          ``event.targets``) and every broadcast event (empty ``targets``).
        - It sees its own events when ``include_own=True``.

        Additional filters:

        Args:
            since_tick: Drop events from earlier ticks (inclusive lower
                bound when set).  Useful for "give me everything since
                last round."
            last_n: Cap the result to the most recent *n* matching
                events.  Applied last.
            kinds: Restrict to events whose ``kind`` is in this iterable.
            topics: Restrict to events whose ``topic`` is in this
                iterable (``None``-topic events are kept only when
                ``None`` itself appears in *topics*).
        """
        kinds_set = set(kinds) if kinds is not None else None
        topics_set = set(topics) if topics is not None else None

        with self._lock:
            candidates = list(self._events)

        visible: list[Event] = []
        for ev in candidates:
            if since_tick is not None and ev.tick < since_tick:
                continue
            is_own = ev.agent_id == agent_id
            if is_own and not include_own:
                continue
            if ev.targets and agent_id not in ev.targets and not is_own:
                continue
            if kinds_set is not None and ev.kind not in kinds_set:
                continue
            if topics_set is not None and ev.topic not in topics_set:
                continue
            visible.append(ev)

        if last_n is not None and last_n >= 0:
            visible = visible[-last_n:]
        return visible

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        """Drop the event log and world state.  Mostly for tests."""
        with self._lock:
            self._events.clear()
            self._state.clear()
            self._tick = 0
