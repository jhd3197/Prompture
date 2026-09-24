"""``ResilientDriver`` — retries, cooldowns and failover behind the Driver interface.

Because it *is* a :class:`~prompture.drivers.base.Driver`, it drops into
anything that takes a driver: ``Conversation``, ``Agent``, ``ask_for_json``,
``extract_with_model``, the HTTP server.

Example::

    from prompture import resilient

    driver = resilient("openai/gpt-4o", "claude/claude-sonnet-4-5", "groq/llama-3.3-70b-versatile")
    resp = driver.generate_messages([{"role": "user", "content": "hi"}], {})
    resp["meta"]["route"]   # {"served_by": ..., "fallback": bool, "attempts": [...]}
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from ..drivers.base import Driver
from .backoff import RetryPolicy
from .breaker import BreakerRegistry
from .errors import AllTargetsFailedError, ErrorAction, classify_error
from .headroom import HeadroomTracker
from .router import (
    RoutePlan,
    Target,
    attempt_record,
    failure_message,
    route_summary,
    should_retry_same,
    sticky_hash,
)
from .strategies import RouteStats

_CAPABILITY_FLAGS = (
    "supports_json_mode",
    "supports_json_schema",
    "supports_messages",
    "supports_tool_use",
    "supports_streaming",
    "supports_streaming_tool_use",
    "supports_vision",
)

_EMPTY = object()


def _default_factory(model: str, *, api_key: str | None = None, **overrides: Any) -> Any:
    from ..drivers import get_driver_for_model

    return get_driver_for_model(model, api_key=api_key, **overrides)


def _attach_route(result: Any, route: dict[str, Any]) -> Any:
    if isinstance(result, dict):
        meta = result.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            result["meta"] = meta
        meta["route"] = route
    return result


def _attach_route_to_event(event: Any, route: dict[str, Any]) -> Any:
    if isinstance(event, dict) and event.get("type") == "done":
        return _attach_route(event, route)
    usage = getattr(event, "usage", None)
    if getattr(event, "event_type", None) == "message_stop" and isinstance(usage, dict):
        usage["route"] = route
    return event


class ResilientDriver(Driver):
    """A driver that routes each call across ordered targets.

    Args:
        targets: Model strings (``"provider/model"``), :class:`Target` objects,
            or ready driver instances, in preference order. A model string
            whose provider has a key pool (``OPENAI_API_KEYS=a,b``) expands to
            one target per key, rotated round-robin between calls.
        policy: Retry / backoff / cooldown timings.
        breakers: Breaker registry; defaults to the process-wide one so every
            resilient driver shares health knowledge.
        use_key_pools: Expand model strings using registered/env key pools.
        factory: ``(model, api_key=..., **overrides) -> driver``; defaults to
            :func:`prompture.drivers.get_driver_for_model`.
        sleep: Injected for tests.
    """

    def __init__(
        self,
        targets: Sequence[str | Target | Any],
        *,
        policy: RetryPolicy | None = None,
        breakers: BreakerRegistry | None = None,
        use_key_pools: bool = True,
        strategy: str = "priority",
        weights: Sequence[float] | None = None,
        sticky: bool = False,
        stats: RouteStats | None = None,
        headroom: HeadroomTracker | None = None,
        factory: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._plan = RoutePlan(
            targets,
            policy=policy,
            breakers=breakers,
            use_key_pools=use_key_pools,
            strategy=strategy,
            weights=weights,
            stats=stats,
            headroom=headroom,
            factory=factory or _default_factory,
        )
        self._sleep = sleep
        self._sticky = sticky
        self._flags: dict[str, bool] | None = None
        self.model = self._plan.primary.model
        self.last_route: dict[str, Any] | None = None

    # -- introspection ------------------------------------------------------

    @property
    def targets(self) -> list[Target]:
        return self._plan.targets

    @property
    def policy(self) -> RetryPolicy:
        return self._plan.policy

    def _capabilities(self) -> dict[str, bool]:
        if self._flags is None:
            drivers = self._plan.peek_drivers()
            self._flags = {f: any(getattr(d, f, False) for d in drivers) for f in _CAPABILITY_FLAGS}
        return self._flags

    def __repr__(self) -> str:
        return f"ResilientDriver({[t.label for t in self.targets]!r})"

    # -- the attempt loop ---------------------------------------------------

    def _route(
        self,
        call: Callable[[Any], Any],
        *,
        need: str | None = None,
        sticky: int | None = None,
    ) -> tuple[Any, Target, list[dict[str, Any]], list[dict[str, Any]]]:
        plan = self._plan
        attempts: list[dict[str, Any]] = []
        deprioritized: list[dict[str, Any]] = []
        last_exc: BaseException | None = None

        for round_ in range(2):
            attempted = False
            min_wait: float | None = None
            candidates, deprioritized = plan.ordered_candidates(sticky)
            for idx, target in enumerate(candidates):
                is_last = idx == len(candidates) - 1
                allowed, wait = plan.gate(target)
                if not allowed:
                    attempts.append(attempt_record(target, outcome="skipped", wait=wait))
                    if wait > 0:
                        min_wait = wait if min_wait is None else min(min_wait, wait)
                    continue
                try:
                    drv = plan.driver_for(target)
                except Exception as exc:
                    last_exc = exc
                    plan.release(target)
                    attempts.append(attempt_record(target, outcome="unavailable", info=classify_error(exc)))
                    continue
                if need and not getattr(drv, need, False):
                    plan.release(target)
                    attempts.append(attempt_record(target, outcome="unsupported"))
                    continue

                attempted = True
                attempt = 0
                while True:
                    started = time.perf_counter()
                    try:
                        result = call(drv)
                    except Exception as exc:
                        elapsed = (time.perf_counter() - started) * 1000
                        info = classify_error(exc)
                        last_exc = exc
                        plan.observe(target, exc)
                        attempts.append(attempt_record(target, outcome="error", elapsed_ms=elapsed, info=info))
                        if info.action is ErrorAction.FATAL:
                            plan.release(target)
                            self.last_route = route_summary(target, attempts, plan.strategy, deprioritized)
                            raise
                        delay = should_retry_same(plan.policy, info, attempt, is_last)
                        if delay is not None:
                            self._sleep(delay)
                            attempt += 1
                            continue
                        plan.penalize(target, info)
                        break
                    elapsed = (time.perf_counter() - started) * 1000
                    plan.succeed(target, elapsed)
                    plan.observe(target, result)
                    attempts.append(attempt_record(target, outcome="ok", elapsed_ms=elapsed))
                    return result, target, attempts, deprioritized

            if attempted or round_ or min_wait is None or min_wait > plan.policy.max_wait:
                break
            # Everything is briefly parked (e.g. a single rate-limited model):
            # wait for the first one to come back rather than failing outright.
            self._sleep(min_wait)

        raise AllTargetsFailedError(failure_message(attempts), attempts=attempts, last_error=last_exc) from last_exc

    def _sticky_for(self, messages: Any) -> int | None:
        return sticky_hash(messages) if self._sticky else None

    def _call(
        self,
        fn: Callable[[Any], dict[str, Any]],
        *,
        need: str | None = None,
        messages: Any = None,
    ) -> dict[str, Any]:
        result, target, attempts, deprioritized = self._route(fn, need=need, sticky=self._sticky_for(messages))
        route = route_summary(target, attempts, self._plan.strategy, deprioritized)
        self.last_route = route
        return _attach_route(result, route)

    def _stream(self, open_fn: Callable[[Any], Any], *, need: str, messages: Any = None) -> Iterator[Any]:
        def opener(drv: Any) -> tuple[Any, Iterator[Any]]:
            it = iter(open_fn(drv))
            try:
                first = next(it)
            except StopIteration:
                first = _EMPTY
            return first, it

        (first, it), target, attempts, deprioritized = self._route(opener, need=need, sticky=self._sticky_for(messages))
        route = route_summary(target, attempts, self._plan.strategy, deprioritized)
        self.last_route = route
        if first is _EMPTY:
            return
        self._plan.observe(target, first)
        yield _attach_route_to_event(first, route)
        try:
            for event in it:
                self._plan.observe(target, event)
                yield _attach_route_to_event(event, route)
        except Exception as exc:
            # Bytes already reached the caller, so we can't fail over; just
            # make sure the target's health reflects the broken stream.
            self._plan.penalize(target, classify_error(exc))
            raise

    # -- Driver interface ---------------------------------------------------

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._call(lambda d: d.generate(prompt, dict(options or {})), messages=prompt)

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._call(lambda d: d.generate_messages(messages, dict(options or {})), messages=messages)

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        return self._call(
            lambda d: d.generate_messages_with_tools(messages, tools, dict(options or {})),
            need="supports_tool_use",
            messages=messages,
        )

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        yield from self._stream(
            lambda d: d.generate_messages_stream(messages, dict(options or {})),
            need="supports_streaming",
            messages=messages,
        )

    def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[Any]:
        yield from self._stream(
            lambda d: d.generate_messages_with_tools_stream(messages, tools, dict(options or {})),
            need="supports_tool_use",
            messages=messages,
        )


def _flag_property(name: str) -> property:
    return property(lambda self: self._capabilities()[name])


for _flag in _CAPABILITY_FLAGS:
    setattr(ResilientDriver, _flag, _flag_property(_flag))


def resilient(
    *targets: str | Target | Any,
    policy: RetryPolicy | None = None,
    **kwargs: Any,
) -> ResilientDriver:
    """Build a :class:`ResilientDriver` from targets in preference order."""
    return ResilientDriver(list(targets), policy=policy, **kwargs)
