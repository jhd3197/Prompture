"""``AsyncResilientDriver`` — async twin of :class:`~.driver.ResilientDriver`."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from ..drivers.async_base import AsyncDriver
from .backoff import RetryPolicy
from .breaker import BreakerRegistry
from .driver import _CAPABILITY_FLAGS, _EMPTY, _attach_route, _attach_route_to_event
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


def _default_async_factory(model: str, *, api_key: str | None = None, **overrides: Any) -> Any:
    from ..drivers.async_registry import get_async_driver_for_model

    return get_async_driver_for_model(model, api_key=api_key, **overrides)


class AsyncResilientDriver(AsyncDriver):
    """Async driver that routes each call across ordered targets.

    Same arguments and semantics as :class:`~.driver.ResilientDriver`; targets
    are built with :func:`prompture.drivers.async_registry.get_async_driver_for_model`.
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
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
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
            factory=factory or _default_async_factory,
        )
        self._sleep = sleep
        self._sticky = sticky
        self._flags: dict[str, bool] | None = None
        self.model = self._plan.primary.model
        self.last_route: dict[str, Any] | None = None

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
        return f"AsyncResilientDriver({[t.label for t in self.targets]!r})"

    async def _route(
        self,
        call: Callable[[Any], Awaitable[Any]],
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
                        result = await call(drv)
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
                            await self._sleep(delay)
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
            await self._sleep(min_wait)

        raise AllTargetsFailedError(failure_message(attempts), attempts=attempts, last_error=last_exc) from last_exc

    def _sticky_for(self, messages: Any) -> int | None:
        return sticky_hash(messages) if self._sticky else None

    async def _call(
        self,
        fn: Callable[[Any], Awaitable[dict[str, Any]]],
        *,
        need: str | None = None,
        messages: Any = None,
    ) -> dict[str, Any]:
        result, target, attempts, deprioritized = await self._route(fn, need=need, sticky=self._sticky_for(messages))
        route = route_summary(target, attempts, self._plan.strategy, deprioritized)
        self.last_route = route
        return _attach_route(result, route)

    async def _stream(
        self,
        open_fn: Callable[[Any], AsyncIterator[Any]],
        *,
        need: str,
        messages: Any = None,
    ) -> AsyncIterator[Any]:
        async def opener(drv: Any) -> tuple[Any, AsyncIterator[Any]]:
            it = open_fn(drv).__aiter__()
            try:
                first = await it.__anext__()
            except StopAsyncIteration:
                first = _EMPTY
            return first, it

        (first, it), target, attempts, deprioritized = await self._route(
            opener, need=need, sticky=self._sticky_for(messages)
        )
        route = route_summary(target, attempts, self._plan.strategy, deprioritized)
        self.last_route = route
        if first is _EMPTY:
            return
        self._plan.observe(target, first)
        yield _attach_route_to_event(first, route)
        try:
            async for event in it:
                self._plan.observe(target, event)
                yield _attach_route_to_event(event, route)
        except Exception as exc:
            self._plan.penalize(target, classify_error(exc))
            raise

    # -- AsyncDriver interface ------------------------------------------------

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return await self._call(lambda d: d.generate(prompt, dict(options or {})), messages=prompt)

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._call(lambda d: d.generate_messages(messages, dict(options or {})), messages=messages)

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._call(
            lambda d: d.generate_messages_with_tools(messages, tools, dict(options or {})),
            need="supports_tool_use",
            messages=messages,
        )

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        async for event in self._stream(
            lambda d: d.generate_messages_stream(messages, dict(options or {})),
            need="supports_streaming",
            messages=messages,
        ):
            yield event

    async def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[Any]:
        async for event in self._stream(
            lambda d: d.generate_messages_with_tools_stream(messages, tools, dict(options or {})),
            need="supports_tool_use",
            messages=messages,
        ):
            yield event


for _flag in _CAPABILITY_FLAGS:
    setattr(AsyncResilientDriver, _flag, property(lambda self, _n=_flag: self._capabilities()[_n]))


def async_resilient(
    *targets: str | Target | Any,
    policy: RetryPolicy | None = None,
    **kwargs: Any,
) -> AsyncResilientDriver:
    """Build an :class:`AsyncResilientDriver` from targets in preference order."""
    return AsyncResilientDriver(list(targets), policy=policy, **kwargs)
