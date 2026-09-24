"""Headroom-aware routing: targets that are nearly out of a rate-limit window go last."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
import requests

from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver
from prompture.infra.rate_limits import LimitSnapshot, LimitWindow
from prompture.resilience import (
    AsyncResilientDriver,
    BreakerRegistry,
    HeadroomTracker,
    ResilientDriver,
    RetryPolicy,
    explain_route,
    partition_by_headroom,
)
from prompture.resilience.strategies import RouteStats

NOW = 1_800_000_000.0


class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _limits(remaining: int, limit: int = 100, resets_in: float = 30.0, now: float = NOW) -> dict[str, Any]:
    snap = LimitSnapshot(
        windows={"tokens": LimitWindow(limit=limit, remaining=remaining, resets_at=now + resets_in)},
        observed_at=now,
    )
    return snap.to_dict()


class LimitDriver(Driver):
    """Returns a scripted ``meta["rate_limits"]`` (or raises) on each call."""

    supports_messages = True
    supports_streaming = True

    def __init__(self, name: str, script: list[Any]) -> None:
        self.model = name
        self.api_key = None
        self.script = list(script)
        self.calls = 0

    def _next(self) -> Any:
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        limits = self._next()
        meta: dict[str, Any] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0}
        if limits is not None:
            meta["rate_limits"] = limits
        return {"text": self.model, "meta": meta}

    def generate_messages(self, messages, options):
        return self.generate("", options)

    def generate_messages_stream(self, messages, options):
        limits = self._next()
        yield {"type": "delta", "text": self.model}
        meta: dict[str, Any] = {"prompt_tokens": 1, "completion_tokens": 1}
        if limits is not None:
            meta["rate_limits"] = limits
        yield {"type": "done", "text": self.model, "meta": meta}


class AsyncLimitDriver(AsyncDriver):
    supports_messages = True

    def __init__(self, name: str, script: list[Any]) -> None:
        self._sync = LimitDriver(name, script)
        self.model = name
        self.api_key = None

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._sync.generate(prompt, options)

    async def generate_messages(self, messages, options):
        return self._sync.generate("", options)


def _route(*drivers: Any, clock: Clock | None = None, **kwargs: Any) -> tuple[ResilientDriver, HeadroomTracker]:
    tracker = HeadroomTracker(clock=clock or Clock())
    driver = ResilientDriver(
        list(drivers),
        breakers=BreakerRegistry(),
        stats=RouteStats(),
        headroom=tracker,
        sleep=lambda _s: None,
        **kwargs,
    )
    return driver, tracker


class TestCurrentHeadroom:
    def test_expired_windows_no_longer_count(self):
        snap = LimitSnapshot(
            windows={
                "requests": LimitWindow(limit=100, remaining=1, resets_at=NOW - 1),
                "tokens": LimitWindow(limit=100, remaining=40, resets_at=NOW + 10),
            },
            observed_at=NOW - 5,
        )
        assert snap.current_headroom(NOW) == (0.4, "tokens")

    def test_windows_without_reset_expire_by_age(self):
        snap = LimitSnapshot(windows={"requests": LimitWindow(limit=10, remaining=1)}, observed_at=NOW)
        assert snap.current_headroom(NOW + 60, max_age=120) == (0.1, "requests")
        assert snap.current_headroom(NOW + 121, max_age=120) == (None, None)


class TestPartition:
    def _tracker(self, **fractions: float) -> HeadroomTracker:
        tracker = HeadroomTracker(clock=Clock())
        for label, fraction in fractions.items():
            tracker.record(label, LimitSnapshot.from_dict(_limits(int(fraction * 100))))
        return tracker

    def test_low_targets_move_last_in_stable_order(self):
        tracker = self._tracker(a=0.01, b=0.5, c=0.02)
        ordered, notes = partition_by_headroom(["a", "b", "c", "d"], label=str, tracker=tracker, min_headroom=0.05)
        assert ordered == ["b", "d", "a", "c"]
        assert [n["target"] for n in notes] == ["a", "c"]
        assert notes[0] == {"target": "a", "headroom": 0.01, "window": "tokens"}

    def test_all_low_keeps_order(self):
        tracker = self._tracker(a=0.01, b=0.02)
        assert partition_by_headroom(["a", "b"], label=str, tracker=tracker, min_headroom=0.05) == (["a", "b"], [])

    def test_disabled_threshold_keeps_order(self):
        tracker = self._tracker(a=0.0)
        assert partition_by_headroom(["a", "b"], label=str, tracker=tracker, min_headroom=None) == (["a", "b"], [])


class TestResilientRouting:
    def test_low_headroom_primary_is_moved_behind_fallback(self):
        a = LimitDriver("openai/a", [_limits(2)])
        b = LimitDriver("groq/b", [_limits(90)])
        driver, _ = _route(a, b)

        first = driver.generate("hi", {})
        assert first["text"] == "openai/a"

        second = driver.generate("hi", {})
        route = second["meta"]["route"]
        assert second["text"] == "groq/b"
        assert route["deprioritized"] == [{"target": "openai/a", "headroom": 0.02, "window": "tokens"}]
        assert route["fallback"] is False
        assert "moved last: openai/a — 2% of tokens left" in explain_route(route)

    def test_primary_returns_once_its_window_resets(self):
        clock = Clock()
        a = LimitDriver("openai/a", [_limits(2, resets_in=30)])
        b = LimitDriver("groq/b", [None])
        driver, _ = _route(a, b, clock=clock)
        driver.generate("hi", {})
        assert driver.generate("hi", {})["text"] == "groq/b"
        clock.now += 31
        result = driver.generate("hi", {})
        assert result["text"] == "openai/a"
        assert "deprioritized" not in result["meta"]["route"]

    def test_threshold_is_configurable_and_can_be_disabled(self):
        a = LimitDriver("openai/a", [_limits(2)])
        b = LimitDriver("groq/b", [None])
        driver, _ = _route(a, b, policy=RetryPolicy(min_headroom=None))
        driver.generate("hi", {})
        assert driver.generate("hi", {})["text"] == "openai/a"

    def test_rate_limit_error_headers_are_recorded(self):
        resp = requests.Response()
        resp.status_code = 429
        resp.headers.update(
            {
                "retry-after": "20",
                "x-ratelimit-limit-requests": "60",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "20s",
            }
        )
        error = requests.HTTPError("429 Too Many Requests", response=resp)
        a = LimitDriver("openai/a", [error])
        b = LimitDriver("groq/b", [None])
        # Error headers are parsed against wall-clock time, so the tracker must use it too.
        driver, tracker = _route(a, b, clock=Clock(time.time()), policy=RetryPolicy(max_attempts=1))
        assert driver.generate("hi", {})["text"] == "groq/b"
        fraction, window = tracker.headroom("openai/a")
        assert fraction == 0.0
        assert window == "requests"

    def test_stream_done_event_is_recorded(self):
        a = LimitDriver("openai/a", [_limits(1)])
        b = LimitDriver("groq/b", [None])
        driver, tracker = _route(a, b)
        events = list(driver.generate_messages_stream([{"role": "user", "content": "hi"}], {}))
        assert events[-1]["type"] == "done"
        assert tracker.headroom("openai/a")[0] == pytest.approx(0.01)
        second = list(driver.generate_messages_stream([{"role": "user", "content": "hi"}], {}))
        assert second[-1]["text"] == "groq/b"

    def test_tracker_snapshot_is_json_ready(self):
        a = LimitDriver("openai/a", [_limits(25)])
        driver, tracker = _route(a, LimitDriver("groq/b", [None]))
        driver.generate("hi", {})
        snap = tracker.snapshot()["openai/a"]
        assert snap["current_headroom"] == 0.25
        assert snap["current_window"] == "tokens"
        assert snap["windows"]["tokens"]["remaining"] == 25


class TestAsyncResilientRouting:
    def test_low_headroom_primary_is_moved_behind_fallback(self):
        async def run() -> tuple[dict, dict]:
            tracker = HeadroomTracker(clock=Clock())

            async def no_sleep(_s: float) -> None:
                return None

            driver = AsyncResilientDriver(
                [AsyncLimitDriver("openai/a", [_limits(3)]), AsyncLimitDriver("groq/b", [None])],
                breakers=BreakerRegistry(),
                stats=RouteStats(),
                headroom=tracker,
                sleep=no_sleep,
            )
            return await driver.generate("hi", {}), await driver.generate("hi", {})

        first, second = asyncio.run(run())
        assert first["text"] == "openai/a"
        assert second["text"] == "groq/b"
        assert second["meta"]["route"]["deprioritized"][0]["target"] == "openai/a"
