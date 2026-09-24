"""Tests for the resilience layer: classification, breakers, key pools, failover."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
import requests

from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver, DriverHTTPError
from prompture.resilience import (
    AllTargetsFailedError,
    AsyncResilientDriver,
    BreakerConfig,
    BreakerRegistry,
    BreakerState,
    CircuitBreaker,
    ErrorAction,
    ErrorRule,
    ResilientDriver,
    RetryPolicy,
    Target,
    classify_error,
    clear_key_pools,
    key_id,
    parse_duration,
    register_error_rule,
    register_key_pool,
    reset_error_rules,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_error(status: int, text: str = "", headers: dict[str, str] | None = None) -> requests.HTTPError:
    resp = requests.Response()
    resp.status_code = status
    resp.headers.update(headers or {})
    resp._content = text.encode()
    return requests.HTTPError(f"{status} Error: {text}", response=resp)


def _meta(**extra: Any) -> dict[str, Any]:
    return {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "cost": 0.0, "raw_response": {}, **extra}


class FakeDriver(Driver):
    """Scripted driver: each call pops the next outcome (exception or text)."""

    supports_messages = True
    supports_streaming = True
    supports_tool_use = False

    def __init__(self, name: str, script: list[Any] | None = None, api_key: str | None = None) -> None:
        self.model = name
        self.api_key = api_key
        self.script = list(script or ["ok"])
        self.calls = 0

    def _next(self) -> Any:
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"text": f"{self.model}:{self._next()}", "meta": _meta(model_name=self.model)}

    def generate_messages(self, messages, options):
        return self.generate("", options)

    def generate_messages_stream(self, messages, options):
        item = self._next()
        if isinstance(item, list):  # [text chunks..., exception-or-None]
            *chunks, tail = item
        else:
            chunks, tail = [item], None
        for c in chunks:
            yield {"type": "delta", "text": c}
        if isinstance(tail, BaseException):
            raise tail
        yield {"type": "done", "text": "".join(chunks), "meta": _meta()}


class Factory:
    def __init__(self, scripts: dict[str, list[Any]]) -> None:
        self.scripts = scripts
        self.built: list[FakeDriver] = []

    def __call__(self, model: str, *, api_key: str | None = None, **_: Any) -> FakeDriver:
        key = f"{model}#{api_key}" if api_key and f"{model}#{api_key}" in self.scripts else model
        drv = FakeDriver(model, self.scripts.get(key, ["ok"]), api_key=api_key)
        self.built.append(drv)
        return drv


@pytest.fixture(autouse=True)
def _isolate():
    clear_key_pools()
    reset_error_rules()
    yield
    clear_key_pools()
    reset_error_rules()


def _driver(targets, scripts, **kw) -> tuple[ResilientDriver, Factory, list[float]]:
    sleeps: list[float] = []
    factory = Factory(scripts)
    policy = kw.pop("policy", RetryPolicy(max_attempts=2, base_delay=0.01, jitter=0.0, max_wait=5.0))
    drv = ResilientDriver(
        targets,
        policy=policy,
        breakers=BreakerRegistry(),
        factory=factory,
        sleep=sleeps.append,
        **kw,
    )
    return drv, factory, sleeps


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_driver_http_error_statuses(self):
        assert classify_error(DriverHTTPError("x", status_code=429)).action is ErrorAction.COOLDOWN
        assert classify_error(DriverHTTPError("x", status_code=502)).action is ErrorAction.RETRY
        assert classify_error(DriverHTTPError("x", status_code=401)).action is ErrorAction.DISABLE_KEY
        assert classify_error(DriverHTTPError("x", status_code=404)).category == "model_not_found"
        assert classify_error(DriverHTTPError("bad param", status_code=400)).action is ErrorAction.FATAL

    def test_requests_error_with_retry_after_header(self):
        info = classify_error(_http_error(429, "slow down", {"Retry-After": "7"}))
        assert info.status_code == 429
        assert info.action is ErrorAction.COOLDOWN
        assert info.retry_after == 7.0

    def test_retry_after_ms_header(self):
        info = classify_error(_http_error(503, "busy", {"retry-after-ms": "1500"}))
        assert info.retry_after == pytest.approx(1.5)

    def test_wrapped_error_walks_cause_chain(self):
        try:
            try:
                raise _http_error(503, "overloaded")
            except requests.HTTPError as inner:
                raise RuntimeError("Grok API request failed") from inner
        except RuntimeError as outer:
            info = classify_error(outer)
        assert info.status_code == 503
        assert info.category == "overloaded"
        assert info.action is ErrorAction.COOLDOWN

    def test_status_parsed_from_text(self):
        info = classify_error(RuntimeError("API request failed: 429 Too Many Requests"))
        assert info.status_code == 429
        assert info.action is ErrorAction.COOLDOWN
        info = classify_error(RuntimeError("Error code: 500 - {'error': 'boom'}"))
        assert info.status_code == 500
        assert info.action is ErrorAction.RETRY

    def test_insufficient_quota_beats_plain_rate_limit(self):
        info = classify_error(DriverHTTPError("insufficient_quota: add credits", status_code=429))
        assert info.category == "quota_exhausted"
        assert info.action is ErrorAction.DISABLE_KEY

    def test_context_length_on_400_is_failover_not_fatal(self):
        info = classify_error(DriverHTTPError("This model's maximum context length is 8192 tokens", status_code=400))
        assert info.category == "context_length"
        assert info.action is ErrorAction.FAILOVER

    def test_httpx_status_error(self):
        req = httpx.Request("POST", "https://example.test")
        resp = httpx.Response(529, request=req, headers={"retry-after": "2"})
        info = classify_error(httpx.HTTPStatusError("overloaded", request=req, response=resp))
        assert info.status_code == 529
        assert info.retry_after == 2.0

    def test_openai_sdk_error(self):
        openai = pytest.importorskip("openai")
        req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        resp = httpx.Response(429, request=req, headers={"retry-after": "3"})
        exc = openai.RateLimitError("Rate limit reached", response=resp, body=None)
        info = classify_error(exc)
        assert (info.status_code, info.action, info.retry_after) == (429, ErrorAction.COOLDOWN, 3.0)

    def test_transport_errors_retry(self):
        assert classify_error(requests.Timeout("read timed out")).category == "timeout"
        assert classify_error(requests.ConnectionError("refused")).category == "connection"
        assert classify_error(httpx.ConnectTimeout("t")).action is ErrorAction.RETRY
        assert classify_error(TimeoutError()).action is ErrorAction.RETRY

    def test_not_implemented_is_failover(self):
        assert classify_error(NotImplementedError("no tools")).action is ErrorAction.FAILOVER

    def test_retry_hint_in_text(self):
        info = classify_error(RuntimeError("Rate limit reached. Please try again in 1m30s."))
        assert info.retry_after == 90.0

    def test_custom_rule_takes_priority(self):
        register_error_rule(ErrorRule("vendor_hiccup", ErrorAction.RETRY, pattern=r"ERR_VENDOR_42", priority=200))
        info = classify_error(DriverHTTPError("ERR_VENDOR_42", status_code=400))
        assert (info.category, info.action) == ("vendor_hiccup", ErrorAction.RETRY)

    @pytest.mark.parametrize(
        ("text", "seconds"),
        [("20s", 20.0), ("250ms", 0.25), ("1m30s", 90.0), ("7.5 seconds", 7.5), ("2 minutes", 120.0), ("1h", 3600.0)],
    )
    def test_parse_duration(self, text, seconds):
        assert parse_duration(text) == pytest.approx(seconds)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestCircuitBreaker:
    def test_opens_after_threshold_and_recovers_lazily(self):
        clock = Clock()
        b = CircuitBreaker("model:x", BreakerConfig(failure_threshold=3, recovery_timeout=10), clock=clock)
        for _ in range(3):
            assert b.allow()
            b.record_failure("boom")
        assert b.state is BreakerState.OPEN
        assert not b.allow()

        clock.now += 10
        assert b.state is BreakerState.HALF_OPEN
        assert b.allow()  # the single probe
        assert not b.allow()  # second caller waits for the probe
        b.record_success()
        assert b.state is BreakerState.CLOSED
        assert b.allow()

    def test_failed_probe_reopens(self):
        clock = Clock()
        b = CircuitBreaker("model:x", BreakerConfig(failure_threshold=1, recovery_timeout=5), clock=clock)
        b.record_failure()
        clock.now += 5
        assert b.allow()
        b.record_failure()
        assert b.state is BreakerState.OPEN
        clock.now += 4.9
        assert not b.allow()

    def test_cooldown_blocks_then_expires(self):
        clock = Clock()
        b = CircuitBreaker("key:openai#abc", clock=clock)
        b.cooldown(30, "rate_limit")
        assert not b.allow()
        assert b.available_in() == pytest.approx(30)
        assert b.snapshot()["cooldown_reason"] == "rate_limit"
        clock.now += 30
        assert b.allow()

    def test_cooldown_never_shortens(self):
        clock = Clock()
        b = CircuitBreaker("s", clock=clock)
        b.cooldown(60)
        b.cooldown(5)
        assert b.available_in() == pytest.approx(60)


# ---------------------------------------------------------------------------
# ResilientDriver
# ---------------------------------------------------------------------------


class TestResilientDriver:
    def test_success_on_first_target_records_route(self):
        drv, _, sleeps = _driver(["a/one", "b/two"], {})
        resp = drv.generate("hi", {})
        assert resp["text"] == "a/one:ok"
        route = resp["meta"]["route"]
        assert route["served_by"] == "a/one"
        assert route["fallback"] is False
        assert [a["outcome"] for a in route["attempts"]] == ["ok"]
        assert sleeps == []

    def test_retries_transient_error_on_same_target(self):
        drv, _, sleeps = _driver(["a/one", "b/two"], {"a/one": [_http_error(502, "bad gateway"), "ok"]})
        resp = drv.generate("hi", {})
        assert resp["text"] == "a/one:ok"
        assert len(sleeps) == 1
        assert [a["outcome"] for a in resp["meta"]["route"]["attempts"]] == ["error", "ok"]

    def test_fails_over_after_retries_exhausted(self):
        drv, _, _ = _driver(["a/one", "b/two"], {"a/one": [_http_error(500, "boom")]})
        resp = drv.generate("hi", {})
        assert resp["text"] == "b/two:ok"
        route = resp["meta"]["route"]
        assert route["served_by"] == "b/two"
        assert route["fallback"] is True
        assert [a["model"] for a in route["attempts"]] == ["a/one", "a/one", "b/two"]

    def test_rate_limit_skips_straight_to_next_target_and_parks_it(self):
        drv, factory, sleeps = _driver(
            ["a/one", "b/two"], {"a/one": [_http_error(429, "slow", {"Retry-After": "120"}), "ok"]}
        )
        assert drv.generate("hi", {})["text"] == "b/two:ok"
        assert sleeps == []  # no inline wait when another target exists
        # Second call: a/one is still cooling down, so it's skipped, not called.
        resp = drv.generate("hi", {})
        assert resp["text"] == "b/two:ok"
        assert resp["meta"]["route"]["attempts"][0]["outcome"] == "skipped"
        a_driver = next(d for d in factory.built if d.model == "a/one")
        assert a_driver.calls == 1

    def test_single_target_waits_out_short_rate_limit(self):
        drv, _, sleeps = _driver(["a/one"], {"a/one": [_http_error(429, "slow", {"Retry-After": "2"}), "ok"]})
        assert drv.generate("hi", {})["text"] == "a/one:ok"
        assert sleeps == [2.0]

    def test_fatal_error_raises_without_failover(self):
        drv, factory, _ = _driver(["a/one", "b/two"], {"a/one": [_http_error(400, "invalid temperature")]})
        with pytest.raises(requests.HTTPError):
            drv.generate("hi", {})
        assert all(d.model != "b/two" for d in factory.built if d.calls)
        assert drv.last_route["attempts"][-1]["error"]["action"] == "fatal"

    def test_all_targets_failed(self):
        drv, _, _ = _driver(
            ["a/one", "b/two"],
            {"a/one": [_http_error(401, "bad key")], "b/two": [_http_error(404, "model not found")]},
        )
        with pytest.raises(AllTargetsFailedError) as ei:
            drv.generate("hi", {})
        err = ei.value
        assert [a["error"]["category"] for a in err.attempts] == ["auth", "model_not_found"]
        assert isinstance(err.last_error, requests.HTTPError)
        assert "a/one: auth 401" in str(err)

    def test_model_not_found_is_remembered(self):
        drv, factory, _ = _driver(["a/gone", "b/two"], {"a/gone": [_http_error(404, "no such model")]})
        drv.generate("hi", {})
        drv.generate("hi", {})
        gone = next(d for d in factory.built if d.model == "a/gone")
        assert gone.calls == 1

    def test_breaker_opens_after_repeated_failures(self):
        breakers = BreakerRegistry(configs={"model": BreakerConfig(failure_threshold=2, recovery_timeout=999)})
        factory = Factory({"a/one": [_http_error(500, "boom")]})
        drv = ResilientDriver(
            ["a/one", "b/two"],
            policy=RetryPolicy(max_attempts=1),
            breakers=breakers,
            factory=factory,
            sleep=lambda s: None,
        )
        drv.generate("hi", {})
        drv.generate("hi", {})
        resp = drv.generate("hi", {})
        assert resp["meta"]["route"]["attempts"][0]["outcome"] == "skipped"
        assert breakers.get("model:a/one").state is BreakerState.OPEN

    def test_key_pool_rotates_and_parks_bad_key(self):
        register_key_pool("a", ["k-good", "k-bad"])
        scripts = {"a/one#k-bad": [_http_error(401, "invalid api key")], "a/one#k-good": ["ok"]}
        drv, factory, _ = _driver(["a/one"], scripts)
        assert len(drv.targets) == 2
        seen = set()
        for _ in range(4):
            resp = drv.generate("hi", {})
            seen.add(resp["meta"]["route"]["key_id"])
        assert seen == {key_id("k-good")}
        bad = next(d for d in factory.built if d.api_key == "k-bad")
        assert bad.calls == 1  # disabled after the first 401

    def test_key_pool_from_env(self, monkeypatch):
        monkeypatch.setenv("ZZTEST_API_KEYS", "k1, k2 ,k1")
        drv, _, _ = _driver(["zztest/m"], {})
        assert sorted(t.api_key for t in drv.targets) == ["k1", "k2"]

    def test_explicit_target_and_instance(self):
        inst = FakeDriver("c/three", ["ok"])
        drv, factory, _ = _driver([Target("a/one", api_key="x"), inst], {"a/one": [_http_error(503, "down")]})
        assert drv.generate("hi", {})["text"] == "c/three:ok"
        assert factory.built[0].api_key == "x"

    def test_unavailable_target_is_skipped(self):
        def factory(model, *, api_key=None, **_):
            if model == "a/one":
                raise ValueError("missing API key")
            return FakeDriver(model)

        drv = ResilientDriver(["a/one", "b/two"], breakers=BreakerRegistry(), factory=factory, sleep=lambda s: None)
        resp = drv.generate("hi", {})
        assert resp["text"] == "b/two:ok"
        assert resp["meta"]["route"]["attempts"][0]["outcome"] == "unavailable"

    def test_capability_flags_are_union(self):
        class ToolDriver(FakeDriver):
            supports_tool_use = True

        drv = ResilientDriver(
            [FakeDriver("a/one"), ToolDriver("b/two")], breakers=BreakerRegistry(), sleep=lambda s: None
        )
        assert drv.supports_tool_use is True
        assert drv.supports_messages is True
        assert drv.supports_vision is False

    def test_tool_calls_skip_targets_without_tool_support(self):
        class ToolDriver(FakeDriver):
            supports_tool_use = True

            def generate_messages_with_tools(self, messages, tools, options):
                return {"text": "", "meta": _meta(), "tool_calls": [], "stop_reason": "end_turn"}

        drv = ResilientDriver(
            [FakeDriver("a/one"), ToolDriver("b/two")], breakers=BreakerRegistry(), sleep=lambda s: None
        )
        resp = drv.generate_messages_with_tools([], [], {})
        assert resp["meta"]["route"]["served_by"] == "b/two"
        assert resp["meta"]["route"]["attempts"][0]["outcome"] == "unsupported"

    def test_options_are_not_shared_between_attempts(self):
        class Mutating(FakeDriver):
            def generate(self, prompt, options):
                options["poisoned"] = True
                return super().generate(prompt, options)

        opts = {"temperature": 0}
        drv = ResilientDriver([Mutating("a/one")], breakers=BreakerRegistry(), sleep=lambda s: None)
        drv.generate("hi", opts)
        assert opts == {"temperature": 0}

    def test_hooks_wrapper_records_serving_model(self):
        seen = []

        from prompture.infra.callbacks import DriverCallbacks

        drv, _, _ = _driver(["a/one", "b/two"], {"a/one": [_http_error(503, "down")]})
        drv.callbacks = DriverCallbacks(on_response=seen.append)
        drv.generate_messages_with_hooks([{"role": "user", "content": "hi"}], {})
        assert seen[0]["meta"]["route"]["served_by"] == "b/two"


class TestResilientStreaming:
    def test_fails_over_before_first_chunk(self):
        drv, _, _ = _driver(["a/one", "b/two"], {"a/one": [_http_error(503, "down")], "b/two": [["he", "llo", None]]})
        events = list(drv.generate_messages_stream([], {}))
        assert "".join(e["text"] for e in events if e["type"] == "delta") == "hello"
        done = events[-1]
        assert done["type"] == "done"
        assert done["meta"]["route"]["served_by"] == "b/two"

    def test_error_after_first_chunk_propagates(self):
        drv, _, _ = _driver(["a/one", "b/two"], {"a/one": [["partial", _http_error(502, "cut")]]})
        gen = drv.generate_messages_stream([], {})
        assert next(gen) == {"type": "delta", "text": "partial"}
        with pytest.raises(requests.HTTPError):
            next(gen)

    def test_skips_non_streaming_targets(self):
        class NoStream(FakeDriver):
            supports_streaming = False

        drv = ResilientDriver(
            [NoStream("a/one"), FakeDriver("b/two", [["x", None]])], breakers=BreakerRegistry(), sleep=lambda s: None
        )
        events = list(drv.generate_messages_stream([], {}))
        assert events[-1]["meta"]["route"]["attempts"][0]["outcome"] == "unsupported"


# ---------------------------------------------------------------------------
# Integration with extraction
# ---------------------------------------------------------------------------


def test_ask_for_json_through_resilient_driver():
    from prompture.extraction.core import ask_for_json

    class JsonDriver(FakeDriver):
        def generate(self, prompt, options):
            self._next()
            return {"text": '{"name": "Ada"}', "meta": _meta()}

    primary = JsonDriver("a/one", [_http_error(503, "down")])
    backup = JsonDriver("b/two", ["ok"])
    drv = ResilientDriver(
        [primary, backup], policy=RetryPolicy(max_attempts=1), breakers=BreakerRegistry(), sleep=lambda s: None
    )
    result = ask_for_json(
        drv,
        "Ada Lovelace",
        {"type": "object", "properties": {"name": {"type": "string"}}},
        json_mode="off",
    )
    assert result["json_object"] == {"name": "Ada"}
    assert backup.calls == 1


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


class AsyncFake(AsyncDriver):
    supports_messages = True
    supports_streaming = True

    def __init__(self, name: str, script: list[Any]) -> None:
        self.model = name
        self.script = list(script)
        self.calls = 0

    def _next(self) -> Any:
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item

    async def generate(self, prompt, options):
        return {"text": f"{self.model}:{self._next()}", "meta": _meta()}

    async def generate_messages_stream(self, messages, options):
        item = self._next()
        yield {"type": "delta", "text": item}
        yield {"type": "done", "text": item, "meta": _meta()}


class TestAsyncResilientDriver:
    def _make(self, drivers):
        sleeps: list[float] = []

        async def fake_sleep(s: float) -> None:
            sleeps.append(s)

        drv = AsyncResilientDriver(
            drivers,
            policy=RetryPolicy(max_attempts=2, base_delay=0.01, jitter=0.0),
            breakers=BreakerRegistry(),
            sleep=fake_sleep,
        )
        return drv, sleeps

    def test_failover(self):
        drv, sleeps = self._make([AsyncFake("a/one", [_http_error(500, "x")]), AsyncFake("b/two", ["ok"])])
        resp = asyncio.run(drv.generate("hi", {}))
        assert resp["text"] == "b/two:ok"
        assert resp["meta"]["route"]["fallback"] is True
        assert len(sleeps) == 1

    def test_stream_failover(self):
        drv, _ = self._make([AsyncFake("a/one", [_http_error(503, "x")]), AsyncFake("b/two", ["hey"])])

        async def collect():
            return [e async for e in drv.generate_messages_stream([], {})]

        events = asyncio.run(collect())
        assert events[0] == {"type": "delta", "text": "hey"}
        assert events[-1]["meta"]["route"]["served_by"] == "b/two"

    def test_all_failed(self):
        drv, _ = self._make([AsyncFake("a/one", [_http_error(401, "nope")])])
        with pytest.raises(AllTargetsFailedError):
            asyncio.run(drv.generate("hi", {}))


def test_conversation_runs_over_resilient_driver():
    from prompture import Conversation

    primary = FakeDriver("a/one", [_http_error(503, "down")])
    backup = FakeDriver("b/two", ["ok"])
    drv = ResilientDriver(
        [primary, backup], policy=RetryPolicy(max_attempts=1), breakers=BreakerRegistry(), sleep=lambda s: None
    )
    conv = Conversation(driver=drv)
    assert conv.ask("hello") == "b/two:ok"
    assert conv.ask("again") == "b/two:ok"
    assert drv.last_route["served_by"] == "b/two"
