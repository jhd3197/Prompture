"""Route explanations and OpenTelemetry callbacks."""

from __future__ import annotations

from typing import Any

import pytest
from test_resilience import FakeDriver, _http_error

from prompture.infra.callbacks import DriverCallbacks
from prompture.infra.otel import instrument_driver, otel_callbacks
from prompture.resilience import AllTargetsFailedError, BreakerRegistry, ResilientDriver, RetryPolicy, explain_route


class FakeSpan:
    def __init__(self, name: str, start_time: int, attributes: dict[str, Any]) -> None:
        self.name = name
        self.start_time = start_time
        self.attributes = dict(attributes)
        self.status: Any = None
        self.exceptions: list[BaseException] = []
        self.ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)

    def end(self) -> None:
        self.ended = True


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_span(self, name: str, start_time: int, attributes: dict[str, Any]) -> FakeSpan:
        span = FakeSpan(name, start_time, attributes)
        self.spans.append(span)
        return span


def _resilient(*drivers: FakeDriver) -> ResilientDriver:
    return ResilientDriver(
        list(drivers), policy=RetryPolicy(max_attempts=1), breakers=BreakerRegistry(), sleep=lambda s: None
    )


class TestExplainRoute:
    def test_fallback_story(self):
        drv = _resilient(FakeDriver("a/one", [_http_error(429, "slow", {"Retry-After": "20"})]), FakeDriver("b/two"))
        route = drv.generate("hi", {})["meta"]["route"]
        text = explain_route(route)
        assert text.splitlines()[0] == "Served by b/two after 2 attempts"
        assert "a/one — rate_limit (429), retry after 20s, parked" in text
        assert "b/two — ok" in text

    def test_failure(self):
        drv = _resilient(FakeDriver("a/one", [_http_error(401, "bad key")]))
        with pytest.raises(AllTargetsFailedError) as ei:
            drv.generate("hi", {})
        text = explain_route({"attempts": ei.value.attempts})
        assert text.startswith("Failed after 1 attempt")
        assert "auth (401)" in text and "key disabled" in text

    def test_skipped_and_empty(self):
        route = {
            "served_by": "b/two",
            "strategy": "latency",
            "attempts": [
                {"model": "a/one", "key_id": "abcd1234", "outcome": "skipped", "available_in": 12.5},
                {"model": "b/two", "outcome": "ok", "elapsed_ms": 88.0},
            ],
        }
        text = explain_route(route)
        assert "strategy: latency" in text
        assert "a/one (key abcd1234) — skipped, unavailable for 12.5s" in text
        assert explain_route(None).startswith("No routing information")


class TestOtel:
    def test_span_per_call_with_genai_attributes(self):
        tracer = FakeTracer()
        drv = instrument_driver(
            _resilient(FakeDriver("a/one", [_http_error(503, "down")]), FakeDriver("b/two")), tracer
        )
        drv.generate_messages_with_hooks([{"role": "user", "content": "hi"}], {"temperature": 0.3, "max_tokens": 50})

        (span,) = tracer.spans
        assert span.ended
        assert span.name == "chat a/one"
        a = span.attributes
        assert a["gen_ai.operation.name"] == "chat"
        assert a["gen_ai.request.model"] == "a/one"
        assert a["gen_ai.response.model"] == "b/two"
        assert a["gen_ai.system"] == "b"
        assert a["gen_ai.request.temperature"] == 0.3
        assert a["gen_ai.request.max_tokens"] == 50
        assert (a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"]) == (1, 2)
        assert a["prompture.route.served_by"] == "b/two"
        assert a["prompture.route.fallback"] is True
        assert a["prompture.route.attempts"] == 2
        assert "gen_ai.prompt" not in a

    def test_error_span(self):
        tracer = FakeTracer()
        drv = instrument_driver(_resilient(FakeDriver("a/one", [_http_error(401, "nope")])), tracer)
        with pytest.raises(AllTargetsFailedError):
            drv.generate_with_hooks("hi", {})
        (span,) = tracer.spans
        assert span.attributes["error.type"] == "AllTargetsFailedError"
        assert span.status is not None
        assert span.exceptions and span.ended

    def test_capture_content_and_chained_callbacks(self):
        tracer = FakeTracer()
        seen: list[str] = []
        drv = FakeDriver("a/one")
        drv.callbacks = DriverCallbacks(on_response=lambda info: seen.append(info["text"]))
        instrument_driver(drv, tracer, capture_content=True)
        drv.generate_with_hooks("what is up", {})
        assert seen == ["a/one:ok"]
        span = tracer.spans[0]
        assert span.attributes["gen_ai.prompt"] == "what is up"
        assert span.attributes["gen_ai.completion"] == "a/one:ok"

    def test_nested_calls_pair_start_times(self):
        tracer = FakeTracer()
        cbs = otel_callbacks(tracer)
        cbs.on_request({"driver": "a/outer", "options": {"temperature": 0.9}})
        cbs.on_request({"driver": "a/inner", "options": {"temperature": 0.1}})
        cbs.on_response({"driver": "a/inner", "meta": {}, "text": ""})
        cbs.on_response({"driver": "a/outer", "meta": {}, "text": ""})
        inner, outer = tracer.spans
        assert inner.start_time >= outer.start_time
        assert outer.attributes["gen_ai.request.temperature"] == 0.9
        assert inner.attributes["gen_ai.request.temperature"] == 0.1
