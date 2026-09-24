"""OpenTelemetry spans for driver calls, following the GenAI semantic conventions.

Plugs into :class:`~prompture.infra.callbacks.DriverCallbacks`, so it works
for every driver without touching driver code::

    from prompture import instrument_driver
    from prompture.drivers import get_driver_for_model

    driver = instrument_driver(get_driver_for_model("openai/gpt-4o"))

Each call becomes one ``chat <model>`` span carrying ``gen_ai.*`` attributes
(system, request/response model, token usage) plus ``prompture.*`` extras
(cost, and for resilient drivers who served the call and whether it fell
back). Prompt and completion text are only recorded with
``capture_content=True``.

Requires ``opentelemetry-api`` (``pip install prompture[otel]``) unless you
pass your own tracer object.
"""

from __future__ import annotations

import contextvars
import time
from typing import Any

from .callbacks import DriverCallbacks

# Per-context stack of (start_ns, request_info) so nested calls pair correctly.
_stack: contextvars.ContextVar[tuple[tuple[int, dict[str, Any]], ...]] = contextvars.ContextVar(
    "prompture_otel_stack", default=()
)

_CONTENT_LIMIT = 4000


def _default_tracer() -> Any:
    try:
        from opentelemetry import trace
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "OpenTelemetry tracing needs opentelemetry-api: pip install prompture[otel] (or pass tracer=...)"
        ) from exc
    return trace.get_tracer("prompture")


def _push(info: dict[str, Any]) -> None:
    _stack.set((*_stack.get(), (time.time_ns(), info)))


def _pop() -> tuple[int, dict[str, Any]]:
    stack = _stack.get()
    if not stack:
        return time.time_ns(), {}
    _stack.set(stack[:-1])
    return stack[-1]


def _split(model: str | None) -> tuple[str | None, str | None]:
    if not model:
        return None, None
    if "/" in model:
        provider, rest = model.split("/", 1)
        return provider, rest
    return None, model


def _set_error(span: Any, exc: BaseException) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, str(exc)[:500]))
    except ImportError:
        span.set_status("ERROR")
    record = getattr(span, "record_exception", None)
    if callable(record):
        record(exc)


def otel_callbacks(
    tracer: Any = None,
    *,
    capture_content: bool = False,
    base: DriverCallbacks | None = None,
    operation: str = "chat",
) -> DriverCallbacks:
    """Callbacks that emit one span per driver call.

    Args:
        tracer: An OpenTelemetry ``Tracer`` (or anything with a compatible
            ``start_span(name, start_time=..., attributes=...)``). Defaults to
            ``opentelemetry.trace.get_tracer("prompture")``.
        capture_content: Also record prompt/completion text (truncated).
        base: Existing callbacks to keep firing alongside the tracing ones.
        operation: ``gen_ai.operation.name`` value.
    """
    tracer = tracer if tracer is not None else _default_tracer()

    def on_request(info: dict[str, Any]) -> None:
        _push(info)
        if base and base.on_request:
            base.on_request(info)

    def _start(info: dict[str, Any], start_ns: int, *, response_model: str | None = None) -> Any:
        request_model = str(info.get("driver") or "")
        provider, _ = _split(response_model or request_model)
        attrs: dict[str, Any] = {"gen_ai.operation.name": operation, "gen_ai.request.model": request_model}
        if provider:
            attrs["gen_ai.system"] = provider
        opts = info.get("options") or {}
        for key, attr in (
            ("temperature", "gen_ai.request.temperature"),
            ("max_tokens", "gen_ai.request.max_tokens"),
            ("top_p", "gen_ai.request.top_p"),
        ):
            if opts.get(key) is not None:
                attrs[attr] = opts[key]
        return tracer.start_span(f"{operation} {request_model}".strip(), start_time=start_ns, attributes=attrs)

    def on_response(info: dict[str, Any]) -> None:
        start_ns, req = _pop()
        meta = info.get("meta") or {}
        response_model = meta.get("model_name") or (meta.get("route") or {}).get("served_by")
        span = _start(
            {**req, "driver": info.get("driver") or req.get("driver")}, start_ns, response_model=response_model
        )
        if response_model:
            span.set_attribute("gen_ai.response.model", str(response_model))
        span.set_attribute("gen_ai.usage.input_tokens", int(meta.get("prompt_tokens", 0) or 0))
        span.set_attribute("gen_ai.usage.output_tokens", int(meta.get("completion_tokens", 0) or 0))
        if meta.get("cost") is not None:
            span.set_attribute("prompture.cost_usd", float(meta.get("cost") or 0.0))
        if meta.get("stop_reason"):
            span.set_attribute("gen_ai.response.finish_reasons", [str(meta["stop_reason"])])
        route = meta.get("route")
        if route:
            span.set_attribute("prompture.route.served_by", str(route.get("served_by")))
            span.set_attribute("prompture.route.fallback", bool(route.get("fallback")))
            span.set_attribute("prompture.route.attempts", len(route.get("attempts") or []))
            if route.get("strategy"):
                span.set_attribute("prompture.route.strategy", str(route["strategy"]))
        if capture_content:
            prompt = req.get("prompt") or req.get("messages")
            if prompt is not None:
                span.set_attribute("gen_ai.prompt", str(prompt)[:_CONTENT_LIMIT])
            span.set_attribute("gen_ai.completion", str(info.get("text", ""))[:_CONTENT_LIMIT])
        span.end()
        if base and base.on_response:
            base.on_response(info)

    def on_error(info: dict[str, Any]) -> None:
        start_ns, req = _pop()
        span = _start({**req, **info}, start_ns)
        err = info.get("error")
        span.set_attribute("error.type", type(err).__name__ if err is not None else "unknown")
        if isinstance(err, BaseException):
            _set_error(span, err)
            attempts = getattr(err, "attempts", None)
            if attempts:
                span.set_attribute("prompture.route.attempts", len(attempts))
        span.end()
        if base and base.on_error:
            base.on_error(info)

    return DriverCallbacks(
        on_request=on_request,
        on_response=on_response,
        on_error=on_error,
        on_stream_delta=base.on_stream_delta if base else None,
    )


def instrument_driver(driver: Any, tracer: Any = None, *, capture_content: bool = False) -> Any:
    """Attach OpenTelemetry callbacks to *driver* (keeping existing ones). Returns the driver."""
    driver.callbacks = otel_callbacks(tracer, capture_content=capture_content, base=getattr(driver, "callbacks", None))
    return driver
