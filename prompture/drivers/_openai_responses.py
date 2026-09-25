"""Optional Responses API transport for the existing OpenAI drivers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..infra.cost_mixin import prepare_strict_schema
from ..infra.rate_limits import LimitSnapshot, add_rate_limits, capture_rate_limits, limits_from_response
from ._usage_reporting import as_dict, text_value, usage_meta, value
from .base import _normalize_stop_reason, _tool_call_dict


def responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Prompture's chat history, including tool results and images."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        if role == "tool":
            content = message.get("content", "")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )
            continue
        content = message.get("content")
        if content is not None:
            if isinstance(content, list):
                blocks = []
                for original in content:
                    block = dict(original)
                    kind = block.get("type")
                    if kind == "text":
                        block["type"] = "output_text" if role == "assistant" else "input_text"
                        if role == "assistant":
                            block.setdefault("annotations", [])
                    elif kind == "image_url":
                        image = block["image_url"]
                        block = {
                            "type": "input_image",
                            "image_url": image["url"],
                            "detail": image.get("detail", "auto"),
                        }
                    blocks.append(block)
                content = blocks
            items.append({"role": role, "content": content})
        for call in message.get("tool_calls", []) or []:
            function = call["function"]
            arguments = function.get("arguments", "{}")
            items.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": function["name"],
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                }
            )
    return items


def responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"type": "function", **tool["function"]}
        if tool.get("type") == "function" and "function" in tool
        else dict(tool)
        for tool in tools
    ]


def build_request(
    driver: Any,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    *,
    counting: bool = False,
) -> dict[str, Any]:
    model = options.get("model", driver.model)
    request: dict[str, Any] = {"model": model, "input": responses_input(messages)}
    if tools:
        request["tools"] = responses_tools(tools)
    if options.get("json_mode"):
        schema = options.get("json_schema")
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "extraction",
                "strict": True,
                "schema": prepare_strict_schema(schema),
            }
            if schema
            else {"type": "json_object"}
        }
    for key in ("instructions", "previous_response_id", "reasoning", "text", "truncation"):
        if key in options:
            request[key] = options[key]
    if "reasoning_effort" in options and "reasoning" not in request:
        request["reasoning"] = {"effort": options["reasoning_effort"]}
    if counting:
        return request
    request["max_output_tokens"] = options.get("max_output_tokens", options.get("max_tokens", 4096 if tools else 512))
    # Preserve the existing driver's stateless behavior. Callers can explicitly
    # enable storage when they want server-managed conversation state.
    request["store"] = options.get("store", False)
    for key in (
        "service_tier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "prompt_cache_options",
        "parallel_tool_calls",
        "include",
        "timeout",
        "metadata",
    ):
        if key in options:
            request[key] = options[key]
    if "temperature" in options and driver._get_model_config("openai", model)["supports_temperature"]:
        request["temperature"] = options["temperature"]
    if "tool_choice" in options:
        choice = options["tool_choice"]
        if isinstance(choice, dict) and "function" in choice:
            choice = {"type": "function", "name": choice["function"]["name"]}
        elif isinstance(choice, dict) and "name" in choice:
            choice = {"type": "function", **choice}
        request["tool_choice"] = choice
    return request


def result_from_response(driver: Any, response: Any, model: str, options: dict[str, Any]) -> dict[str, Any]:
    status = text_value(response, "status")
    if status in ("failed", "cancelled"):
        from ..exceptions import DriverError

        raise DriverError(f"OpenAI Responses request {status}")
    reason = value(value(response, "incomplete_details"), "reason") if status == "incomplete" else "stop"
    raw_reason = "length" if reason == "max_output_tokens" else reason
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in value(response, "output", []) or []:
        if value(item, "type") == "message":
            for part in value(item, "content", []) or []:
                if value(part, "type") == "output_text":
                    text_parts.append(value(part, "text", ""))
                elif value(part, "type") == "refusal":
                    text_parts.append(value(part, "refusal", ""))
        elif value(item, "type") == "function_call":
            tool_calls.append(
                _tool_call_dict(value(item, "call_id"), value(item, "name"), value(item, "arguments", "{}"), raw_reason)
            )
    meta = usage_meta(
        driver, "openai", model, value(response, "usage"), response=response, options=options, responses_api=True
    )
    meta["raw_response"] = as_dict(response)
    meta["raw_stop_reason"] = raw_reason
    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls,
        "meta": meta,
        "stop_reason": _normalize_stop_reason(raw_reason, tool_calls_present=bool(tool_calls)),
    }


def generate(
    driver: Any, messages: list[dict[str, Any]], options: dict[str, Any], tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    request = build_request(driver, messages, options, tools)
    with capture_rate_limits() as limits:
        response = driver.client.responses.create(**request)
    result = result_from_response(driver, response, request["model"], options)
    add_rate_limits(result["meta"], limits.snapshot)
    return result


async def agenerate(
    driver: Any, messages: list[dict[str, Any]], options: dict[str, Any], tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    request = build_request(driver, messages, options, tools)
    with capture_rate_limits() as limits:
        response = await driver.client.responses.create(**request)
    result = result_from_response(driver, response, request["model"], options)
    add_rate_limits(result["meta"], limits.snapshot)
    return result


def _events(event: Any, state: dict[str, Any], live: bool) -> Iterator[Any]:
    from ..agents.live_events import TextDelta, ThinkingDelta, ToolInputDelta, ToolUseStart, ToolUseStop

    kind = value(event, "type")
    if kind == "response.output_text.delta":
        delta = value(event, "delta", "")
        state["text"] += delta
        yield TextDelta(delta) if live else {"type": "delta", "text": delta}
    elif kind == "response.reasoning_summary_text.delta":
        delta = value(event, "delta", "")
        yield ThinkingDelta(delta) if live else {"type": "thinking_delta", "text": delta}
    elif kind == "response.output_item.added":
        item = value(event, "item")
        if value(item, "type") == "function_call":
            state["tools"][value(item, "id")] = as_dict(item)
            if live:
                yield ToolUseStart(id=value(item, "call_id"), name=value(item, "name"))
    elif kind == "response.function_call_arguments.delta" and live:
        item = state["tools"].get(value(event, "item_id"))
        if item:
            yield ToolInputDelta(id=item["call_id"], fragment=value(event, "delta", ""))
    elif kind == "response.output_item.done" and live:
        item = value(event, "item")
        if value(item, "type") == "function_call":
            if value(item, "id") not in state["tools"]:
                yield ToolUseStart(id=value(item, "call_id"), name=value(item, "name"))
            call = _tool_call_dict(value(item, "call_id"), value(item, "name"), value(item, "arguments", "{}"), None)
            if call.get("arguments_error"):
                # The terminal event follows output_item.done. Wait for its
                # reason before deciding whether malformed JSON was truncated.
                state.setdefault("pending_tool_stops", []).append(call)
            else:
                yield ToolUseStop(id=call["id"], name=call["name"], input=call["arguments"])
    elif kind in ("response.completed", "response.incomplete", "response.failed"):
        state["response"] = value(event, "response")
    elif kind == "error":
        from ..exceptions import DriverError

        raise DriverError("OpenAI Responses stream returned an error")


def _pending_tool_stops(state: dict[str, Any]) -> Iterator[Any]:
    from ..agents.live_events import ToolUseStop

    response = state["response"]
    status = text_value(response, "status")
    if status == "incomplete":
        reason = text_value(value(response, "incomplete_details"), "reason")
    else:
        reason = "stop" if status == "completed" else "error"
    for call in state.get("pending_tool_stops", []):
        yield ToolUseStop(
            id=call["id"],
            name=call["name"],
            input=call["arguments"],
            truncated=reason == "max_output_tokens",
            raw_stop_reason=reason,
        )


def _done(
    driver: Any,
    state: dict[str, Any],
    model: str,
    options: dict[str, Any],
    live: bool,
    limits: LimitSnapshot | None = None,
) -> Any:
    from ..agents.live_events import MessageStop

    if state["response"] is not None:
        result = result_from_response(driver, state["response"], model, options)
        add_rate_limits(result["meta"], limits)
        if live:
            return MessageStop(stop_reason=result["stop_reason"], usage=result["meta"])
        return {"type": "done", "text": result["text"], "meta": result["meta"]}
    meta = usage_meta(driver, "openai", model, None, options=options, complete=False, responses_api=True)
    meta["raw_response"] = {}
    add_rate_limits(meta, limits)
    return (
        MessageStop(stop_reason="error", usage=meta) if live else {"type": "done", "text": state["text"], "meta": meta}
    )


def stream(
    driver: Any, messages: list[dict[str, Any]], options: dict[str, Any], tools: list[dict[str, Any]] | None = None
) -> Iterator[Any]:
    request = build_request(driver, messages, options, tools)
    response_stream = driver.client.responses.create(**request, stream=True)
    state: dict[str, Any] = {"text": "", "tools": {}, "response": None}
    try:
        for event in response_stream:
            yield from _events(event, state, tools is not None)
        yield from _pending_tool_stops(state)
        limits = limits_from_response(getattr(response_stream, "response", None))
        yield _done(driver, state, request["model"], options, tools is not None, limits)
    finally:
        if callable(getattr(response_stream, "close", None)):
            response_stream.close()


async def astream(
    driver: Any, messages: list[dict[str, Any]], options: dict[str, Any], tools: list[dict[str, Any]] | None = None
) -> AsyncIterator[Any]:
    request = build_request(driver, messages, options, tools)
    response_stream = await driver.client.responses.create(**request, stream=True)
    state: dict[str, Any] = {"text": "", "tools": {}, "response": None}
    try:
        async for event in response_stream:
            for output in _events(event, state, tools is not None):
                yield output
        for output in _pending_tool_stops(state):
            yield output
        limits = limits_from_response(getattr(response_stream, "response", None))
        yield _done(driver, state, request["model"], options, tools is not None, limits)
    finally:
        if callable(getattr(response_stream, "close", None)):
            await response_stream.close()
