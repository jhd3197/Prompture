"""Shared Gemini wire format for Developer API and Vertex, sync and async."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..exceptions import DriverError
from ._effort import apply_effort
from .base import _normalize_stop_reason, _translate_tool_choice


def build_generation_args(
    driver: Any, messages: list[dict[str, Any]], options: dict[str, Any] | None, types: Any
) -> tuple[Any, dict[str, Any]]:
    """Build config and replay signed model parts without changing their order."""
    merged = {**driver.options, **(options or {})}
    model = driver.model
    config: dict[str, Any] = {}
    for key in ("temperature", "top_p", "top_k"):
        if key in merged:
            config[key] = merged[key]
    if "max_tokens" in merged:
        config["max_output_tokens"] = merged["max_tokens"]
    for key, value in merged.get("generation_config", {}).items():
        config.setdefault(key, value)
    apply_effort(config, merged, "google", model=model)
    if not driver._get_model_config("google", model)["supports_temperature"]:
        config.pop("temperature", None)
    if model.removeprefix("models/") == "gemini-4-argon":
        # Argon ignores these sampling parameters, even when explicitly supplied.
        for key in ("temperature", "top_p", "top_k"):
            config.pop(key, None)
    if merged.get("safety_settings"):
        config["safety_settings"] = merged["safety_settings"]
    if merged.get("json_mode"):
        config["response_mime_type"] = "application/json"
        if merged.get("json_schema"):
            config["response_schema"] = merged["json_schema"]

    contents: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    api_call_ids: set[str] = set()
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if role == "system":
            config["system_instruction"] = content if isinstance(content, str) else str(content)
            continue
        if role == "assistant":
            calls = msg.get("tool_calls") or []
            for call in calls:
                call_names[call.get("id", "")] = call.get("function", call).get("name", "")
            if msg.get("_google_parts") is not None:
                # SDK JSON serialization encodes opaque bytes as base64. Its
                # validator restores them, including non-UTF8 thought signatures.
                parts = [types.Part.model_validate(part) for part in msg["_google_parts"]]
                api_call_ids.update(p.function_call.id for p in parts if p.function_call and p.function_call.id)
            elif calls:
                parts = [content] if content else []
                for call in calls:
                    fn = call.get("function", call)
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    parts.append(types.Part(function_call=types.FunctionCall(name=fn.get("name", ""), args=args)))
            else:
                parts = content if msg.get("_vision_parts") else [content]
            contents.append({"role": "model", "parts": parts})
        elif role == "tool":
            call_id = msg.get("tool_call_id", "")
            name = call_names.get(call_id)
            if name is None:
                raise DriverError(
                    f"Tool result message references unknown tool_call_id {call_id!r}: "
                    "no matching assistant tool call exists in the conversation history."
                )
            result = content
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    result = {"result": result}
            if not isinstance(result, dict):
                result = {"result": str(result)}
            result_part = types.Part(
                function_response=types.FunctionResponse(
                    name=name,
                    response=result,
                    id=call_id if call_id in api_call_ids else None,
                )
            )
            # All results of parallel calls belong to one user turn.
            if (
                contents
                and contents[-1]["role"] == "user"
                and all(getattr(p, "function_response", None) is not None for p in contents[-1]["parts"])
            ):
                contents[-1]["parts"].append(result_part)
            else:
                contents.append({"role": "user", "parts": [result_part]})
        else:
            contents.append({"role": "user", "parts": content if msg.get("_vision_parts") else [content]})
    if len(contents) == 1 and len(contents[0]["parts"]) == 1 and isinstance(contents[0]["parts"][0], str):
        return contents[0]["parts"][0], config
    return contents, config


def add_tools(config: dict[str, Any], tools: list[dict[str, Any]], options: dict[str, Any], types: Any) -> None:
    """Translate portable tool definitions and tool choice."""
    declarations = []
    for tool in tools:
        fn = tool["function"] if tool.get("type") == "function" else tool
        if "name" not in fn:
            continue
        params = fn.get("parameters") or fn.get("input_schema")
        declarations.append(
            types.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                **({"parameters_json_schema": params} if params else {}),
            )
        )
    config["tools"] = [types.Tool(function_declarations=declarations)]
    choice = _translate_tool_choice(options.get("tool_choice"), "google")
    if choice is not None:
        config["tool_config"] = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(**choice))


def response_parts(response: Any) -> list[dict[str, Any]]:
    """JSON-safe history state for the selected candidate (never combine alternatives)."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    content = candidates[0].content
    return [part.model_dump(mode="json", exclude_none=True) for part in (content.parts or [])] if content else []


def tool_response(response: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """Expose portable calls while retaining the exact signed parts for replay."""
    text = ""
    calls = []
    candidates = response.candidates or []
    candidate = candidates[0] if candidates else None
    parts = candidate.content.parts if candidate and candidate.content else []
    for part in parts or []:
        if part.text and not part.thought:
            text += part.text
        fc = part.function_call
        if fc and fc.name:
            calls.append({"id": fc.id or str(uuid.uuid4()), "name": fc.name, "arguments": dict(fc.args or {})})
    reason = getattr(candidate, "finish_reason", None)
    meta["raw_stop_reason"] = str(reason) if reason is not None else None
    return {
        "text": text,
        "meta": meta,
        "tool_calls": calls,
        "stop_reason": _normalize_stop_reason(reason, tool_calls_present=bool(calls)),
        "_google_parts": response_parts(response),
    }


def extract_usage(response: Any, messages: list[dict[str, Any]], calculate_cost: Any) -> dict[str, Any]:
    """Bill all generated tokens, including thoughts, with cached-input discounts."""
    usage = getattr(response, "usage_metadata", None)

    def count(name: str) -> int:
        value = getattr(usage, name, 0)
        return int(value) if isinstance(value, (int, float)) else 0

    reasoning = count("thoughts_token_count")
    cached = count("cached_content_token_count")
    if usage:
        prompt = count("prompt_token_count")
        completion = count("candidates_token_count") + reasoning
        total = count("total_token_count") or prompt + completion
    else:
        chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                chars += sum(
                    len(p if isinstance(p, str) else p.get("text", "")) for p in content if isinstance(p, (str, dict))
                )
        prompt = chars // 4
        completion = len(getattr(response, "text", "") or "") // 4
        total = prompt + completion
    meta = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "cached_prompt_tokens": cached,
        "cost": round(calculate_cost(prompt, completion, cached_tokens=cached), 6),
    }
    initial = count("initial_prompt_token_count")
    if initial:
        meta["initial_prompt_tokens"] = initial
    return meta
