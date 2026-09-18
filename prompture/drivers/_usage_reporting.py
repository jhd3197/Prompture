"""Normalize provider usage without retaining prompts in reporting metadata."""

from __future__ import annotations

from typing import Any


def value(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if callable(getattr(obj, "model_dump", None)):
        result = obj.model_dump()
        return result if isinstance(result, dict) else {}
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def count(obj: Any, key: str) -> int:
    number = value(obj, key, 0)
    return max(0, int(number)) if isinstance(number, (int, float)) else 0


def text_value(obj: Any, key: str) -> str | None:
    item = value(obj, key)
    return item if isinstance(item, str) else None


def usage_meta(
    driver: Any,
    provider: str,
    model: str,
    usage: Any,
    *,
    response: Any = None,
    options: dict[str, Any] | None = None,
    complete: bool = True,
    responses_api: bool = False,
) -> dict[str, Any]:
    """Return counts, pricing provenance and safe request identity for one call.

    Reasoning and predicted tokens are informational subsets of output tokens;
    they must never be added to completion usage a second time.
    """
    options = options or {}
    required = (
        ("input_tokens", "output_tokens")
        if provider == "claude" or responses_api
        else ("prompt_tokens", "completion_tokens")
    )
    complete = complete and all(isinstance(value(usage, key), (int, float)) for key in required)
    returned_model = text_value(response, "model")
    effective_model = returned_model or model
    service_tier = text_value(response, "service_tier") or text_value(usage, "service_tier")
    service_tier = service_tier or options.get("service_tier")
    geo = text_value(usage, "inference_geo") or options.get("inference_geo")
    details: dict[str, Any] = {}
    cost_options: dict[str, Any] = {}
    if provider == "claude":
        uncached = count(usage, "input_tokens")
        cached = count(usage, "cache_read_input_tokens")
        writes = count(usage, "cache_creation_input_tokens")
        output = count(usage, "output_tokens")
        prompt = uncached + cached + writes
        creation = value(usage, "cache_creation")
        if any(isinstance(value(creation, f"ephemeral_{ttl}_input_tokens"), (int, float)) for ttl in ("5m", "1h")):
            for ttl in ("5m", "1h"):
                tokens = count(creation, f"ephemeral_{ttl}_input_tokens")
                details[f"cache_creation_{ttl}_tokens"] = tokens
                cost_options[f"cache_creation_{ttl}_tokens"] = tokens
        from ._prompt_cache import cache_write_multiplier

        cost_options["cache_write_multiplier"] = cache_write_multiplier(options.get("cache_ttl", "5m"))
        tool_usage = as_dict(value(usage, "server_tool_use"))
        # Iteration counts explain server-side loops without double-counting
        # their tokens, which are already represented in top-level usage.
        iterations = value(usage, "iterations")
        if isinstance(iterations, list):
            details["iterations"] = [as_dict(item) for item in iterations]
        for key in ("speed", "inference_geo", "service_tier"):
            if text_value(usage, key):
                details[key] = text_value(usage, key)
    else:
        prompt = count(usage, "input_tokens" if responses_api else "prompt_tokens")
        output = count(usage, "output_tokens" if responses_api else "completion_tokens")
        input_details = value(usage, "input_tokens_details" if responses_api else "prompt_tokens_details")
        output_details = value(usage, "output_tokens_details" if responses_api else "completion_tokens_details")
        cached = count(input_details, "cached_tokens")
        writes = count(input_details, "cache_write_tokens")
        uncached = max(0, prompt - cached - writes)
        for key in (
            "reasoning_tokens",
            "accepted_prediction_tokens",
            "rejected_prediction_tokens",
            "audio_tokens",
            "text_tokens",
        ):
            if isinstance(value(output_details, key), (int, float)):
                details[key if key not in ("audio_tokens", "text_tokens") else f"output_{key}"] = count(
                    output_details, key
                )
        for key in ("audio_tokens", "image_tokens", "text_tokens"):
            if isinstance(value(input_details, key), (int, float)):
                details[f"input_{key}"] = count(input_details, key)
        tool_usage = {}
        declared_tools = value(response, "tools", [])
        if not isinstance(declared_tools, list):
            declared_tools = []
        search_types = {
            text_value(tool, "type")
            for tool in declared_tools
            if text_value(tool, "type") in ("web_search", "web_search_preview", "web_search_preview_2025_03_11")
        }
        shell_environments = {
            text_value(value(tool, "environment"), "type")
            for tool in declared_tools
            if text_value(tool, "type") == "shell"
        }
        for item in value(response, "output", []) or []:
            kind = text_value(item, "type")
            if kind == "shell_call" and shell_environments != {"local"}:
                # Hosted containers are billed by session resources/duration;
                # a call count alone cannot determine their charge. A local
                # shell declaration is executed by the caller and adds no fee.
                tool_usage[kind] = tool_usage.get(kind, 0) + 1
            if kind in ("web_search_call", "file_search_call", "code_interpreter_call", "image_generation_call"):
                if kind == "web_search_call" and len(search_types) == 1:
                    kind = "web_search_requests" if "web_search" in search_types else "web_search_preview_calls"
                tool_usage[kind] = tool_usage.get(kind, 0) + 1
    details.update(uncached_input_tokens=uncached, cached_input_tokens=cached, cache_creation_tokens=writes)
    if tool_usage:
        details["server_tool_usage"] = tool_usage
    cost_options.update(
        cached_tokens=cached,
        cache_creation_tokens=writes,
        service_tier=service_tier,
        inference_geo=geo,
        usage_complete=complete,
        tool_usage=tool_usage,
    )
    pricing = driver._calculate_cost_details(provider, effective_model, prompt, output, **cost_options)
    if provider == "claude" and details.get("speed") not in (None, "standard"):
        pricing["cost_status"] = "partial" if pricing["rates_available"] else "unknown"
        pricing["pricing"].setdefault("unpriced", []).append("speed:" + details["speed"])
    # Text token rates cannot fully price audio or generated images. Retain the
    # known subtotal and make the missing component visible.
    if provider == "openai" and any(details.get(k, 0) for k in ("input_audio_tokens", "output_audio_tokens")):
        pricing["cost_status"] = "partial" if pricing["rates_available"] else "unknown"
        pricing["pricing"].setdefault("unpriced", []).append("audio_tokens")
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": prompt + output,
        "cached_prompt_tokens": cached,
        "cache_creation_tokens": writes,
        "model_name": effective_model,
        "requested_model": model,
        "returned_model": returned_model,
        "request_id": text_value(response, "_request_id"),
        "response_id": text_value(response, "id"),
        "service_tier": service_tier,
        "usage_details": details,
        **pricing,
    }
    diagnostics = value(response, "prompt_cache_diagnostics")
    if diagnostics is not None:
        result["prompt_cache_diagnostics"] = as_dict(diagnostics)
    for key in ("retry_attempt", "fallback", "extraction_success"):
        if key in options:
            result[key] = options[key]
    return result
