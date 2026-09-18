"""Explicit provider preflight counts. Importing this module performs no I/O.

OpenAI counts the Responses representation, even for a Chat Completions
configured driver. Counts are provider estimates, not billed generation usage;
output length and future cache hits must still be forecast by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .budget import CostEstimate, estimate_call_cost


@dataclass(frozen=True)
class TokenCount:
    """Provider estimate of request input size; never a generation event."""

    input_tokens: int
    model: str
    provider: str
    token_counter: str = "provider"
    request_id: str | None = None


def _provider(model_name: str) -> tuple[str, str]:
    provider, _, model = model_name.partition("/")
    if not model or provider not in {"openai", "claude", "anthropic"}:
        raise ValueError("Provider token counting requires openai/model or claude/model")
    return provider, model


def _request(
    provider: str,
    model: str,
    driver: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    import copy

    messages = copy.deepcopy(messages)
    options = {"model": model, **copy.deepcopy(options)}
    if provider == "openai":
        from ..drivers._openai_responses import build_request
        from ..drivers.vision_helpers import _prepare_openai_vision_messages

        request = build_request(driver, _prepare_openai_vision_messages(messages), options, tools, counting=True)
        for key in ("parallel_tool_calls", "conversation", "timeout"):
            if key in options:
                request[key] = options[key]
        if "tool_choice" in options:
            choice = options["tool_choice"]
            if isinstance(choice, dict) and "function" in choice:
                choice = {"type": "function", "name": choice["function"]["name"]}
            elif isinstance(choice, dict) and "name" in choice:
                choice = {"type": "function", **choice}
            request["tool_choice"] = choice
        return request
    from ..drivers.claude_driver import (
        _build_anthropic_json_mode_tool_def,
        _convert_tools_to_anthropic,
        _extract_anthropic_system_and_messages,
    )
    from ..drivers.vision_helpers import _prepare_claude_vision_messages

    # Native Anthropic image blocks are already prepared. Universal ImageContent
    # blocks need conversion; preserve native blocks before running the helper.
    prepared = []
    for message in messages:
        if isinstance(message.get("content"), list):
            blocks = []
            for block in message["content"]:
                if block.get("type") == "image" and not isinstance(block.get("source"), dict):
                    blocks.extend(_prepare_claude_vision_messages([{"role": "user", "content": [block]}])[0]["content"])
                else:
                    blocks.append(block)
            message["content"] = blocks
        prepared.append(message)
    system, api_messages = _extract_anthropic_system_and_messages(prepared)
    request = {"model": options["model"], "messages": api_messages}
    if system is not None:
        request["system"] = system
    if tools:
        request["tools"] = _convert_tools_to_anthropic(copy.deepcopy(tools))
    elif options.get("json_mode") and options.get("json_schema"):
        request["tools"] = [_build_anthropic_json_mode_tool_def(options["json_schema"])]
        request["tool_choice"] = {"type": "tool", "name": "extract_json"}
    if tools and "tool_choice" in options:
        from ..drivers.base import _translate_tool_choice

        choice = _translate_tool_choice(options["tool_choice"], "anthropic")
        if choice is not None:
            request["tool_choice"] = choice
    for key in ("thinking", "timeout"):
        if key in options:
            request[key] = options[key]
    return request


def _endpoint(client: Any, provider: str) -> Any:
    try:
        endpoint = client.responses.input_tokens.count if provider == "openai" else client.messages.count_tokens
    except AttributeError as exc:
        raise RuntimeError(f"{provider} SDK lacks the token counting endpoint; upgrade the provider SDK") from exc
    if not callable(endpoint):
        raise RuntimeError(f"{provider} SDK lacks the token counting endpoint; upgrade the provider SDK")
    return endpoint


def _result(response: Any, provider: str, model: str) -> TokenCount:
    count = response.get("input_tokens") if isinstance(response, dict) else getattr(response, "input_tokens", None)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("Provider returned an invalid input token count")
    request_id = response.get("_request_id") if isinstance(response, dict) else getattr(response, "_request_id", None)
    return TokenCount(count, f"{provider}/{model}", provider, request_id=request_id)


def count_request_tokens(
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
    driver: Any = None,
) -> TokenCount:
    """Make one explicit provider counting request, without generating output."""
    provider, model = _provider(model_name)
    owned_driver = driver is None
    if driver is None:
        from ..drivers import get_driver_for_model

        driver = get_driver_for_model(f"{'claude' if provider == 'anthropic' else provider}/{model}")
    client = getattr(driver, "client", None)
    owned_client = owned_driver
    if client is None and provider in {"claude", "anthropic"}:
        import anthropic

        client = anthropic.Anthropic(api_key=driver.api_key)
        owned_client = True
    try:
        request = _request(provider, model, driver, messages, tools, options or {})
        response = _endpoint(client, provider)(**request)
        return _result(response, provider, request["model"])
    finally:
        close = getattr(client, "close", None)
        if owned_client and callable(close):
            close()


async def acount_request_tokens(
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
    driver: Any = None,
) -> TokenCount:
    """Async explicit provider count; supplied drivers must have async clients."""
    provider, model = _provider(model_name)
    owned_driver = driver is None
    if driver is None:
        from ..drivers.async_registry import get_async_driver_for_model

        driver = get_async_driver_for_model(f"{'claude' if provider == 'anthropic' else provider}/{model}")
    client = getattr(driver, "client", None)
    try:
        request = _request(provider, model, driver, messages, tools, options or {})
        response = await _endpoint(client, provider)(**request)
        return _result(response, provider, request["model"])
    finally:
        close = getattr(client, "close", None)
        if owned_driver and callable(close):
            await close()


def _estimate(
    count: TokenCount, expected_completion_tokens: int, options: dict[str, Any] | None, pricing_context: dict[str, Any]
) -> CostEstimate:
    context = {key: options[key] for key in ("service_tier", "inference_geo") if options and key in options}
    context.update(pricing_context)
    return replace(
        estimate_call_cost(
            count.model, count.input_tokens, expected_completion_tokens=expected_completion_tokens, **context
        ),
        token_counter="provider",
    )


def estimate_request_cost(
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
    driver: Any = None,
    expected_completion_tokens: int = 500,
    **pricing_context: Any,
) -> CostEstimate:
    """Count inputs with the provider and forecast output/cache/tool costs."""
    count = count_request_tokens(model_name, messages, tools=tools, options=options, driver=driver)
    return _estimate(count, expected_completion_tokens, options, pricing_context)


async def aestimate_request_cost(
    model_name: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
    driver: Any = None,
    expected_completion_tokens: int = 500,
    **pricing_context: Any,
) -> CostEstimate:
    """Async provider input count and output/cache/tool cost forecast."""
    count = await acount_request_tokens(model_name, messages, tools=tools, options=options, driver=driver)
    return _estimate(count, expected_completion_tokens, options, pricing_context)
