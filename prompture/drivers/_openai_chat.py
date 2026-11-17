"""One implementation of tool calling for every OpenAI-wire provider.

A dozen providers speak the OpenAI ``chat/completions`` format with small
differences: one rejects ``tool_choice="required"``, one wants ``"any"``
instead, one needs every tool to carry a description, one clamps temperature.
Those differences used to live as hand-edited copies of the same eighty lines
in each driver, sync and async, buffered and streamed, and every fix had to be
repeated in each copy.

Here the request is built once (:func:`build_chat_request`) and the reply is
parsed once (:func:`parse_chat_response`). What differs between providers is
data, a :class:`ChatProfile`. A driver supplies its profile and inherits the
tool methods from :class:`OpenAIChatTools` / :class:`AsyncOpenAIChatTools`, so
the buffered and streamed paths, sync and async, cannot drift apart.

The order of the pipeline is fixed and no step is the driver's to remember:

1. convert attached images (``driver._prepare_messages``)
2. drop Prompture-private keys from the history
3. flatten and sanitize tool schemas
4. translate ``tool_choice`` for this provider
5. send
6. parse arguments (reporting malformed ones), fill missing ids, normalize the
   stop reason, usage and cost
"""

from __future__ import annotations

import contextlib
import copy
import logging
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from ..infra.rate_limits import add_rate_limits, limits_from_response
from ..infra.tool_schema import flatten_tool_definitions
from .base import DriverHTTPError, _normalize_stop_reason, _tool_call_dict, _translate_tool_choice

logger = logging.getLogger(__name__)


class ChatRequestError(DriverHTTPError, RuntimeError):
    """A chat-completions request failed.

    Still a ``RuntimeError``, which is what these drivers have always raised,
    and now also a :class:`DriverHTTPError` carrying ``status_code`` so the
    resilience layer can tell a rate limit from a bad request.
    """


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatProfile:
    """How one provider's chat-completions endpoint differs from OpenAI's."""

    #: Pricing / capability key (``"zai"``, ``"grok"`` …).
    provider: str
    #: Name used in error messages.
    label: str = ""
    default_temperature: float = 1.0
    default_max_tokens: int = 4096
    #: ``"openai"`` (``reasoning_effort``), ``"openrouter"`` (``reasoning.effort``)
    #: or ``None`` when the provider takes no effort setting.
    effort_style: str | None = None
    #: ``False`` drops ``tool_choice`` entirely.
    tool_choice: bool = True
    #: What a normalized ``tool_choice="required"`` is called here. Mistral
    #: says ``"any"``; Moonshot has no equivalent and falls back to ``"auto"``.
    required_tool_choice: str = "required"
    #: ``False`` when the provider cannot force one named tool; the choice is
    #: then downgraded to :attr:`required_tool_choice`.
    named_tool_choice: bool = True
    #: Forward ``parallel_tool_calls`` when the caller sets it.
    parallel_tool_calls: bool = True
    #: Clamp temperature into this range before sending.
    temperature_range: tuple[float, float] | None = None
    #: Give description-less tools a description and drop empty ``required``
    #: arrays, for providers that answer 400 otherwise.
    strict_tool_shape: bool = False
    #: The provider accepts ``"strict": true`` on a function tool.
    strict_tools: bool = False
    #: Reasoning models answer in ``reasoning_content`` unless thinking is
    #: switched off for the tool turn.
    disable_thinking_for_tools: bool = False
    #: Path appended to the driver's base URL.
    path: str = "/chat/completions"
    #: Seconds; ``options["timeout"]`` overrides.
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not self.label:
            object.__setattr__(self, "label", self.provider)


_PROFILES: dict[str, ChatProfile] = {}


def register_chat_profile(profile: ChatProfile) -> ChatProfile:
    """Register (or replace) the profile for ``profile.provider``."""
    _PROFILES[profile.provider] = profile
    return profile


def get_chat_profile(provider: str) -> ChatProfile:
    """The registered profile for *provider*, or plain OpenAI behaviour."""
    return _PROFILES.get(provider) or ChatProfile(provider=provider)


for _profile in (
    ChatProfile("openai", "OpenAI", effort_style="openai", strict_tools=True),
    ChatProfile("azure", "Azure OpenAI", effort_style="openai", strict_tools=True),
    ChatProfile("groq", "Groq", default_temperature=0.7, effort_style="openai"),
    ChatProfile("grok", "Grok", effort_style="openai"),
    ChatProfile("openrouter", "OpenRouter", effort_style="openrouter"),
    ChatProfile("mistral", "Mistral", default_temperature=0.7, required_tool_choice="any"),
    ChatProfile(
        "moonshot",
        "Moonshot",
        required_tool_choice="auto",
        temperature_range=(0.0, 1.0),
        strict_tool_shape=True,
        disable_thinking_for_tools=True,
        timeout=300.0,
    ),
    ChatProfile("zai", "Z.ai"),
    ChatProfile("modelscope", "ModelScope"),
    ChatProfile("cachibot", "CachiBot"),
    ChatProfile("minimax", "MiniMax", path="/text/chatcompletion_v2", timeout=300.0),
    ChatProfile("openai_compatible", "OpenAI-compatible", default_temperature=0.7, effort_style="openai"),
):
    register_chat_profile(_profile)
del _profile


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatRoute:
    """Where one request goes and how it is priced."""

    #: The model name on the wire (a deployment id, a gateway-prefixed id …).
    api_model: str
    #: Provider and model the cost and capability tables are keyed by.
    cost_provider: str
    cost_model: str
    #: Whatever the driver's transport needs to send it (endpoint config …).
    context: Any = None


@dataclass
class ChatRequest:
    """A request ready to send, plus what parsing the reply will need."""

    payload: dict[str, Any]
    #: The model as the caller named it (reported in ``meta["model_name"]``).
    model: str
    route: ChatRoute
    timeout: float
    #: The caller's options, for transports that take per-call overrides.
    options: dict[str, Any]

    @property
    def cost_provider(self) -> str:
        return self.route.cost_provider

    @property
    def cost_model(self) -> str:
        return self.route.cost_model


def wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop Prompture-private keys (``_google_parts`` and friends) from history.

    Conversations keep provider state on assistant messages under
    underscore-prefixed keys so a later turn on the same provider can replay
    it. On any other provider those keys are noise at best and a 400 at worst.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict) and any(isinstance(k, str) and k.startswith("_") for k in msg):
            msg = {k: v for k, v in msg.items() if not (isinstance(k, str) and k.startswith("_"))}
        out.append(msg)
    return out


def _tighten_tool_shape(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            out.append(tool)
            continue
        fn = dict(fn)
        if not fn.get("description"):
            fn["description"] = f"Call the {fn.get('name', 'unknown')} function"
        params = dict(fn.get("parameters") or {})
        if "required" in params and not params["required"]:
            del params["required"]
        params.setdefault("properties", {})
        fn["parameters"] = params
        out.append({**tool, "function": fn})
    return out


def _strict_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from ..infra.cost_mixin import prepare_strict_schema

    out = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict) or fn.get("strict") is not None:
            out.append(tool)
            continue
        params = prepare_strict_schema(copy.deepcopy(fn.get("parameters") or {"type": "object", "properties": {}}))
        out.append({**tool, "function": {**fn, "parameters": params, "strict": True}})
    return out


def prepare_tools(tools: list[dict[str, Any]], profile: ChatProfile, options: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool definitions as this provider accepts them.

    Runs on every request, whoever built the tools: a ``ToolDefinition``, the
    gateway relaying a client's raw schema, or a caller's hand-written dict.
    """
    tools = flatten_tool_definitions(list(tools or []))
    if profile.strict_tool_shape:
        tools = _tighten_tool_shape(tools)
    if options.get("strict_tools") and profile.strict_tools:
        tools = _strict_tools(tools)
    return tools


def provider_tool_choice(tool_choice: Any, profile: ChatProfile) -> Any:
    """Normalized ``tool_choice`` → this provider's wire value, or ``None``."""
    if tool_choice is None or not profile.tool_choice:
        return None
    wire = _translate_tool_choice(tool_choice, "openai")
    if wire == "required":
        return profile.required_tool_choice
    if isinstance(wire, dict) and not profile.named_tool_choice:
        return profile.required_tool_choice
    return wire


def build_chat_request(
    driver: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: dict[str, Any],
    profile: ChatProfile,
    *,
    stream: bool = False,
) -> ChatRequest:
    """Build the one tool-call request every path sends.

    The buffered and streamed paths, sync and async, all come through here;
    ``stream`` only adds the streaming flags.
    """
    from ._effort import apply_effort
    from ._openai_compat import merge_extra_body

    # Images first: nothing else may run on messages a transport cannot send.
    wire = wire_messages(driver._prepare_messages(messages))

    model = options.get("model", driver.model)
    route: ChatRoute = driver._chat_route(model, options)
    api_model, cost_provider, cost_model = route.api_model, route.cost_provider, route.cost_model

    config = driver._get_model_config(cost_provider, cost_model)
    with contextlib.suppress(Exception):
        driver._validate_model_capabilities(cost_provider, cost_model, using_tool_use=True)

    opts = {"temperature": profile.default_temperature, "max_tokens": profile.default_max_tokens, **options}

    payload: dict[str, Any] = {
        "model": api_model,
        "messages": wire,
        "tools": prepare_tools(tools, profile, options),
    }
    payload[config["tokens_param"]] = opts.get("max_tokens") or profile.default_max_tokens
    if profile.effort_style:
        apply_effort(payload, opts, profile.effort_style)

    if config["supports_temperature"] and opts.get("temperature") is not None:
        temperature = opts["temperature"]
        if profile.temperature_range is not None:
            low, high = profile.temperature_range
            temperature = max(low, min(high, float(temperature)))
        payload["temperature"] = temperature

    choice = provider_tool_choice(options.get("tool_choice"), profile)
    if choice is not None:
        payload["tool_choice"] = choice
    if profile.parallel_tool_calls and "parallel_tool_calls" in options:
        payload["parallel_tool_calls"] = options["parallel_tool_calls"]

    if profile.disable_thinking_for_tools:
        from ..infra.model_rates import get_model_capabilities

        caps = get_model_capabilities(cost_provider, cost_model)
        if caps is not None and caps.is_reasoning is True:
            payload["thinking"] = options.get("thinking", {"type": "disabled"})

    driver._chat_finalize(payload, options)
    if getattr(driver, "_chat_uses_sdk", False):
        # An SDK rejects unknown keyword arguments and has its own escape hatch.
        if isinstance(options.get("extra_body"), dict):
            payload["extra_body"] = options["extra_body"]
        if options.get("timeout") is not None:
            payload["timeout"] = options["timeout"]
    else:
        merge_extra_body(payload, options)

    if stream:
        payload["stream"] = True
        payload["stream_options"] = {**payload.get("stream_options", {}), "include_usage": True}

    return ChatRequest(
        payload=payload,
        model=model,
        route=route,
        timeout=float(opts.get("timeout") or profile.timeout),
        options=options,
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def _as_dict(resp: Any) -> dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    for name in ("model_dump", "to_dict"):
        method = getattr(resp, name, None)
        if callable(method):
            dumped = method()
            if isinstance(dumped, dict):
                return dumped
    raise TypeError(f"Cannot read a chat completion from {type(resp).__name__}")


def cached_prompt_tokens(usage: dict[str, Any]) -> int:
    """Cached input tokens, wherever this provider reports them."""
    direct = usage.get("cached_tokens")
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return int(direct)
    details = usage.get("prompt_tokens_details")
    nested = details.get("cached_tokens") if isinstance(details, dict) else None
    if isinstance(nested, (int, float)) and not isinstance(nested, bool):
        return int(nested)
    return 0


def parse_chat_response(driver: Any, resp: Any, request: ChatRequest) -> dict[str, Any]:
    """The reply to *request* → the normalized tool-call result."""
    return parse_chat_completion(
        driver, resp, model=request.model, cost_provider=request.cost_provider, cost_model=request.cost_model
    )


def parse_chat_completion(
    driver: Any, resp: Any, *, model: str, cost_provider: str, cost_model: str | None = None
) -> dict[str, Any]:
    """A chat completion → the normalized tool-call result.

    Accepts the raw JSON dict or an SDK object. The result always has
    ``text``, ``meta``, ``tool_calls`` and a ``stop_reason`` from the shared
    vocabulary; the provider's own finish reason is kept in
    ``meta["raw_stop_reason"]``.
    """
    data = _as_dict(resp)
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    cached = cached_prompt_tokens(usage)
    cost = driver._calculate_cost(
        cost_provider, cost_model or model, prompt_tokens, completion_tokens, cached_tokens=cached
    )

    choice = (data.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    raw_stop_reason = choice.get("finish_reason")

    tool_calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        tool_calls.append(_tool_call_dict(call.get("id"), fn.get("name") or "", fn.get("arguments"), raw_stop_reason))

    meta: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": cached,
        "cost": round(cost, 6),
        "raw_response": data,
        "model_name": model,
        "raw_stop_reason": raw_stop_reason,
    }
    result: dict[str, Any] = {
        "text": message.get("content") or "",
        "meta": meta,
        "tool_calls": tool_calls,
        "stop_reason": _normalize_stop_reason(raw_stop_reason, tool_calls_present=bool(tool_calls)),
    }
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    if isinstance(reasoning, str):
        result["reasoning_content"] = reasoning
    return result


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _error(profile: ChatProfile, exc: Exception, *, status: int | None, body: str) -> ChatRequestError:
    message = f"{profile.label} API request failed: {exc!s}"
    if body:
        message += f"\nResponse: {body[:2000]}"
    return ChatRequestError(message, status_code=status, provider=profile.provider)


def require_api_key(driver: Any, profile: ChatProfile) -> None:
    """Fail before the request, with a clear message, when the key is missing."""
    if hasattr(driver, "api_key") and not driver.api_key:
        raise ChatRequestError(
            f"{profile.label} API key is not set", status_code=401, provider=profile.provider, retryable=False
        )


def post_chat(driver: Any, request: ChatRequest, profile: ChatProfile) -> tuple[dict[str, Any], Any]:
    """POST the request with ``requests``; returns ``(json, rate_limits)``."""
    import requests

    from ._openai_compat_stream import _compat_headers, _compat_url

    require_api_key(driver, profile)
    try:
        response = requests.post(
            _compat_url(driver, profile.path),
            headers=_compat_headers(driver),
            json=request.payload,
            timeout=request.timeout,
        )
        response.raise_for_status()
        return response.json(), limits_from_response(response)
    except requests.exceptions.HTTPError as exc:
        status = body = None
        if exc.response is not None:
            status = exc.response.status_code
            with contextlib.suppress(Exception):
                body = exc.response.text
        raise _error(profile, exc, status=status, body=body or "") from exc
    except requests.exceptions.RequestException as exc:
        raise _error(profile, exc, status=None, body="") from exc


async def apost_chat(driver: Any, request: ChatRequest, profile: ChatProfile) -> tuple[dict[str, Any], Any]:
    """Async :func:`post_chat` over ``httpx``."""
    import httpx

    from ._openai_compat_stream import _compat_headers, _compat_url

    require_api_key(driver, profile)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                _compat_url(driver, profile.path),
                headers=_compat_headers(driver),
                json=request.payload,
                timeout=request.timeout,
            )
            response.raise_for_status()
            return response.json(), limits_from_response(response)
    except httpx.HTTPStatusError as exc:
        body = ""
        with contextlib.suppress(Exception):
            body = exc.response.text
        raise _error(profile, exc, status=exc.response.status_code, body=body) from exc
    except httpx.HTTPError as exc:
        raise _error(profile, exc, status=None, body="") from exc


# ---------------------------------------------------------------------------
# Driver mixins
# ---------------------------------------------------------------------------


class _ChatToolsBase:
    """What both mixins share: the profile and the two per-driver hooks."""

    #: Set by each driver.
    chat_profile: ClassVar[ChatProfile]

    def _chat_route(self, model: str, options: dict[str, Any]) -> ChatRoute:
        """Where a request for *model* goes and how it is priced.

        Override when the name on the wire differs from the one the caller
        used (a deployment id, a gateway prefix).
        """
        return ChatRoute(model, self.chat_profile.provider, model)

    def _chat_finalize(self, payload: dict[str, Any], options: dict[str, Any]) -> None:
        """Last word on the payload, for the rare field no profile flag covers."""

    def _chat_meta(self, meta: dict[str, Any], request: ChatRequest) -> None:
        """Add driver-specific keys to ``meta`` (deployment id, endpoint …)."""


class OpenAIChatTools(_ChatToolsBase):
    """Buffered tool calling for a raw-HTTP OpenAI-wire driver."""

    def _chat_send(self, request: ChatRequest) -> tuple[Any, Any]:
        """Send *request*; return ``(reply, rate_limits_or_None)``.

        The default POSTs with ``requests``. SDK drivers override this.
        """
        return post_chat(self, request, self.chat_profile)

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls."""
        request = build_chat_request(self, messages, tools, options, self.chat_profile)
        reply, limits = self._chat_send(request)
        result = parse_chat_response(self, reply, request)
        add_rate_limits(result["meta"], limits)
        self._chat_meta(result["meta"], request)
        return result


class OpenAIChatToolStream(_ChatToolsBase):
    """Streamed tool calling for a raw-HTTP OpenAI-wire driver.

    Sends the request :func:`build_chat_request` builds, so a streamed turn is
    shaped exactly like a buffered one.
    """

    def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[Any]:
        """Stream one turn as :class:`~prompture.agents.live_events.LiveEvent`."""
        from ._openai_compat_stream import _compat_url, stream_raw_http_compat_tool_call

        profile = self.chat_profile
        request = build_chat_request(self, messages, tools, options, profile, stream=True)
        require_api_key(self, profile)
        for event in stream_raw_http_compat_tool_call(
            self,
            messages,
            tools,
            options,
            provider=request.cost_provider,
            url=_compat_url(self, profile.path),
            payload=request.payload,
            timeout=request.timeout,
            cost_model=request.cost_model,
            model_name=request.model,
        ):
            if event.event_type == "message_stop":
                self._chat_meta(event.usage, request)
            yield event


class AsyncOpenAIChatTools(_ChatToolsBase):
    """Async :class:`OpenAIChatTools`."""

    async def _chat_send(self, request: ChatRequest) -> tuple[Any, Any]:
        return await apost_chat(self, request, self.chat_profile)

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls."""
        request = build_chat_request(self, messages, tools, options, self.chat_profile)
        reply, limits = await self._chat_send(request)
        result = parse_chat_response(self, reply, request)
        add_rate_limits(result["meta"], limits)
        self._chat_meta(result["meta"], request)
        return result


class AsyncOpenAIChatToolStream(_ChatToolsBase):
    """Async :class:`OpenAIChatToolStream`."""

    async def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Stream one turn as :class:`~prompture.agents.live_events.LiveEvent`."""
        from ._openai_compat_stream import _compat_url, astream_raw_http_compat_tool_call

        profile = self.chat_profile
        request = build_chat_request(self, messages, tools, options, profile, stream=True)
        require_api_key(self, profile)
        async for event in astream_raw_http_compat_tool_call(
            self,
            messages,
            tools,
            options,
            provider=request.cost_provider,
            url=_compat_url(self, profile.path),
            payload=request.payload,
            timeout=request.timeout,
            cost_model=request.cost_model,
            model_name=request.model,
        ):
            if event.event_type == "message_stop":
                self._chat_meta(event.usage, request)
            yield event


class _SDKChat(_ChatToolsBase):
    """Shared by the SDK mixins: the client lives on the driver."""

    _chat_uses_sdk: ClassVar[bool] = True

    def _chat_client(self, request: ChatRequest) -> Any:
        """The OpenAI-style SDK client to send *request* with."""
        client = getattr(self, "client", None)
        if client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(f"{self.chat_profile.label} SDK client is not available; install its package")
        return client


class OpenAISDKChatTools(_SDKChat, OpenAIChatTools):
    """Buffered tool calling through an OpenAI-style SDK client."""

    def _chat_send(self, request: ChatRequest) -> tuple[Any, Any]:
        from ..infra.rate_limits import capture_rate_limits

        client = self._chat_client(request)
        with capture_rate_limits() as limits:
            reply = client.chat.completions.create(**request.payload)
        return reply, limits.snapshot


class OpenAISDKChatToolStream(_SDKChat):
    """Streamed tool calling through an OpenAI-style SDK client."""

    def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[Any]:
        """Stream one turn as :class:`~prompture.agents.live_events.LiveEvent`."""
        from ._openai_compat_stream import iter_openai_compat_live_events

        request = build_chat_request(self, messages, tools, options, self.chat_profile, stream=True)
        stream = self._chat_client(request).chat.completions.create(**request.payload)
        for event in iter_openai_compat_live_events(stream, model=request.model, cost_fn=_cost_fn(self, request)):
            if event.event_type == "message_stop":
                self._chat_meta(event.usage, request)
            yield event


class AsyncOpenAISDKChatTools(_SDKChat, AsyncOpenAIChatTools):
    """Async :class:`OpenAISDKChatTools`."""

    async def _chat_send(self, request: ChatRequest) -> tuple[Any, Any]:
        from ..infra.rate_limits import capture_rate_limits

        client = self._chat_client(request)
        with capture_rate_limits() as limits:
            reply = await client.chat.completions.create(**request.payload)
        return reply, limits.snapshot


class AsyncOpenAISDKChatToolStream(_SDKChat):
    """Async :class:`OpenAISDKChatToolStream`."""

    async def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Stream one turn as :class:`~prompture.agents.live_events.LiveEvent`."""
        from ._openai_compat_stream import aiter_openai_compat_live_events

        request = build_chat_request(self, messages, tools, options, self.chat_profile, stream=True)
        stream = await self._chat_client(request).chat.completions.create(**request.payload)
        async for event in aiter_openai_compat_live_events(
            stream, model=request.model, cost_fn=_cost_fn(self, request)
        ):
            if event.event_type == "message_stop":
                self._chat_meta(event.usage, request)
            yield event


def _cost_fn(driver: Any, request: ChatRequest) -> Any:
    def cost(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> float:
        return driver._calculate_cost(
            request.cost_provider, request.cost_model, prompt_tokens, completion_tokens, cached_tokens=cached_tokens
        )

    return cost


__all__ = [
    "AsyncOpenAIChatToolStream",
    "AsyncOpenAIChatTools",
    "AsyncOpenAISDKChatToolStream",
    "AsyncOpenAISDKChatTools",
    "ChatProfile",
    "ChatRequest",
    "ChatRequestError",
    "ChatRoute",
    "OpenAIChatToolStream",
    "OpenAIChatTools",
    "OpenAISDKChatToolStream",
    "OpenAISDKChatTools",
    "apost_chat",
    "build_chat_request",
    "cached_prompt_tokens",
    "get_chat_profile",
    "parse_chat_completion",
    "parse_chat_response",
    "post_chat",
    "prepare_tools",
    "provider_tool_choice",
    "register_chat_profile",
    "wire_messages",
]
