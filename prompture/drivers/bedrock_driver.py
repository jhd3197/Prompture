"""AWS Bedrock LLM driver implementation.

Routes a single ``bedrock/<vendor>.<model-id>`` model string to the
appropriate vendor-specific request/response shape via Bedrock's
``InvokeModel`` API.

Supported vendors (dispatched on the first segment of the model ID):

* ``anthropic.*``    — Claude Messages format (vision + tool use supported)
* ``meta.llama*``    — Llama text-completion format
* ``mistral.*``      — Mistral text-completion format
* ``cohere.*``       — Command-R chat format
* ``amazon.titan*``  — Titan text generation format
* ``amazon.nova*``   — Routed through the Anthropic-style Messages shape
                       (Bedrock Nova uses the Converse API natively; for
                       InvokeModel we treat it as a Claude-shaped vendor
                       since both expose the ``messages`` field).

This driver uses the **InvokeModel** endpoint; the (newer) Converse API
is a future improvement — see :pyfile:`Deferred items` in the Phase 4
notes.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import Iterator
from typing import Any

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    _BOTO3_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in tests via patch
    boto3 = None  # type: ignore[assignment]
    BotoCoreError = Exception  # type: ignore[assignment,misc]
    ClientError = Exception  # type: ignore[assignment,misc]
    _BOTO3_AVAILABLE = False

from ..infra.cost_mixin import CostMixin
from ._prompt_cache import (
    apply_cache_control_to_messages,
    apply_cache_control_to_system,
    apply_cache_control_to_tools,
    breakpoint_budget,
)
from .base import Driver

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _vendor_for(model: str) -> str:
    """Return the canonical Bedrock vendor key for *model*.

    Examples
    --------
    >>> _vendor_for("anthropic.claude-3-5-sonnet-20240620-v1:0")
    'anthropic'
    >>> _vendor_for("meta.llama3-1-70b-instruct-v1:0")
    'meta'
    >>> _vendor_for("amazon.titan-text-express-v1")
    'amazon'
    """
    return (model or "").split(".", 1)[0].lower()


def _extract_system_and_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split the system message out of the conversation."""
    system_content: str | None = None
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
        else:
            api_messages.append(msg)
    return system_content, api_messages


def _flatten_messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten a chat into a single prompt for non-message vendors."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "".join(text_parts)
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _prepare_anthropic_messages_for_bedrock(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI-style messages (incl. vision) to Bedrock-Anthropic blocks."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    blocks.append({"type": "text", "text": str(block)})
                    continue
                btype = block.get("type")
                if btype == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif btype in ("image_url", "image"):
                    img = block.get("image_url") or block.get("image") or block.get("source")
                    if isinstance(img, dict) and img.get("url"):
                        url = img["url"]
                        if url.startswith("data:"):
                            # data:image/jpeg;base64,...
                            header, _, b64 = url.partition(",")
                            media_type = header.split(";")[0].split(":", 1)[-1] or "image/jpeg"
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64,
                                    },
                                }
                            )
                        else:
                            blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64.b64encode(url.encode()).decode(),
                                    },
                                }
                            )
                    elif isinstance(img, dict) and img.get("data"):
                        blocks.append({"type": "image", "source": img})
                else:
                    # Pass-through: assume already in Anthropic block form
                    blocks.append(block)
            out.append({"role": role, "content": blocks})
        else:
            out.append({"role": role, "content": content})
    return out


def _convert_tools_to_anthropic_bedrock(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anthropic_tools: list[dict[str, Any]] = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            anthropic_tools.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        elif "input_schema" in t:
            anthropic_tools.append(t)
        else:
            anthropic_tools.append(t)
    return anthropic_tools


# ----------------------------------------------------------------------
# Vendor request/response adapters
# ----------------------------------------------------------------------


def _build_anthropic_request(
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    system_content, api_messages = _extract_system_and_messages(messages)
    api_messages = _prepare_anthropic_messages_for_bedrock(api_messages)

    # Bedrock serves the same Anthropic Messages body, so the same
    # cache_control breakpoints apply — system, tools, and (crucially for
    # agentic loops) the message array itself.
    cache_kwargs = {
        "cache_prompt": options.get("cache_prompt", True),
        "model": options.get("model"),
        "ttl": options.get("cache_ttl", "5m"),
    }
    wrapped_system = apply_cache_control_to_system(system_content, **cache_kwargs)
    anthropic_tools = (
        apply_cache_control_to_tools(_convert_tools_to_anthropic_bedrock(tools), **cache_kwargs) if tools else None
    )
    api_messages = apply_cache_control_to_messages(
        api_messages,
        max_breakpoints=breakpoint_budget(wrapped_system, anthropic_tools),
        **cache_kwargs,
    )

    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": int(options.get("max_tokens", 512)),
        "messages": api_messages,
    }
    if wrapped_system is not None:
        body["system"] = wrapped_system
    if "temperature" in options:
        body["temperature"] = options["temperature"]
    if anthropic_tools is not None:
        body["tools"] = anthropic_tools
    return body


def _parse_anthropic_response(
    payload: dict[str, Any],
    *,
    response_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    content_blocks = payload.get("content", []) or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content_blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                }
            )
    usage = payload.get("usage", {}) or {}
    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls,
        "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
        "completion_tokens": int(usage.get("output_tokens", 0) or 0),
        "stop_reason": payload.get("stop_reason"),
    }


def _build_meta_request(messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
    prompt = _flatten_messages_to_prompt(messages)
    body: dict[str, Any] = {
        "prompt": prompt,
        "max_gen_len": int(options.get("max_tokens", 512)),
    }
    if "temperature" in options:
        body["temperature"] = options["temperature"]
    if "top_p" in options:
        body["top_p"] = options["top_p"]
    return body


def _parse_meta_response(
    payload: dict[str, Any],
    *,
    response_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "text": payload.get("generation", "") or "",
        "tool_calls": [],
        "prompt_tokens": int(payload.get("prompt_token_count", 0) or 0),
        "completion_tokens": int(payload.get("generation_token_count", 0) or 0),
        "stop_reason": payload.get("stop_reason"),
    }


def _build_mistral_request(messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
    system_content, api_messages = _extract_system_and_messages(messages)
    flat = _flatten_messages_to_prompt(api_messages)
    inst = flat if not system_content else f"{system_content}\n\n{flat}"
    body: dict[str, Any] = {
        "prompt": f"<s>[INST] {inst} [/INST]",
        "max_tokens": int(options.get("max_tokens", 512)),
    }
    if "temperature" in options:
        body["temperature"] = options["temperature"]
    if "top_p" in options:
        body["top_p"] = options["top_p"]
    return body


def _parse_mistral_response(
    payload: dict[str, Any],
    *,
    response_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    outputs = payload.get("outputs", []) or []
    text = ""
    stop_reason = None
    if outputs:
        text = outputs[0].get("text", "") or ""
        stop_reason = outputs[0].get("stop_reason")
    pt = 0
    ct = 0
    if response_headers:
        pt = int(response_headers.get("x-amzn-bedrock-input-token-count", 0) or 0)
        ct = int(response_headers.get("x-amzn-bedrock-output-token-count", 0) or 0)
    return {
        "text": text,
        "tool_calls": [],
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "stop_reason": stop_reason,
    }


def _build_cohere_request(messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
    # Cohere Command-R on Bedrock uses ``message`` + chat_history (single
    # latest user turn + prior turns). Keep it simple: take the last user
    # message as ``message`` and pre-history as chat_history.
    system_content, api_messages = _extract_system_and_messages(messages)
    last_user = ""
    prior: list[dict[str, Any]] = []
    for i, msg in enumerate(api_messages):
        if i == len(api_messages) - 1 and msg.get("role") == "user":
            last_user = msg.get("content", "")
        else:
            prior.append(
                {
                    "role": "USER" if msg.get("role") == "user" else "CHATBOT",
                    "message": msg.get("content", ""),
                }
            )
    body: dict[str, Any] = {
        "message": last_user or _flatten_messages_to_prompt(api_messages),
        "max_tokens": int(options.get("max_tokens", 512)),
    }
    if system_content:
        body["preamble"] = system_content
    if prior:
        body["chat_history"] = prior
    if "temperature" in options:
        body["temperature"] = options["temperature"]
    return body


def _parse_cohere_response(
    payload: dict[str, Any],
    *,
    response_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    text = payload.get("text", "") or ""
    # Cohere on Bedrock sometimes also returns ``generations``
    if not text and isinstance(payload.get("generations"), list) and payload["generations"]:
        text = payload["generations"][0].get("text", "") or ""
    pt = 0
    ct = 0
    if response_headers:
        pt = int(response_headers.get("x-amzn-bedrock-input-token-count", 0) or 0)
        ct = int(response_headers.get("x-amzn-bedrock-output-token-count", 0) or 0)
    return {
        "text": text,
        "tool_calls": [],
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "stop_reason": payload.get("finish_reason"),
    }


def _build_titan_request(messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
    prompt = _flatten_messages_to_prompt(messages)
    body: dict[str, Any] = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": int(options.get("max_tokens", 512)),
        },
    }
    if "temperature" in options:
        body["textGenerationConfig"]["temperature"] = options["temperature"]
    if "top_p" in options:
        body["textGenerationConfig"]["topP"] = options["top_p"]
    return body


def _parse_titan_response(
    payload: dict[str, Any],
    *,
    response_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    results = payload.get("results", []) or []
    text = ""
    stop_reason = None
    completion_tokens = 0
    if results:
        text = results[0].get("outputText", "") or ""
        stop_reason = results[0].get("completionReason")
        completion_tokens = int(results[0].get("tokenCount", 0) or 0)
    return {
        "text": text,
        "tool_calls": [],
        "prompt_tokens": int(payload.get("inputTextTokenCount", 0) or 0),
        "completion_tokens": completion_tokens,
        "stop_reason": stop_reason,
    }


# ----------------------------------------------------------------------
# BedrockDriver
# ----------------------------------------------------------------------


class BedrockDriver(CostMixin, Driver):
    """AWS Bedrock multi-vendor LLM driver.

    Authentication uses boto3's standard credential chain. Pass
    ``aws_access_key_id`` / ``aws_secret_access_key`` / ``aws_region``
    explicitly or rely on env vars / IAM role / shared credentials.
    """

    # Conservative defaults — vendor capability varies. Anthropic models
    # on Bedrock support tool use & vision; others do not.
    supports_json_mode = False
    supports_json_schema = False
    # ``True`` because we provide a ``generate_messages_with_tools`` impl —
    # however that path only succeeds for Anthropic models on Bedrock
    # InvokeModel; other vendors raise ``NotImplementedError`` at call time.
    supports_tool_use = True
    supports_streaming = True  # Anthropic only; raises NotImplementedError for others
    supports_vision = True  # Anthropic only
    supports_messages = True

    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_region: str | None = None,
        model: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
    ):
        if not _BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for the Bedrock driver. Install with: pip install prompture[bedrock]")

        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = aws_region or os.getenv("AWS_REGION") or "us-east-1"
        self.model = model

        client_kwargs: dict[str, Any] = {"region_name": self.aws_region}
        if self.aws_access_key_id and self.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = self.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        # SigV4 auth is handled internally by boto3 — nothing else needed.
        self._client = boto3.client("bedrock-runtime", **client_kwargs)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(list(messages), options)

    def _do_generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        vendor = _vendor_for(model)

        try:
            self._validate_model_capabilities("bedrock", model)
        except Exception:  # pragma: no cover - graceful degradation
            logger.debug("Capability validation failed for bedrock/%s", model, exc_info=True)

        opts = {"max_tokens": 512, **options}

        if vendor == "anthropic":
            body = _build_anthropic_request(messages, opts)
            parser = _parse_anthropic_response
        elif vendor == "meta":
            body = _build_meta_request(messages, opts)
            parser = _parse_meta_response
        elif vendor == "mistral":
            body = _build_mistral_request(messages, opts)
            parser = _parse_mistral_response
        elif vendor == "cohere":
            body = _build_cohere_request(messages, opts)
            parser = _parse_cohere_response
        elif vendor == "amazon":
            # Nova uses Anthropic-style messages; Titan uses its own shape.
            if model.lower().startswith("amazon.nova"):
                body = _build_anthropic_request(messages, opts)
                parser = _parse_anthropic_response
            else:
                body = _build_titan_request(messages, opts)
                parser = _parse_titan_response
        else:
            raise ValueError(
                f"Unsupported Bedrock vendor for model '{model}'. "
                "Supported prefixes: anthropic., meta.llama, mistral., cohere., amazon."
            )

        payload, headers = self._invoke(model, body)
        parsed = parser(payload, response_headers=headers)

        total_cost = self._calculate_cost(
            "bedrock",
            model,
            parsed["prompt_tokens"],
            parsed["completion_tokens"],
        )

        meta = {
            "prompt_tokens": parsed["prompt_tokens"],
            "completion_tokens": parsed["completion_tokens"],
            "total_tokens": parsed["prompt_tokens"] + parsed["completion_tokens"],
            "cost": round(total_cost, 6),
            "raw_response": payload,
            "model_name": model,
        }
        return {"text": parsed["text"], "meta": meta}

    # ------------------------------------------------------------------
    # Tool use (Anthropic only on Bedrock InvokeModel)
    # ------------------------------------------------------------------

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        model = options.get("model", self.model)
        vendor = _vendor_for(model)
        if vendor != "anthropic":
            raise NotImplementedError(
                f"Tool use on Bedrock is only supported for Anthropic models on the "
                f"InvokeModel API. Got vendor '{vendor}' for model '{model}'. "
                "Consider the Bedrock Converse API for broader vendor support."
            )

        try:
            self._validate_model_capabilities("bedrock", model, using_tool_use=True)
        except Exception:  # pragma: no cover
            logger.debug("Capability validation failed", exc_info=True)

        opts = {"max_tokens": 4096, **options}
        body = _build_anthropic_request(messages, opts, tools=tools)
        payload, _headers = self._invoke(model, body)
        parsed = _parse_anthropic_response(payload)

        total_cost = self._calculate_cost(
            "bedrock",
            model,
            parsed["prompt_tokens"],
            parsed["completion_tokens"],
        )
        meta = {
            "prompt_tokens": parsed["prompt_tokens"],
            "completion_tokens": parsed["completion_tokens"],
            "total_tokens": parsed["prompt_tokens"] + parsed["completion_tokens"],
            "cost": round(total_cost, 6),
            "raw_response": payload,
            "model_name": model,
        }
        return {
            "text": parsed["text"],
            "meta": meta,
            "tool_calls": parsed["tool_calls"],
            "stop_reason": parsed["stop_reason"],
        }

    # ------------------------------------------------------------------
    # Streaming (Anthropic only)
    # ------------------------------------------------------------------

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        model = options.get("model", self.model)
        vendor = _vendor_for(model)
        if vendor != "anthropic" and not model.lower().startswith("amazon.nova"):
            raise NotImplementedError(
                "Streaming on Bedrock is currently implemented only for Anthropic / Nova "
                "models in this driver. Use a non-streaming generate() call for other vendors."
            )

        opts = {"max_tokens": 512, **options}
        body = _build_anthropic_request(messages, opts)

        try:
            response = self._client.invoke_model_with_response_stream(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except (ClientError, BotoCoreError) as e:
            raise RuntimeError(f"Bedrock streaming request failed: {e!s}\nBody: {body!r}") from e

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        for event in response.get("body", []):
            chunk = event.get("chunk") if isinstance(event, dict) else None
            if not chunk:
                continue
            try:
                data = json.loads(chunk["bytes"])
            except (KeyError, json.JSONDecodeError):
                continue
            ev_type = data.get("type")
            if ev_type == "content_block_delta":
                delta = data.get("delta", {}) or {}
                text = delta.get("text", "")
                if text:
                    full_text += text
                    yield {"type": "delta", "text": text}
            elif ev_type == "message_start":
                usage = (data.get("message") or {}).get("usage", {}) or {}
                prompt_tokens = int(usage.get("input_tokens", 0) or 0)
            elif ev_type == "message_delta":
                usage = data.get("usage", {}) or {}
                completion_tokens = int(usage.get("output_tokens", completion_tokens) or completion_tokens)

        total_cost = self._calculate_cost("bedrock", model, prompt_tokens, completion_tokens)
        yield {
            "type": "done",
            "text": full_text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": round(total_cost, 6),
                "raw_response": {},
                "model_name": model,
            },
        }

    # ------------------------------------------------------------------
    # Internal: InvokeModel
    # ------------------------------------------------------------------

    def _invoke(self, model: str, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """Call Bedrock InvokeModel and return ``(payload_dict, http_headers)``.

        Errors surface the request body to make debugging easier (mirrors
        the openrouter driver pattern).
        """
        try:
            response = self._client.invoke_model(
                modelId=model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except (ClientError, BotoCoreError) as e:
            raise RuntimeError(f"Bedrock API request failed for model '{model}': {e!s}\nRequest body: {body!r}") from e

        try:
            raw = response["body"].read()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to read Bedrock response body: {e!s}") from e

        try:
            payload = json.loads(raw) if isinstance(raw, (bytes, bytearray, str)) else raw
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bedrock returned non-JSON response: {raw!r}") from e

        headers: dict[str, str] = {}
        meta = response.get("ResponseMetadata") or {}
        http_headers = meta.get("HTTPHeaders") or {}
        if isinstance(http_headers, dict):
            headers = {str(k).lower(): str(v) for k, v in http_headers.items()}
        return payload, headers
