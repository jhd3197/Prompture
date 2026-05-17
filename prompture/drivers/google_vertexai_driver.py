"""Google Vertex AI driver — dual-backend.

* **Gemini models** → ``google-genai`` SDK with ``vertexai=True``
* **Claude models** → ``anthropic[vertex]`` SDK (``AnthropicVertex``)

The model name determines which backend is used automatically.
Auth: Vertex AI Express API key (``GOOGLE_VERTEX_API_KEY``) for Gemini,
project + ADC for Claude (requires ``GOOGLE_VERTEX_PROJECT_ID``).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from typing import Any

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

try:
    from anthropic import AnthropicVertex
except Exception:
    AnthropicVertex = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from .base import Driver

logger = logging.getLogger(__name__)


def _is_claude_model(model: str) -> bool:
    return model.startswith("claude")


class GoogleVertexAIDriver(CostMixin, Driver):
    """Driver for Google Vertex AI (Gemini + Claude via Model Garden)."""

    supports_json_mode = True
    supports_json_schema = True
    supports_vision = True
    supports_tool_use = True
    supports_streaming = True
    supports_messages = True

    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    _PROVIDER = "google_vertexai"

    # ── Client construction ────────────────────────────────────��──────

    @classmethod
    def _build_gemini_client(
        cls,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
    ) -> Any:
        """Build a google-genai Vertex AI client for Gemini models."""
        key = api_key or os.getenv("GOOGLE_VERTEX_API_KEY")
        proj = project_id or os.getenv("GOOGLE_VERTEX_PROJECT_ID")
        loc = location or os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")

        kwargs: dict[str, Any] = {"vertexai": True}
        if key:
            # API key and project/location are mutually exclusive in google-genai
            kwargs["api_key"] = key
        elif proj:
            kwargs["project"] = proj
            kwargs["location"] = loc
        return genai.Client(**kwargs)

    @classmethod
    def _build_claude_client(
        cls,
        project_id: str | None = None,
        location: str | None = None,
        access_token: str | None = None,
    ) -> Any:
        """Build an AnthropicVertex client for Claude models.

        Auth precedence: explicit ``access_token`` arg → ``GOOGLE_VERTEX_ACCESS_TOKEN``
        env var → Application Default Credentials (ADC) discovered by google-auth.
        """
        proj = project_id or os.getenv("GOOGLE_VERTEX_PROJECT_ID")
        loc = location or os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
        token = access_token or os.getenv("GOOGLE_VERTEX_ACCESS_TOKEN")
        if not proj:
            raise ValueError(
                "Claude on Vertex AI requires GOOGLE_VERTEX_PROJECT_ID "
                "(API key auth is not supported for partner models)."
            )
        kwargs: dict[str, Any] = {"project_id": proj, "region": loc}
        if token:
            kwargs["access_token"] = token
        return AnthropicVertex(**kwargs)

    @classmethod
    def list_models(
        cls,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        timeout: int = 10,
        **kw: object,
    ) -> list[str] | None:
        """List models available via Vertex AI."""
        key = api_key or os.getenv("GOOGLE_VERTEX_API_KEY")
        proj = project_id or os.getenv("GOOGLE_VERTEX_PROJECT_ID")
        if not key and not proj:
            return None

        model_ids: list[str] = []

        # Gemini models via google-genai
        if genai is not None:
            try:
                client = cls._build_gemini_client(api_key=key, project_id=proj, location=location)
                for m in client.models.list():
                    name = getattr(m, "name", None)
                    if name:
                        model_id = name.rsplit("/", 1)[-1] if "/" in name else name
                        model_ids.append(model_id)
            except Exception:
                logger.debug("Vertex AI Gemini list_models failed", exc_info=True)

        # Claude models — known available on Vertex AI Model Garden
        if proj and AnthropicVertex is not None:
            _KNOWN_CLAUDE_VERTEX = [
                "claude-sonnet-4-6-20250514",
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-haiku-4-5-20251001",
            ]
            model_ids.extend(_KNOWN_CLAUDE_VERTEX)

        return model_ids or None

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        model: str = "gemini-2.5-flash",
        access_token: str | None = None,
    ):
        if _is_claude_model(model):
            if AnthropicVertex is None:
                raise RuntimeError("anthropic package is not installed. Install it with: pip install anthropic[vertex]")
        else:
            if genai is None:
                raise RuntimeError(
                    "google-genai package is not installed. Install it with: pip install prompture[google]"
                )

        self.api_key = api_key or os.getenv("GOOGLE_VERTEX_API_KEY")
        self.project_id = project_id or os.getenv("GOOGLE_VERTEX_PROJECT_ID")
        self.location = location or os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
        self.access_token = access_token or os.getenv("GOOGLE_VERTEX_ACCESS_TOKEN")
        self.model = model

        if not self.api_key and not self.project_id:
            raise ValueError(
                "Google Vertex AI credentials not found. "
                "Set GOOGLE_VERTEX_API_KEY (Gemini) or GOOGLE_VERTEX_PROJECT_ID (Gemini + Claude)."
            )

        # Build the appropriate client for the initial model
        self._gemini_client: Any = None
        self._claude_client: Any = None
        if _is_claude_model(model):
            self._claude_client = self._build_claude_client(self.project_id, self.location, self.access_token)
        else:
            self._gemini_client = self._build_gemini_client(self.api_key, self.project_id, self.location)

        self.options: dict[str, Any] = {}

    def _get_gemini_client(self) -> Any:
        if self._gemini_client is None:
            self._gemini_client = self._build_gemini_client(self.api_key, self.project_id, self.location)
        return self._gemini_client

    def _get_claude_client(self) -> Any:
        if self._claude_client is None:
            self._claude_client = self._build_claude_client(self.project_id, self.location, self.access_token)
        return self._claude_client

    # ── Cost helpers ──────────────────────────────────────────────────

    def _calculate_cost_chars(self, prompt_chars: int, completion_chars: int) -> float:
        from ..infra.model_rates import get_model_rates

        live_rates = get_model_rates(self._PROVIDER, self.model)
        if not live_rates:
            # Fall back to upstream provider rates
            upstream = "anthropic" if _is_claude_model(self.model) else "google"
            live_rates = get_model_rates(upstream, self.model)
        if live_rates:
            est_prompt = prompt_chars / 4
            est_completion = completion_chars / 4
            return round(
                (est_prompt / 1_000_000) * live_rates["input"] + (est_completion / 1_000_000) * live_rates["output"],
                6,
            )
        return 0.0

    def _cost_for_tokens(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = self._calculate_cost(self._PROVIDER, self.model, prompt_tokens, completion_tokens)
        if cost == 0.0:
            upstream = "anthropic" if _is_claude_model(self.model) else "google"
            cost = self._calculate_cost(upstream, self.model, prompt_tokens, completion_tokens)
        return cost

    # ── Gemini usage extraction ───────────────────────────────────────

    def _extract_gemini_usage(self, response: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)
            cost = self._cost_for_tokens(prompt_tokens, completion_tokens)
        else:
            total_prompt_chars = sum(
                len(msg.get("content", "")) for msg in messages if isinstance(msg.get("content"), str)
            )
            completion_chars = len(response.text) if response.text else 0
            prompt_tokens = total_prompt_chars // 4
            completion_tokens = completion_chars // 4
            total_tokens = prompt_tokens + completion_tokens
            cost = self._calculate_cost_chars(total_prompt_chars, completion_chars)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(cost, 6),
        }

    # ── Message helpers ───────────────────────────────────────────────

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if _is_claude_model(self.model):
            from .vision_helpers import _prepare_claude_vision_messages

            return _prepare_claude_vision_messages(messages)
        from .vision_helpers import _prepare_google_vision_messages

        return _prepare_google_vision_messages(messages)

    @staticmethod
    def _extract_system_and_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_content = None
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                api_messages.append(msg)
        return system_content, api_messages

    def _build_gemini_generation_args(
        self, messages: list[dict[str, Any]], options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        merged = {**self.options, **(options or {})}
        config_dict: dict[str, Any] = {}

        if "temperature" in merged:
            config_dict["temperature"] = merged["temperature"]
        if "max_tokens" in merged:
            config_dict["max_output_tokens"] = merged["max_tokens"]
        if "top_p" in merged:
            config_dict["top_p"] = merged["top_p"]
        if "top_k" in merged:
            config_dict["top_k"] = merged["top_k"]

        for k, v in merged.get("generation_config", {}).items():
            config_dict.setdefault(k, v)

        if merged.get("safety_settings"):
            config_dict["safety_settings"] = merged["safety_settings"]

        if merged.get("json_mode"):
            config_dict["response_mime_type"] = "application/json"
            if merged.get("json_schema"):
                config_dict["response_schema"] = merged["json_schema"]

        system_instruction = None
        contents: list[dict[str, Any]] = []
        tool_call_names: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content if isinstance(content, str) else str(content)
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    parts: list[Any] = []
                    if content:
                        parts.append(content)
                    for tc in tool_calls:
                        fn = tc.get("function", tc)
                        name = fn.get("name", "")
                        tc_id = tc.get("id", "")
                        tool_call_names[tc_id] = name
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                        parts.append(types.Part(function_call=types.FunctionCall(name=name, args=args)))
                    contents.append({"role": "model", "parts": parts})
                elif msg.get("_vision_parts"):
                    contents.append({"role": "model", "parts": content})
                else:
                    contents.append({"role": "model", "parts": [content]})
            elif role == "tool":
                tc_id = msg.get("tool_call_id", "")
                name = tool_call_names.get(tc_id, "unknown_tool")
                result_content = content
                if isinstance(result_content, str):
                    try:
                        result_content = json.loads(result_content)
                    except (json.JSONDecodeError, TypeError):
                        result_content = {"result": result_content}
                if not isinstance(result_content, dict):
                    result_content = {"result": str(result_content)}
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            types.Part(function_response=types.FunctionResponse(name=name, response=result_content))
                        ],
                    }
                )
            else:
                if msg.get("_vision_parts"):
                    contents.append({"role": "user", "parts": content})
                else:
                    contents.append({"role": "user", "parts": [content]})

        gen_input: str | list[dict[str, Any]]
        if len(contents) == 1:
            parts = contents[0]["parts"]
            if len(parts) == 1 and isinstance(parts[0], str):
                gen_input = parts[0]
            else:
                gen_input = contents
        else:
            gen_input = contents

        if system_instruction:
            config_dict["system_instruction"] = system_instruction

        return gen_input, config_dict

    # ══════════════════════════════════════════════════════════════════
    #  CLAUDE BACKEND (via anthropic[vertex])
    # ══════════════════════════════════════════════════════════════════

    def _claude_generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        client = self._get_claude_client()
        opts = {**{"temperature": 0.0, "max_tokens": 4096}, **options}
        model = opts.get("model", self.model)

        system_content, api_messages = self._extract_system_and_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": opts["max_tokens"],
        }
        if "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]
        if system_content:
            kwargs["system"] = system_content

        if options.get("json_mode") and options.get("json_schema"):
            tool_def = {
                "name": "extract_json",
                "description": "Extract structured data matching the schema",
                "input_schema": options["json_schema"],
            }
            resp = client.messages.create(
                **kwargs,
                tools=[tool_def],
                tool_choice={"type": "tool", "name": "extract_json"},
            )
            text = ""
            for block in resp.content:
                if block.type == "tool_use":
                    text = json.dumps(block.input)
                    break
        else:
            resp = client.messages.create(**kwargs)
            text = resp.content[0].text

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        cost = self._cost_for_tokens(prompt_tokens, completion_tokens)

        return {
            "text": text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": round(cost, 6),
                "raw_response": dict(resp),
                "model_name": model,
            },
        }

    def _claude_generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        client = self._get_claude_client()
        opts = {**{"temperature": 0.0, "max_tokens": 4096}, **options}
        model = opts.get("model", self.model)

        system_content, api_messages = self._extract_system_and_messages(messages)

        anthropic_tools = []
        for t in tools:
            if "type" in t and t["type"] == "function":
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

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": opts["max_tokens"],
            "tools": anthropic_tools,
        }
        if "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]
        if system_content:
            kwargs["system"] = system_content

        resp = client.messages.create(**kwargs)

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        cost = self._cost_for_tokens(prompt_tokens, completion_tokens)

        text = ""
        tool_calls_out: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls_out.append({"id": block.id, "name": block.name, "arguments": block.input})

        return {
            "text": text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": round(cost, 6),
                "raw_response": dict(resp),
                "model_name": model,
            },
            "tool_calls": tool_calls_out,
            "stop_reason": resp.stop_reason,
        }

    def _claude_generate_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        client = self._get_claude_client()
        opts = {**{"temperature": 0.0, "max_tokens": 4096}, **options}
        model = opts.get("model", self.model)

        system_content, api_messages = self._extract_system_and_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": opts["max_tokens"],
        }
        if "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]
        if system_content:
            kwargs["system"] = system_content

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        with client.messages.stream(**kwargs) as stream:
            for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta" and hasattr(event, "delta"):
                        delta_text = getattr(event.delta, "text", "")
                        if delta_text:
                            full_text += delta_text
                            yield {"type": "delta", "text": delta_text}
                    elif event.type == "message_delta" and hasattr(event, "usage"):
                        completion_tokens = getattr(event.usage, "output_tokens", 0)
                    elif event.type == "message_start" and hasattr(event, "message"):
                        usage = getattr(event.message, "usage", None)
                        if usage:
                            prompt_tokens = getattr(usage, "input_tokens", 0)

        total_tokens = prompt_tokens + completion_tokens
        cost = self._cost_for_tokens(prompt_tokens, completion_tokens)

        yield {
            "type": "done",
            "text": full_text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": round(cost, 6),
                "raw_response": {},
                "model_name": model,
            },
        }

    # ══════════════════════════════════════════════════════════════════
    #  GEMINI BACKEND (via google-genai)
    # ══════════════════════════════════════════════════════════════════

    def _gemini_generate(self, messages: list[dict[str, Any]], options: dict[str, Any] | None = None) -> dict[str, Any]:
        gen_input, config_dict = self._build_gemini_generation_args(messages, options)

        try:
            config = types.GenerateContentConfig(**config_dict)
            response = self._get_gemini_client().models.generate_content(
                model=self.model, contents=gen_input, config=config
            )
            if not response.text:
                raise ValueError("Empty response from Vertex AI model")

            usage_meta = self._extract_gemini_usage(response, messages)
            return {
                "text": response.text,
                "meta": {
                    **usage_meta,
                    "raw_response": getattr(response, "prompt_feedback", None),
                    "model_name": self.model,
                },
            }
        except Exception as e:
            raise RuntimeError(f"Vertex AI Gemini request failed: {e}") from e

    def _gemini_generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        gen_input, config_dict = self._build_gemini_generation_args(self._prepare_messages(messages), options)

        function_declarations = []
        for t in tools:
            if "type" in t and t["type"] == "function":
                fn = t["function"]
                params = fn.get("parameters")
                decl = types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    **({"parameters_json_schema": params} if params else {}),
                )
                function_declarations.append(decl)
            elif "name" in t:
                params = t.get("parameters") or t.get("input_schema")
                decl = types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    **({"parameters_json_schema": params} if params else {}),
                )
                function_declarations.append(decl)

        config_dict["tools"] = [types.Tool(function_declarations=function_declarations)]

        try:
            config = types.GenerateContentConfig(**config_dict)
            response = self._get_gemini_client().models.generate_content(
                model=self.model, contents=gen_input, config=config
            )
            usage_meta = self._extract_gemini_usage(response, messages)
            meta = {**usage_meta, "raw_response": getattr(response, "prompt_feedback", None), "model_name": self.model}

            text = ""
            tool_calls_out: list[dict[str, Any]] = []
            stop_reason = "stop"

            for candidate in response.candidates or []:
                if candidate.content is None or candidate.content.parts is None:
                    continue
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
                    if hasattr(part, "function_call") and part.function_call is not None and part.function_call.name:
                        fc = part.function_call
                        tool_calls_out.append(
                            {
                                "id": str(uuid.uuid4()),
                                "name": fc.name,
                                "arguments": dict(fc.args) if fc.args else {},
                            }
                        )
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason is not None:
                    stop_reason = {1: "stop", 2: "max_tokens", 3: "safety", 4: "recitation", 5: "other"}.get(
                        finish_reason, "stop"
                    )
            if tool_calls_out:
                stop_reason = "tool_use"

            return {"text": text, "meta": meta, "tool_calls": tool_calls_out, "stop_reason": stop_reason}
        except Exception as e:
            raise RuntimeError(f"Vertex AI Gemini tool call failed: {e}") from e

    def _gemini_generate_stream(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        gen_input, config_dict = self._build_gemini_generation_args(self._prepare_messages(messages), options)

        try:
            config = types.GenerateContentConfig(**config_dict)
            response = self._get_gemini_client().models.generate_content_stream(
                model=self.model, contents=gen_input, config=config
            )
            full_text = ""
            for chunk in response:
                chunk_text = getattr(chunk, "text", None) or ""
                if chunk_text:
                    full_text += chunk_text
                    yield {"type": "delta", "text": chunk_text}

            usage_meta = self._extract_gemini_usage(response, messages)
            yield {
                "type": "done",
                "text": full_text,
                "meta": {**usage_meta, "raw_response": {}, "model_name": self.model},
            }
        except Exception as e:
            raise RuntimeError(f"Vertex AI Gemini streaming failed: {e}") from e

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API — dispatches to the right backend
    # ══════════════════════════════════════════════════════════════════

    def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        if _is_claude_model(self.model):
            return self._claude_generate(self._prepare_messages(messages), options or {})
        return self._gemini_generate(self._prepare_messages(messages), options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        prepared = self._prepare_messages(messages)
        if _is_claude_model(self.model):
            return self._claude_generate(prepared, options)
        return self._gemini_generate(prepared, options)

    def generate_messages_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]:
        if _is_claude_model(self.model):
            return self._claude_generate_with_tools(self._prepare_messages(messages), tools, options)
        return self._gemini_generate_with_tools(messages, tools, options)

    def generate_messages_stream(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        if _is_claude_model(self.model):
            return self._claude_generate_stream(self._prepare_messages(messages), options)
        return self._gemini_generate_stream(messages, options)
