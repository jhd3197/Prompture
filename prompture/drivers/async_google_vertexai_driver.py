"""Async Google Vertex AI driver — dual-backend.

* **Gemini models** → ``google-genai`` async API
* **Claude models** → ``anthropic[vertex]`` async API (``AsyncAnthropicVertex``)
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

try:
    from anthropic import AsyncAnthropicVertex
except ImportError:
    AsyncAnthropicVertex = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from .async_base import AsyncDriver
from .google_vertexai_driver import GoogleVertexAIDriver, _is_claude_model

logger = logging.getLogger(__name__)


class AsyncGoogleVertexAIDriver(CostMixin, AsyncDriver):
    """Async driver for Google Vertex AI (Gemini + Claude via Model Garden)."""

    supports_json_mode = True
    supports_json_schema = True
    supports_vision = True
    supports_tool_use = True
    supports_streaming = True
    supports_messages = True

    MODEL_PRICING = GoogleVertexAIDriver.MODEL_PRICING

    _PROVIDER = "google_vertexai"

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        model: str = "gemini-2.5-flash",
    ):
        if _is_claude_model(model):
            if AsyncAnthropicVertex is None:
                raise RuntimeError("anthropic package is not installed. Install it with: pip install anthropic[vertex]")
        else:
            if genai is None:
                raise RuntimeError(
                    "google-genai package is not installed. Install it with: pip install prompture[google]"
                )

        self.api_key = api_key or os.getenv("GOOGLE_VERTEX_API_KEY")
        self.project_id = project_id or os.getenv("GOOGLE_VERTEX_PROJECT_ID")
        self.location = location or os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
        self.model = model

        if not self.api_key and not self.project_id:
            raise ValueError(
                "Google Vertex AI credentials not found. "
                "Set GOOGLE_VERTEX_API_KEY (Gemini) or GOOGLE_VERTEX_PROJECT_ID (Gemini + Claude)."
            )

        self._gemini_client: Any = None
        self._claude_client: Any = None
        if _is_claude_model(model):
            self._claude_client = self._build_claude_client()
        else:
            self._gemini_client = GoogleVertexAIDriver._build_gemini_client(
                self.api_key, self.project_id, self.location
            )

        self.options: dict[str, Any] = {}

    def _build_claude_client(self) -> Any:
        proj = self.project_id
        if not proj:
            raise ValueError(
                "Claude on Vertex AI requires GOOGLE_VERTEX_PROJECT_ID "
                "(API key auth is not supported for partner models)."
            )
        return AsyncAnthropicVertex(project_id=proj, region=self.location)

    def _get_gemini_client(self) -> Any:
        if self._gemini_client is None:
            self._gemini_client = GoogleVertexAIDriver._build_gemini_client(
                self.api_key, self.project_id, self.location
            )
        return self._gemini_client

    def _get_claude_client(self) -> Any:
        if self._claude_client is None:
            self._claude_client = self._build_claude_client()
        return self._claude_client

    # ── Cost helpers ──────────────────────────────────────────────────

    def _cost_for_tokens(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = self._calculate_cost(self._PROVIDER, self.model, prompt_tokens, completion_tokens)
        if cost == 0.0:
            upstream = "anthropic" if _is_claude_model(self.model) else "google"
            cost = self._calculate_cost(upstream, self.model, prompt_tokens, completion_tokens)
        return cost

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
            cost = 0.0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(cost, 6),
        }

    # ── Message helpers ───────────────────────────────────────��───────

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if _is_claude_model(self.model):
            from .vision_helpers import _prepare_claude_vision_messages

            return _prepare_claude_vision_messages(messages)
        from .vision_helpers import _prepare_google_vision_messages

        return _prepare_google_vision_messages(messages)

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
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content if isinstance(content, str) else str(content)
            else:
                gemini_role = "model" if role == "assistant" else "user"
                if msg.get("_vision_parts"):
                    contents.append({"role": gemini_role, "parts": content})
                else:
                    contents.append({"role": gemini_role, "parts": [content]})

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

    # ── Claude backend ────────────────────────────────────────────────

    async def _claude_generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        client = self._get_claude_client()
        opts = {**{"temperature": 0.0, "max_tokens": 4096}, **options}
        model = opts.get("model", self.model)

        system_content, api_messages = GoogleVertexAIDriver._extract_system_and_messages(messages)

        kwargs: dict[str, Any] = {"model": model, "messages": api_messages, "max_tokens": opts["max_tokens"]}
        if "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]
        if system_content:
            kwargs["system"] = system_content

        if options.get("json_mode") and options.get("json_schema"):
            tool_def = {
                "name": "extract_json",
                "description": "Extract structured data",
                "input_schema": options["json_schema"],
            }
            resp = await client.messages.create(
                **kwargs, tools=[tool_def], tool_choice={"type": "tool", "name": "extract_json"}
            )
            text = ""
            for block in resp.content:
                if block.type == "tool_use":
                    text = json.dumps(block.input)
                    break
        else:
            resp = await client.messages.create(**kwargs)
            text = resp.content[0].text

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        cost = self._cost_for_tokens(prompt_tokens, completion_tokens)

        return {
            "text": text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": round(cost, 6),
                "raw_response": dict(resp),
                "model_name": model,
            },
        }

    async def _claude_generate_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]:
        client = self._get_claude_client()
        opts = {**{"temperature": 0.0, "max_tokens": 4096}, **options}
        model = opts.get("model", self.model)

        system_content, api_messages = GoogleVertexAIDriver._extract_system_and_messages(messages)

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

        resp = await client.messages.create(**kwargs)

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
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
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": round(cost, 6),
                "raw_response": dict(resp),
                "model_name": model,
            },
            "tool_calls": tool_calls_out,
            "stop_reason": resp.stop_reason,
        }

    async def _claude_generate_stream(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        client = self._get_claude_client()
        opts = {**{"temperature": 0.0, "max_tokens": 4096}, **options}
        model = opts.get("model", self.model)

        system_content, api_messages = GoogleVertexAIDriver._extract_system_and_messages(messages)

        kwargs: dict[str, Any] = {"model": model, "messages": api_messages, "max_tokens": opts["max_tokens"]}
        if "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]
        if system_content:
            kwargs["system"] = system_content

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
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

        cost = self._cost_for_tokens(prompt_tokens, completion_tokens)
        yield {
            "type": "done",
            "text": full_text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": round(cost, 6),
                "raw_response": {},
                "model_name": model,
            },
        }

    # ── Gemini backend ──────────────────────────────────────────���─────

    async def _gemini_generate(
        self, messages: list[dict[str, Any]], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        gen_input, config_dict = self._build_gemini_generation_args(messages, options)
        try:
            config = types.GenerateContentConfig(**config_dict)
            response = await self._get_gemini_client().aio.models.generate_content(
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

    async def _gemini_generate_stream(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        gen_input, config_dict = self._build_gemini_generation_args(self._prepare_messages(messages), options)
        try:
            config = types.GenerateContentConfig(**config_dict)
            response = await self._get_gemini_client().aio.models.generate_content_stream(
                model=self.model, contents=gen_input, config=config
            )
            full_text = ""
            async for chunk in response:
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

    # ── Public API ────────────────────────────────────────────────────

    async def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        prepared = self._prepare_messages(messages)
        if _is_claude_model(self.model):
            return await self._claude_generate(prepared, options or {})
        return await self._gemini_generate(prepared, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        prepared = self._prepare_messages(messages)
        if _is_claude_model(self.model):
            return await self._claude_generate(prepared, options)
        return await self._gemini_generate(prepared, options)

    async def generate_messages_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]:
        prepared = self._prepare_messages(messages)
        if _is_claude_model(self.model):
            return await self._claude_generate_with_tools(prepared, tools, options)
        # Gemini tool use — reuse sync logic isn't possible, skip for now
        raise NotImplementedError("Async Gemini tool use on Vertex AI not yet implemented")

    async def generate_messages_stream(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        if _is_claude_model(self.model):
            async for chunk in self._claude_generate_stream(self._prepare_messages(messages), options):
                yield chunk
        else:
            async for chunk in self._gemini_generate_stream(messages, options):
                yield chunk
