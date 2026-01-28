"""Async Anthropic Claude driver. Requires the ``anthropic`` package."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import anthropic
except Exception:
    anthropic = None

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin
from .claude_driver import ClaudeDriver


class AsyncClaudeDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True

    MODEL_PRICING = ClaudeDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL_NAME", "claude-3-5-haiku-20241022")

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Native JSON mode: use tool-use for schema enforcement
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                tool_def = {
                    "name": "extract_json",
                    "description": "Extract structured data matching the schema",
                    "input_schema": json_schema,
                }
                resp = await client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[tool_def],
                    tool_choice={"type": "tool", "name": "extract_json"},
                    temperature=opts["temperature"],
                    max_tokens=opts["max_tokens"],
                )
                text = ""
                for block in resp.content:
                    if block.type == "tool_use":
                        text = json.dumps(block.input)
                        break
            else:
                resp = await client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=opts["temperature"],
                    max_tokens=opts["max_tokens"],
                )
                text = resp.content[0].text
        else:
            resp = await client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=opts["temperature"],
                max_tokens=opts["max_tokens"],
            )
            text = resp.content[0].text

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens

        total_cost = self._calculate_cost("claude", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost,
            "raw_response": dict(resp),
            "model_name": model,
        }

        return {"text": text, "meta": meta}
