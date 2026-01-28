"""Driver for Anthropic's Claude models. Requires the `anthropic` library.
Use with API key in CLAUDE_API_KEY env var or provide directly.
"""

import json
import os
from typing import Any

try:
    import anthropic
except Exception:
    anthropic = None

from ..cost_mixin import CostMixin
from ..driver import Driver


class ClaudeDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True

    # Claude pricing per 1000 tokens (prices should be kept current with Anthropic's pricing)
    MODEL_PRICING = {
        # Claude Opus 4.1
        "claude-opus-4-1-20250805": {
            "prompt": 0.015,  # $15 per 1M prompt tokens
            "completion": 0.075,  # $75 per 1M completion tokens
        },
        # Claude Opus 4.0
        "claude-opus-4-20250514": {
            "prompt": 0.015,  # $15 per 1M prompt tokens
            "completion": 0.075,  # $75 per 1M completion tokens
        },
        # Claude Sonnet 4.0
        "claude-sonnet-4-20250514": {
            "prompt": 0.003,  # $3 per 1M prompt tokens
            "completion": 0.015,  # $15 per 1M completion tokens
        },
        # Claude Sonnet 3.7
        "claude-3-7-sonnet-20250219": {
            "prompt": 0.003,  # $3 per 1M prompt tokens
            "completion": 0.015,  # $15 per 1M completion tokens
        },
        # Claude Haiku 3.5
        "claude-3-5-haiku-20241022": {
            "prompt": 0.0008,  # $0.80 per 1M prompt tokens
            "completion": 0.004,  # $4 per 1M completion tokens
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL_NAME", "claude-3-5-haiku-20241022")

    supports_messages = True

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(messages, options)

    def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)

        client = anthropic.Anthropic(api_key=self.api_key)

        # Anthropic requires system messages as a top-level parameter
        system_content = None
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                api_messages.append(msg)

        # Build common kwargs
        common_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": opts["temperature"],
            "max_tokens": opts["max_tokens"],
        }
        if system_content:
            common_kwargs["system"] = system_content

        # Native JSON mode: use tool-use for schema enforcement
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                tool_def = {
                    "name": "extract_json",
                    "description": "Extract structured data matching the schema",
                    "input_schema": json_schema,
                }
                resp = client.messages.create(
                    **common_kwargs,
                    tools=[tool_def],
                    tool_choice={"type": "tool", "name": "extract_json"},
                )
                text = ""
                for block in resp.content:
                    if block.type == "tool_use":
                        text = json.dumps(block.input)
                        break
            else:
                resp = client.messages.create(**common_kwargs)
                text = resp.content[0].text
        else:
            resp = client.messages.create(**common_kwargs)
            text = resp.content[0].text

        # Extract token usage from Claude response
        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens

        # Calculate cost via shared mixin
        total_cost = self._calculate_cost("claude", model, prompt_tokens, completion_tokens)

        # Create standardized meta object
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),  # Round to 6 decimal places
            "raw_response": dict(resp),
            "model_name": model,
        }

        return {"text": text, "meta": meta}
