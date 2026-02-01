# Driver Template

Every Prompture driver follows this skeleton. The sync driver uses `requests`,
the async driver uses `httpx`.

## Sync Driver — `prompture/drivers/{provider}_driver.py`

```python
"""{Provider} driver implementation.
Requires the `requests` package. Uses {PROVIDER}_API_KEY env var.

All pricing comes from models.dev (provider: "{models_dev_name}") — no hardcoded pricing.
"""

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

import requests

from ..cost_mixin import CostMixin, prepare_strict_schema
from ..driver import Driver

logger = logging.getLogger(__name__)


class {Provider}Driver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_vision = False  # set True if the provider supports image input
    supports_messages = True

    # All pricing resolved live from models.dev (provider: "{models_dev_name}")
    # If models.dev does NOT have this provider, add hardcoded pricing:
    #   MODEL_PRICING = {
    #       "model-name": {"prompt": 0.001, "completion": 0.002},
    #   }
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "default-model",
        endpoint: str = "https://api.example.com/v1",
    ):
        self.api_key = api_key or os.getenv("{PROVIDER}_API_KEY")
        if not self.api_key:
            raise ValueError("{Provider} API key not found. Set {PROVIDER}_API_KEY env var.")

        self.model = model
        self.base_url = endpoint.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(messages, options)

    def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)

        # Per-model config from models.dev (tokens_param, supports_temperature, etc.)
        model_config = self._get_model_config("{provider}", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        # Validate capabilities (logs warnings if model doesn't support requested features)
        self._validate_model_capabilities(
            "{provider}",
            model,
            using_json_schema=bool(options.get("json_schema")),
        )

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        # Native JSON mode — check per-model capabilities before sending response_format
        if options.get("json_mode"):
            from ..model_rates import get_model_capabilities

            caps = get_model_capabilities("{provider}", model)
            is_reasoning = caps is not None and caps.is_reasoning is True
            model_supports_structured = (
                caps is None or caps.supports_structured_output is not False
            ) and not is_reasoning

            if model_supports_structured:
                json_schema = options.get("json_schema")
                if json_schema:
                    schema_copy = prepare_strict_schema(json_schema)
                    data["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extraction",
                            "strict": True,
                            "schema": schema_copy,
                        },
                    }
                else:
                    data["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=120,
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"{Provider} API request failed: {e!s}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{Provider} API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Cost calculated from models.dev live rates, falling back to MODEL_PRICING
        total_cost = self._calculate_cost("{provider}", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

        message = resp["choices"][0]["message"]
        text = message.get("content") or ""

        # Reasoning models may return content in reasoning_content when content is empty
        if not text and message.get("reasoning_content"):
            text = message["reasoning_content"]

        return {"text": text, "meta": meta}

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls."""
        model = options.get("model", self.model)
        model_config = self._get_model_config("{provider}", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("{provider}", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        if "tool_choice" in options:
            data["tool_choice"] = options["tool_choice"]

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=120,
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"{Provider} API request failed: {e!s}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{Provider} API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        total_cost = self._calculate_cost("{provider}", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

        choice = resp["choices"][0]
        text = choice["message"].get("content") or ""
        stop_reason = choice.get("finish_reason")

        tool_calls_out: list[dict[str, Any]] = []
        for tc in choice["message"].get("tool_calls", []):
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls_out.append({
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": args,
            })

        return {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield response chunks via streaming API."""
        model = options.get("model", self.model)
        model_config = self._get_model_config("{provider}", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                # Reasoning models stream thinking via reasoning_content
                if not content:
                    content = delta.get("reasoning_content") or ""
                if content:
                    full_text += content
                    yield {"type": "delta", "text": content}

        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost("{provider}", model, prompt_tokens, completion_tokens)

        yield {
            "type": "done",
            "text": full_text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": round(total_cost, 6),
                "raw_response": {},
                "model_name": model,
            },
        }
```

## Lazy Import Pattern (for optional SDKs)

```python
def __init__(self, ...):
    self._client = None
    # defer import

def _ensure_client(self):
    if self._client is not None:
        return
    try:
        from some_sdk import Client
    except ImportError:
        raise ImportError(
            "The 'some-sdk' package is required. "
            "Install with: pip install prompture[provider]"
        )
    self._client = Client(api_key=self.api_key)
```

## Existing Drivers for Reference

| Driver | File | SDK | Auth | models.dev |
|--------|------|-----|------|------------|
| OpenAI | `openai_driver.py` | `openai` | API key | `openai` |
| Claude | `claude_driver.py` | `anthropic` | API key | `anthropic` |
| Google | `google_driver.py` | `google-generativeai` | API key | `google` |
| Groq | `groq_driver.py` | `groq` | API key | `groq` |
| Grok | `grok_driver.py` | `requests` | API key | `xai` |
| Moonshot | `moonshot_driver.py` | `requests` | API key + endpoint | `moonshotai` |
| Z.ai | `zai_driver.py` | `requests` | API key + endpoint | `zai` |
| ModelScope | `modelscope_driver.py` | `requests` | API key + endpoint | — |
| OpenRouter | `openrouter_driver.py` | `requests` | API key | `openrouter` |
| Ollama | `ollama_driver.py` | `requests` | Endpoint URL | — |
| LM Studio | `lmstudio_driver.py` | `requests` | Endpoint URL | — |
| AirLLM | `airllm_driver.py` | `airllm` (lazy) | None (local) | — |
