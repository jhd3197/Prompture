"""Driver for Azure OpenAI Service with multi-endpoint and multi-backend support.

Supports:
- Multiple Azure endpoints with per-model config resolution
- OpenAI models (gpt-*, o1-*, o3-*, o4-*) via AzureOpenAI SDK
- Claude models (claude-*) via Anthropic SDK with Azure endpoint
- Mistral models (mistral-*, mixtral-*) via OpenAI-compatible protocol

Requires the ``openai`` package. Claude backend also requires ``anthropic``.
"""

import json
import logging
import os
from typing import Any

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None  # type: ignore[misc, assignment]

try:
    import anthropic
except Exception:
    anthropic = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin, prepare_strict_schema
from .azure_config import classify_backend, resolve_config
from .base import Driver

logger = logging.getLogger(__name__)


class AzureDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_vision = True

    # All pricing and model config now resolved from JSON rate files (KB) and
    # models.dev live data.  See prompture/infra/rates/azure.json.
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment_id: str | None = None,
        model: str = "gpt-4o-mini",
        claude_api_key: str | None = None,
        claude_endpoint: str | None = None,
        mistral_api_key: str | None = None,
        mistral_endpoint: str | None = None,
    ):
        self.model = model
        # Store default config from env vars (may be partial/None)
        self._default_config = {
            "api_key": api_key or os.getenv("AZURE_API_KEY"),
            "endpoint": endpoint or os.getenv("AZURE_API_ENDPOINT"),
            "deployment_id": deployment_id or os.getenv("AZURE_DEPLOYMENT_ID"),
            "api_version": os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
        }
        # Per-backend defaults: separate credentials for Claude/Mistral on Azure.
        # Falls back to the main default for any field not explicitly set.
        self._backend_defaults: dict[str, dict[str, Any]] = {}

        c_key = claude_api_key or os.getenv("AZURE_CLAUDE_API_KEY")
        c_ep = claude_endpoint or os.getenv("AZURE_CLAUDE_ENDPOINT")
        if c_key or c_ep:
            self._backend_defaults["claude"] = {
                "api_key": c_key or self._default_config["api_key"],
                "endpoint": c_ep or self._default_config["endpoint"],
                "api_version": os.getenv("AZURE_CLAUDE_API_VERSION", "2024-06-01"),
            }

        m_key = mistral_api_key or os.getenv("AZURE_MISTRAL_API_KEY")
        m_ep = mistral_endpoint or os.getenv("AZURE_MISTRAL_ENDPOINT")
        if m_key or m_ep:
            self._backend_defaults["mistral"] = {
                "api_key": m_key or self._default_config["api_key"],
                "endpoint": m_ep or self._default_config["endpoint"],
                "api_version": os.getenv("AZURE_MISTRAL_API_VERSION", "2024-02-15-preview"),
            }

        # Client caches: (endpoint, key) → client instance
        self._openai_clients: dict[tuple[str, str], AzureOpenAI] = {}
        self._anthropic_clients: dict[tuple[str, str], Any] = {}

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    def _resolve_model_config(self, model: str, options: dict[str, Any]) -> dict[str, Any]:
        """Resolve Azure config for this model using the priority chain.

        Uses backend-specific defaults (Claude, Mistral) when available,
        falling back to the generic default config.
        """
        override = options.pop("azure_config", None)
        backend = classify_backend(model)
        default = self._backend_defaults.get(backend, self._default_config)
        return resolve_config(model, override=override, default_config=default)

    def _get_openai_client(self, config: dict[str, Any]) -> "AzureOpenAI":
        """Get or create an AzureOpenAI client for the given config."""
        if AzureOpenAI is None:
            raise RuntimeError("openai package (>=1.0.0) with AzureOpenAI not installed")
        cache_key = (config["endpoint"], config["api_key"])
        if cache_key not in self._openai_clients:
            self._openai_clients[cache_key] = AzureOpenAI(
                api_key=config["api_key"],
                api_version=config.get("api_version", "2024-02-15-preview"),
                azure_endpoint=config["endpoint"],
            )
        return self._openai_clients[cache_key]

    def _get_anthropic_client(self, config: dict[str, Any]) -> Any:
        """Get or create an Anthropic client for the given Azure config."""
        if anthropic is None:
            raise RuntimeError("anthropic package not installed (required for Claude on Azure)")
        cache_key = (config["endpoint"], config["api_key"])
        if cache_key not in self._anthropic_clients:
            self._anthropic_clients[cache_key] = anthropic.Anthropic(
                base_url=config["endpoint"],
                api_key=config["api_key"],
            )
        return self._anthropic_clients[cache_key]

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(self._prepare_messages(messages), options)

    def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        config = self._resolve_model_config(model, options)
        backend = classify_backend(model)

        if backend == "claude":
            return self._generate_claude(messages, options, config, model)
        else:
            # Both "openai" and "mistral" use the OpenAI-compatible protocol
            return self._generate_openai(messages, options, config, model)

    def _generate_openai(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        config: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        """Generate via Azure OpenAI (or Mistral OpenAI-compat) endpoint."""
        client = self._get_openai_client(config)
        deployment_id = config.get("deployment_id") or model

        model_config = self._get_model_config("azure", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        kwargs = {
            "model": deployment_id,
            "messages": messages,
        }
        kwargs[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                schema_copy = prepare_strict_schema(json_schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": schema_copy,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)  # type: ignore[arg-type]

        # Extract usage
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        # Calculate cost via shared mixin
        total_cost = self._calculate_cost("azure", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp.model_dump(),
            "model_name": model,
            "deployment_id": deployment_id,
        }

        text = resp.choices[0].message.content
        return {"text": text, "meta": meta}

    def _generate_claude(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        config: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        """Generate via Anthropic SDK with Azure endpoint."""
        client = self._get_anthropic_client(config)

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}

        # Anthropic requires system messages as a top-level parameter
        system_content = None
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                api_messages.append(msg)

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

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens

        total_cost = self._calculate_cost("azure", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": dict(resp),
            "model_name": model,
        }

        text_result = text or ""
        return {"text": text_result, "meta": meta}

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
        config = self._resolve_model_config(model, options)
        backend = classify_backend(model)

        if backend == "claude":
            return self._generate_claude_with_tools(messages, tools, options, config, model)
        else:
            return self._generate_openai_with_tools(messages, tools, options, config, model)

    def _generate_openai_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
        config: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        """Tool calling via Azure OpenAI endpoint."""
        client = self._get_openai_client(config)
        deployment_id = config.get("deployment_id") or model

        model_config = self._get_model_config("azure", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("azure", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 4096, **options}

        kwargs: dict[str, Any] = {
            "model": deployment_id,
            "messages": messages,
            "tools": tools,
        }
        kwargs[tokens_param] = opts.get("max_tokens", 4096)

        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        resp = client.chat.completions.create(**kwargs)

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)
        total_cost = self._calculate_cost("azure", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp.model_dump(),
            "model_name": model,
            "deployment_id": deployment_id,
        }

        choice = resp.choices[0]
        text = choice.message.content or ""
        stop_reason = choice.finish_reason

        tool_calls_out: list[dict[str, Any]] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    raw = tc.function.arguments
                    if stop_reason == "length":
                        logger.warning(
                            "Tool arguments for %s were truncated due to max_tokens limit. "
                            "Increase max_tokens in options to allow longer tool outputs. "
                            "Truncated arguments: %r",
                            tc.function.name,
                            raw[:200] if raw else raw,
                        )
                    else:
                        logger.warning(
                            "Failed to parse tool arguments for %s: %r",
                            tc.function.name,
                            raw,
                        )
                    args = {}
                tool_calls_out.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": args,
                    }
                )

        return {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }

    def _generate_claude_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
        config: dict[str, Any],
        model: str,
    ) -> dict[str, Any]:
        """Tool calling via Anthropic SDK with Azure endpoint."""
        client = self._get_anthropic_client(config)

        opts = {**{"temperature": 0.0, "max_tokens": 4096}, **options}

        system_content = None
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                api_messages.append(msg)

        # Convert tools from OpenAI format to Anthropic format if needed
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
            "temperature": opts["temperature"],
            "max_tokens": opts["max_tokens"],
            "tools": anthropic_tools,
        }
        if system_content:
            kwargs["system"] = system_content

        resp = client.messages.create(**kwargs)

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost("azure", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": dict(resp),
            "model_name": model,
        }

        text = ""
        tool_calls_out: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls_out.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        return {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": resp.stop_reason,
        }
