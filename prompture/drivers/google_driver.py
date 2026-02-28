import json
import logging
import os
import uuid
from collections.abc import Iterator
from typing import Any, Optional

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from .base import Driver

logger = logging.getLogger(__name__)


class GoogleDriver(CostMixin, Driver):
    """Driver for Google's Generative AI API (Gemini)."""

    supports_json_mode = True
    supports_json_schema = True
    supports_vision = True
    supports_tool_use = True
    supports_streaming = True

    # All pricing and model config now resolved from JSON rate files (KB) and
    # models.dev live data.  See prompture/infra/rates/google.json.
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    @classmethod
    def list_models(cls, *, api_key: str | None = None, timeout: int = 10, **kw: object) -> list[str] | None:
        """List models available via the Google Generative AI API."""
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            return None
        try:
            client = genai.Client(api_key=key)
            model_ids: list[str] = []
            for m in client.models.list():
                name = getattr(m, "name", None)
                if name:
                    # The API returns "models/gemini-1.5-pro" — strip the prefix
                    model_ids.append(name.removeprefix("models/"))
            return model_ids
        except Exception:
            logger.debug("GoogleDriver.list_models failed", exc_info=True)
            return None

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro"):
        """Initialize the Google Driver.

        Args:
            api_key: Google API key. If not provided, will look for GOOGLE_API_KEY env var
            model: Model to use. Defaults to "gemini-1.5-pro"
        """
        if genai is None:
            raise RuntimeError("google-genai package is not installed. Install it with: pip install google-genai")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY env var or pass api_key to constructor")

        self.model = model

        # Create google-genai client
        self._client = genai.Client(api_key=self.api_key)
        self.options: dict[str, Any] = {}

        # Validate connection and model availability
        self._validate_connection()

    def _validate_connection(self) -> None:
        """Validate connection to Google's API and model availability."""
        try:
            # List models to validate API key and connectivity
            list(self._client.models.list(config={"page_size": 1}))
            logger.debug("Connection to Google API validated successfully")
        except Exception as e:
            logger.warning(f"Could not validate connection to Google API: {e}")
            raise

    def _calculate_cost_chars(self, prompt_chars: int, completion_chars: int) -> float:
        """Calculate cost from character counts.

        Estimates tokens from characters (~4 chars/token) and uses
        models.dev live rates.  Returns 0.0 if no rates available.
        """
        from ..infra.model_rates import get_model_rates

        live_rates = get_model_rates("google", self.model)
        if live_rates:
            est_prompt_tokens = prompt_chars / 4
            est_completion_tokens = completion_chars / 4
            prompt_cost = (est_prompt_tokens / 1_000_000) * live_rates["input"]
            completion_cost = (est_completion_tokens / 1_000_000) * live_rates["output"]
            return round(prompt_cost + completion_cost, 6)

        return 0.0

    def _extract_usage_metadata(self, response: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract token counts from response, falling back to character estimation."""
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)
            cost = self._calculate_cost("google", self.model, prompt_tokens, completion_tokens)
        else:
            # Fallback: estimate from character counts
            total_prompt_chars = 0
            for msg in messages:
                c = msg.get("content", "")
                if isinstance(c, str):
                    total_prompt_chars += len(c)
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, str):
                            total_prompt_chars += len(part)
                        elif isinstance(part, dict) and "text" in part:
                            total_prompt_chars += len(part["text"])
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

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_google_vision_messages

        return _prepare_google_vision_messages(messages)

    def _build_generation_args(
        self, messages: list[dict[str, Any]], options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        """Parse messages and options into (contents, config_dict) for generate_content.

        Returns the content input and a config dict suitable for
        constructing ``types.GenerateContentConfig(**config_dict)``.
        """
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        config_dict: dict[str, Any] = {}

        if "temperature" in merged_options:
            config_dict.setdefault("temperature", merged_options["temperature"])
        if "max_tokens" in merged_options:
            config_dict.setdefault("max_output_tokens", merged_options["max_tokens"])
        if "top_p" in merged_options:
            config_dict.setdefault("top_p", merged_options["top_p"])
        if "top_k" in merged_options:
            config_dict.setdefault("top_k", merged_options["top_k"])

        # Forward explicit generation_config entries
        for k, v in merged_options.get("generation_config", {}).items():
            config_dict.setdefault(k, v)

        # Forward safety_settings
        safety_settings = merged_options.get("safety_settings")
        if safety_settings:
            config_dict["safety_settings"] = safety_settings

        # Native JSON mode support
        if merged_options.get("json_mode"):
            config_dict["response_mime_type"] = "application/json"
            json_schema = merged_options.get("json_schema")
            if json_schema:
                config_dict["response_schema"] = json_schema

        # Convert messages to Gemini format, including tool calls/results
        system_instruction = None
        contents: list[dict[str, Any]] = []
        # Track tool_call id → name mapping for tool result messages
        tool_call_names: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content if isinstance(content, str) else str(content)

            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Assistant message with function calls → model with function_call parts
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
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=name,
                                    args=args,
                                )
                            )
                        )
                    contents.append({"role": "model", "parts": parts})
                else:
                    if msg.get("_vision_parts"):
                        contents.append({"role": "model", "parts": content})
                    else:
                        contents.append({"role": "model", "parts": [content]})

            elif role == "tool":
                # Tool result → user with function_response part
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
                parts = [
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response=result_content,
                        )
                    )
                ]
                contents.append({"role": "user", "parts": parts})

            else:
                # Regular user message
                if msg.get("_vision_parts"):
                    contents.append({"role": "user", "parts": content})
                else:
                    contents.append({"role": "user", "parts": [content]})

        # For a single message, unwrap only if it has exactly one string part
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

    def generate(self, prompt: str, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(self._prepare_messages(messages), options)

    def _do_generate(self, messages: list[dict[str, str]], options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        gen_input, config_dict = self._build_generation_args(messages, options)

        # Validate capabilities against models.dev metadata
        self._validate_model_capabilities(
            "google",
            self.model,
            using_json_schema=bool((options or {}).get("json_schema")),
        )

        try:
            logger.debug(f"Generating with model {self.model}")
            config = types.GenerateContentConfig(**config_dict)
            response = self._client.models.generate_content(
                model=self.model,
                contents=gen_input,
                config=config,
            )

            if not response.text:
                raise ValueError("Empty response from model")

            usage_meta = self._extract_usage_metadata(response, messages)

            meta = {
                **usage_meta,
                "raw_response": response.prompt_feedback if hasattr(response, "prompt_feedback") else None,
                "model_name": self.model,
            }

            return {"text": response.text, "meta": meta}

        except Exception as e:
            logger.error(f"Google API request failed: {e}")
            raise RuntimeError(f"Google API request failed: {e}") from e

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool/function calls."""
        model = options.get("model", self.model)
        self._validate_model_capabilities("google", model, using_tool_use=True)

        gen_input, config_dict = self._build_generation_args(self._prepare_messages(messages), options)

        # Convert tools from OpenAI format to Gemini function declarations
        function_declarations = []
        for t in tools:
            if "type" in t and t["type"] == "function":
                fn = t["function"]
                decl = types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                )
                params = fn.get("parameters")
                if params:
                    decl = types.FunctionDeclaration(
                        name=fn["name"],
                        description=fn.get("description", ""),
                        parameters_json_schema=params,
                    )
                function_declarations.append(decl)
            elif "name" in t:
                # Already in a generic format
                params = t.get("parameters") or t.get("input_schema")
                if params:
                    decl = types.FunctionDeclaration(
                        name=t["name"],
                        description=t.get("description", ""),
                        parameters_json_schema=params,
                    )
                else:
                    decl = types.FunctionDeclaration(
                        name=t["name"],
                        description=t.get("description", ""),
                    )
                function_declarations.append(decl)

        config_dict["tools"] = [types.Tool(function_declarations=function_declarations)]

        try:
            config = types.GenerateContentConfig(**config_dict)
            response = self._client.models.generate_content(
                model=self.model,
                contents=gen_input,
                config=config,
            )

            usage_meta = self._extract_usage_metadata(response, messages)
            meta = {
                **usage_meta,
                "raw_response": response.prompt_feedback if hasattr(response, "prompt_feedback") else None,
                "model_name": self.model,
            }

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
                    # Map Gemini finish reasons to standard stop reasons
                    reason_map = {1: "stop", 2: "max_tokens", 3: "safety", 4: "recitation", 5: "other"}
                    stop_reason = reason_map.get(finish_reason, "stop")

            if tool_calls_out:
                stop_reason = "tool_use"

            return {
                "text": text,
                "meta": meta,
                "tool_calls": tool_calls_out,
                "stop_reason": stop_reason,
            }

        except Exception as e:
            logger.error(f"Google API tool call request failed: {e}")
            raise RuntimeError(f"Google API tool call request failed: {e}") from e

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield response chunks via Gemini streaming API."""
        gen_input, config_dict = self._build_generation_args(self._prepare_messages(messages), options)

        try:
            config = types.GenerateContentConfig(**config_dict)
            response = self._client.models.generate_content_stream(
                model=self.model,
                contents=gen_input,
                config=config,
            )

            full_text = ""
            for chunk in response:
                chunk_text = getattr(chunk, "text", None) or ""
                if chunk_text:
                    full_text += chunk_text
                    yield {"type": "delta", "text": chunk_text}

            # After iteration completes, usage_metadata should be available
            usage_meta = self._extract_usage_metadata(response, messages)

            yield {
                "type": "done",
                "text": full_text,
                "meta": {
                    **usage_meta,
                    "raw_response": {},
                    "model_name": self.model,
                },
            }

        except Exception as e:
            logger.error(f"Google API streaming request failed: {e}")
            raise RuntimeError(f"Google API streaming request failed: {e}") from e
