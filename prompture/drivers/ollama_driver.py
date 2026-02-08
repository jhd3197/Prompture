import json
import logging
import os
import uuid
from collections.abc import Iterator
from typing import Any, Optional

import requests

from ..driver import Driver

logger = logging.getLogger(__name__)


class OllamaDriver(Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_streaming = True
    supports_tool_use = True
    supports_vision = True

    # Ollama is free – costs are always zero.
    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(self, endpoint: str | None = None, model: str = "llama3"):
        # Allow override via env var
        self.endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
        self.model = model
        self.options = {}  # Initialize empty options dict

        # Validate connection to Ollama server
        self._validate_connection()

    def _validate_connection(self):
        """Validate connection to the Ollama server."""
        try:
            # Send a simple HEAD request to check if server is accessible
            # Use the base API endpoint without the specific path
            base_url = self.endpoint.split("/api/")[0]
            health_url = f"{base_url}/api/version"

            logger.debug(f"Validating connection to Ollama server at: {health_url}")
            response = requests.head(health_url, timeout=5)
            response.raise_for_status()
            logger.debug("Connection to Ollama server validated successfully")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not validate connection to Ollama server: {e}")
            # We don't raise an error here to allow for delayed server startup
            # The actual error will be raised when generate() is called

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_ollama_vision_messages

        return _prepare_ollama_vision_messages(messages)

    def generate(self, prompt: str, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        # Merge instance options with call-specific options
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        payload = {
            "prompt": prompt,
            "model": merged_options.get("model", self.model),
            "stream": False,
        }

        # Native JSON mode / structured output support
        if merged_options.get("json_mode"):
            json_schema = merged_options.get("json_schema")
            payload["format"] = json_schema if json_schema else "json"

        # Add any Ollama-specific options from merged_options
        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        try:
            logger.debug(f"Sending request to Ollama endpoint: {self.endpoint}")
            logger.debug(f"Request payload: {payload}")

            r = requests.post(self.endpoint, json=payload, timeout=merged_options.get("timeout", 300))
            logger.debug(f"Response status code: {r.status_code}")

            r.raise_for_status()

            response_text = r.text
            logger.debug(f"Raw response text: {response_text}")

            response_data = r.json()
            logger.debug(f"Parsed response data: {response_data}")

            if not isinstance(response_data, dict):
                raise ValueError(f"Expected dict response, got {type(response_data)}")

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to Ollama endpoint: {e}")
            # Preserve original exception
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from Ollama endpoint: {e}")
            # Preserve original exception
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            # Re-raise JSONDecodeError with more context
            raise json.JSONDecodeError(f"Invalid JSON response from Ollama: {e.msg}", e.doc, e.pos) from e
        except Exception as e:
            logger.error(f"Unexpected error in Ollama request: {e}")
            # Only wrap unknown exceptions in RuntimeError
            raise RuntimeError(f"Ollama request failed: {e}") from e

        # Extract token counts
        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Build meta info
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        # Ollama returns text in "response"
        text = response_data.get("response", "")
        reasoning_content = response_data.get("thinking") or None

        # Reasoning models may return content only in thinking
        if not text and reasoning_content:
            text = reasoning_content

        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls via Ollama's /api/chat endpoint."""
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        chat_endpoint = self.endpoint.replace("/api/generate", "/api/chat")

        payload: dict[str, Any] = {
            "model": merged_options.get("model", self.model),
            "messages": messages,
            "tools": tools,
            "stream": False,
        }

        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        try:
            logger.debug(f"Sending tool use request to Ollama endpoint: {chat_endpoint}")
            r = requests.post(chat_endpoint, json=payload, timeout=merged_options.get("timeout", 300))
            r.raise_for_status()
            response_data = r.json()

            if not isinstance(response_data, dict):
                raise ValueError(f"Expected dict response, got {type(response_data)}")
        except requests.exceptions.ConnectionError:
            raise
        except requests.exceptions.HTTPError:
            raise
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON response from Ollama: {e.msg}", e.doc, e.pos) from e
        except Exception as e:
            raise RuntimeError(f"Ollama tool use request failed: {e}") from e

        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        message = response_data.get("message", {})
        text = message.get("content") or ""
        reasoning_content = message.get("thinking") or None
        stop_reason = response_data.get("done_reason", "stop")

        if not text and reasoning_content:
            text = reasoning_content

        tool_calls_out: list[dict[str, Any]] = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            # Ollama returns arguments as a dict already (no JSON string parsing needed)
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    if stop_reason == "length":
                        logger.warning(
                            "Tool arguments for %s were truncated due to max_tokens limit. "
                            "Increase max_tokens in options to allow longer tool outputs. "
                            "Truncated arguments: %r",
                            func.get("name", ""), args[:200] if args else args,
                        )
                    else:
                        logger.warning(
                            "Failed to parse tool arguments for %s: %r",
                            func.get("name", ""), args,
                        )
                    args = {}
            tool_calls_out.append({
                # Ollama does not return tool_call IDs — generate one locally
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "name": func.get("name", ""),
                "arguments": args,
            })

        result: dict[str, Any] = {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield response chunks via Ollama streaming API."""
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        chat_endpoint = self.endpoint.replace("/api/generate", "/api/chat")

        payload: dict[str, Any] = {
            "model": merged_options.get("model", self.model),
            "messages": messages,
            "stream": True,
        }

        if merged_options.get("json_mode"):
            json_schema = merged_options.get("json_schema")
            payload["format"] = json_schema if json_schema else "json"
        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        full_text = ""
        full_reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0

        r = requests.post(chat_endpoint, json=payload, timeout=merged_options.get("timeout", 300), stream=True)
        r.raise_for_status()

        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)
            else:
                msg = chunk.get("message", {})
                thinking = msg.get("thinking", "")
                if thinking:
                    full_reasoning += thinking
                content = msg.get("content", "")
                if content:
                    full_text += content
                    yield {"type": "delta", "text": content}

        total_tokens = prompt_tokens + completion_tokens
        done_chunk: dict[str, Any] = {
            "type": "done",
            "text": full_text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": 0.0,
                "raw_response": {},
                "model_name": merged_options.get("model", self.model),
            },
        }
        if full_reasoning:
            done_chunk["reasoning_content"] = full_reasoning
        yield done_chunk

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        """Use Ollama's /api/chat endpoint for multi-turn conversations."""
        messages = self._prepare_messages(messages)
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        # Derive the chat endpoint from the generate endpoint
        chat_endpoint = self.endpoint.replace("/api/generate", "/api/chat")

        payload: dict[str, Any] = {
            "model": merged_options.get("model", self.model),
            "messages": messages,
            "stream": False,
        }

        # Native JSON mode / structured output support
        if merged_options.get("json_mode"):
            json_schema = merged_options.get("json_schema")
            payload["format"] = json_schema if json_schema else "json"

        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        try:
            logger.debug(f"Sending chat request to Ollama endpoint: {chat_endpoint}")
            r = requests.post(chat_endpoint, json=payload, timeout=merged_options.get("timeout", 300))
            r.raise_for_status()
            response_data = r.json()

            if not isinstance(response_data, dict):
                raise ValueError(f"Expected dict response, got {type(response_data)}")
        except requests.exceptions.ConnectionError:
            raise
        except requests.exceptions.HTTPError:
            raise
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON response from Ollama: {e.msg}", e.doc, e.pos) from e
        except Exception as e:
            raise RuntimeError(f"Ollama chat request failed: {e}") from e

        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        # Chat endpoint returns response in message.content
        message = response_data.get("message", {})
        text = message.get("content", "")
        reasoning_content = message.get("thinking") or None

        if not text and reasoning_content:
            text = reasoning_content

        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result
