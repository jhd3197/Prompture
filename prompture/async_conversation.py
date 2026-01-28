"""Async stateful multi-turn conversation support for Prompture."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, Union

from pydantic import BaseModel

from .async_driver import AsyncDriver
from .callbacks import DriverCallbacks
from .drivers.async_registry import get_async_driver_for_model
from .field_definitions import get_registry_snapshot
from .tools import (
    clean_json_text,
    convert_value,
    get_field_default,
)

logger = logging.getLogger("prompture.async_conversation")


class AsyncConversation:
    """Async stateful multi-turn conversation with an LLM.

    Mirrors :class:`Conversation` but all methods are ``async``.

    Example::

        conv = AsyncConversation("openai/gpt-4", system_prompt="You are a data extractor")
        r1 = await conv.ask_for_json("Extract names from: John, age 30", name_schema)
        r2 = await conv.ask_for_json("Now extract ages", age_schema)
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        driver: AsyncDriver | None = None,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
        callbacks: DriverCallbacks | None = None,
    ) -> None:
        if model_name is None and driver is None:
            raise ValueError("Either model_name or driver must be provided")

        if driver is not None:
            self._driver = driver
        else:
            self._driver = get_async_driver_for_model(model_name)

        if callbacks is not None:
            self._driver.callbacks = callbacks

        self._model_name = model_name or ""
        self._system_prompt = system_prompt
        self._options = dict(options) if options else {}
        self._messages: list[dict[str, str]] = []
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "turns": 0,
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def messages(self) -> list[dict[str, str]]:
        """Read-only view of the conversation history."""
        return list(self._messages)

    @property
    def usage(self) -> dict[str, Any]:
        """Accumulated token/cost totals across all turns."""
        return dict(self._usage)

    def clear(self) -> None:
        """Reset message history (keeps system_prompt and driver)."""
        self._messages.clear()

    def add_context(self, role: str, content: str) -> None:
        """Seed the history with a user or assistant message."""
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        self._messages.append({"role": role, "content": content})

    def usage_summary(self) -> str:
        """Human-readable summary of accumulated usage."""
        u = self._usage
        return f"Conversation: {u['total_tokens']:,} tokens across {u['turns']} turn(s) costing ${u['cost']:.4f}"

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def _build_messages(self, user_content: str) -> list[dict[str, str]]:
        """Build the full messages array for an API call."""
        msgs: list[dict[str, str]] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        msgs.extend(self._messages)
        msgs.append({"role": "user", "content": user_content})
        return msgs

    def _accumulate_usage(self, meta: dict[str, Any]) -> None:
        self._usage["prompt_tokens"] += meta.get("prompt_tokens", 0)
        self._usage["completion_tokens"] += meta.get("completion_tokens", 0)
        self._usage["total_tokens"] += meta.get("total_tokens", 0)
        self._usage["cost"] += meta.get("cost", 0.0)
        self._usage["turns"] += 1

    async def ask(
        self,
        content: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a message and get a raw text response (async)."""
        merged = {**self._options, **(options or {})}
        messages = self._build_messages(content)
        resp = await self._driver.generate_messages_with_hooks(messages, merged)

        text = resp.get("text", "")
        meta = resp.get("meta", {})

        self._messages.append({"role": "user", "content": content})
        self._messages.append({"role": "assistant", "content": text})
        self._accumulate_usage(meta)

        return text

    async def ask_for_json(
        self,
        content: str,
        json_schema: dict[str, Any],
        *,
        ai_cleanup: bool = True,
        options: dict[str, Any] | None = None,
        output_format: Literal["json", "toon"] = "json",
        json_mode: Literal["auto", "on", "off"] = "auto",
    ) -> dict[str, Any]:
        """Send a message with schema enforcement and get structured JSON back (async)."""
        merged = {**self._options, **(options or {})}

        schema_string = json.dumps(json_schema, indent=2)

        use_json_mode = False
        if json_mode == "on":
            use_json_mode = True
        elif json_mode == "auto":
            use_json_mode = getattr(self._driver, "supports_json_mode", False)

        if use_json_mode:
            merged = {**merged, "json_mode": True}
            if getattr(self._driver, "supports_json_schema", False):
                merged["json_schema"] = json_schema

        if use_json_mode and getattr(self._driver, "supports_json_schema", False):
            instruct = "Extract data matching the requested schema.\nIf a value is unknown use null."
        elif use_json_mode:
            instruct = (
                "Return a JSON object that validates against this schema:\n"
                f"{schema_string}\n\n"
                "If a value is unknown use null."
            )
        else:
            instruct = (
                "Return only a single JSON object (no markdown, no extra text) that validates against this JSON schema:\n"
                f"{schema_string}\n\n"
                "If a value is unknown use null. Use double quotes for keys and strings."
            )

        full_user_content = f"{content}\n\n{instruct}"

        messages = self._build_messages(full_user_content)
        resp = await self._driver.generate_messages_with_hooks(messages, merged)

        text = resp.get("text", "")
        meta = resp.get("meta", {})

        self._messages.append({"role": "user", "content": content})

        cleaned = clean_json_text(text)
        try:
            json_obj = json.loads(cleaned)
        except json.JSONDecodeError:
            if ai_cleanup:
                from .async_core import clean_json_text_with_ai

                cleaned = await clean_json_text_with_ai(self._driver, cleaned, self._model_name, merged)
                json_obj = json.loads(cleaned)
            else:
                raise

        self._messages.append({"role": "assistant", "content": cleaned})
        self._accumulate_usage(meta)

        model_name = self._model_name
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]

        usage = {
            **meta,
            "raw_response": resp,
            "model_name": model_name or getattr(self._driver, "model", ""),
        }

        result: dict[str, Any] = {
            "json_string": cleaned,
            "json_object": json_obj,
            "usage": usage,
            "output_format": output_format,
        }

        if output_format == "toon":
            try:
                import toon

                result["toon_string"] = toon.encode(json_obj)
            except ImportError:
                raise RuntimeError("TOON requested but 'python-toon' is not installed.") from None

        return result

    async def extract_with_model(
        self,
        model_cls: type[BaseModel],
        text: str,
        *,
        instruction_template: str = "Extract information from the following text:",
        ai_cleanup: bool = True,
        output_format: Literal["json", "toon"] = "json",
        options: dict[str, Any] | None = None,
        json_mode: Literal["auto", "on", "off"] = "auto",
    ) -> dict[str, Any]:
        """Extract structured information into a Pydantic model with conversation context (async)."""
        from .core import normalize_field_value

        schema = model_cls.model_json_schema()
        content_prompt = f"{instruction_template} {text}"

        result = await self.ask_for_json(
            content=content_prompt,
            json_schema=schema,
            ai_cleanup=ai_cleanup,
            options=options,
            output_format=output_format,
            json_mode=json_mode,
        )

        json_object = result["json_object"]
        schema_properties = schema.get("properties", {})

        for field_name, field_info in model_cls.model_fields.items():
            if field_name in json_object and field_name in schema_properties:
                field_def = {
                    "nullable": not schema_properties[field_name].get("type")
                    or "null"
                    in (
                        schema_properties[field_name].get("anyOf", [])
                        if isinstance(schema_properties[field_name].get("anyOf"), list)
                        else []
                    ),
                    "default": field_info.default
                    if hasattr(field_info, "default") and field_info.default is not ...
                    else None,
                }
                json_object[field_name] = normalize_field_value(
                    json_object[field_name], field_info.annotation, field_def
                )

        model_instance = model_cls(**json_object)

        result_dict = {
            "json_string": result["json_string"],
            "json_object": result["json_object"],
            "usage": result["usage"],
        }
        result_dict["model"] = model_instance

        return type(
            "ExtractResult",
            (dict,),
            {
                "__getattr__": lambda self, key: self.get(key),
                "__call__": lambda self: self["model"],
            },
        )(result_dict)

    # ------------------------------------------------------------------
    # Internal: stepwise with shared context
    # ------------------------------------------------------------------

    async def _stepwise_extract(
        self,
        model_cls: type[BaseModel],
        text: str,
        instruction_template: str,
        ai_cleanup: bool,
        fields: list[str] | None,
        field_definitions: dict[str, Any] | None,
        json_mode: Literal["auto", "on", "off"],
    ) -> dict[str, Union[str, dict[str, Any]]]:
        """Stepwise extraction using async conversation context between fields."""
        if field_definitions is None:
            field_definitions = get_registry_snapshot()

        data: dict[str, Any] = {}
        validation_errors: list[str] = []
        field_results: dict[str, Any] = {}

        accumulated_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "model_name": self._model_name,
            "field_usages": {},
        }

        valid_fields = set(model_cls.model_fields.keys())
        if fields is not None:
            invalid_fields = set(fields) - valid_fields
            if invalid_fields:
                raise KeyError(f"Fields not found in model: {', '.join(invalid_fields)}")
            field_items = [(name, model_cls.model_fields[name]) for name in fields]
        else:
            field_items = list(model_cls.model_fields.items())

        # Seed conversation with the source text
        self.add_context("user", f"I need to extract information from this text:\n\n{text}")
        self.add_context(
            "assistant", "I'll help you extract the information from that text. What would you like to extract?"
        )

        for field_name, field_info in field_items:
            logger.debug("[stepwise-conv] Extracting field: %s", field_name)

            field_schema = {
                "value": {
                    "type": "integer" if field_info.annotation is int else "string",
                    "description": field_info.description or f"Value for {field_name}",
                }
            }

            try:
                prompt = instruction_template.format(field_name=field_name)
                result = await self.ask_for_json(
                    content=f"{prompt} {text}",
                    json_schema=field_schema,
                    ai_cleanup=ai_cleanup,
                    json_mode=json_mode,
                )

                field_usage = result.get("usage", {})
                accumulated_usage["prompt_tokens"] += field_usage.get("prompt_tokens", 0)
                accumulated_usage["completion_tokens"] += field_usage.get("completion_tokens", 0)
                accumulated_usage["total_tokens"] += field_usage.get("total_tokens", 0)
                accumulated_usage["cost"] += field_usage.get("cost", 0.0)
                accumulated_usage["field_usages"][field_name] = field_usage

                extracted_value = result["json_object"]["value"]
                if isinstance(extracted_value, dict) and "value" in extracted_value:
                    raw_value = extracted_value["value"]
                else:
                    raw_value = extracted_value

                from .core import normalize_field_value

                field_def = {}
                if field_definitions and field_name in field_definitions:
                    field_def = field_definitions[field_name] if isinstance(field_definitions[field_name], dict) else {}

                nullable = field_def.get("nullable", True)
                default_value = field_def.get("default")
                if (
                    default_value is None
                    and hasattr(field_info, "default")
                    and field_info.default is not ...
                    and str(field_info.default) != "PydanticUndefined"
                ):
                    default_value = field_info.default

                normalize_def = {"nullable": nullable, "default": default_value}
                raw_value = normalize_field_value(raw_value, field_info.annotation, normalize_def)

                try:
                    converted_value = convert_value(raw_value, field_info.annotation, allow_shorthand=True)
                    data[field_name] = converted_value
                    field_results[field_name] = {"status": "success", "used_default": False}
                except ValueError as e:
                    error_msg = f"Type conversion failed for {field_name}: {e!s}"
                    has_default = _has_default(field_name, field_info, field_definitions)
                    if not has_default:
                        validation_errors.append(error_msg)
                    default_value = get_field_default(field_name, field_info, field_definitions)
                    data[field_name] = default_value
                    field_results[field_name] = {
                        "status": "conversion_failed",
                        "error": error_msg,
                        "used_default": True,
                    }

            except Exception as e:
                error_msg = f"Extraction failed for {field_name}: {e!s}"
                has_default = _has_default(field_name, field_info, field_definitions)
                if not has_default:
                    validation_errors.append(error_msg)
                default_value = get_field_default(field_name, field_info, field_definitions)
                data[field_name] = default_value
                field_results[field_name] = {"status": "extraction_failed", "error": error_msg, "used_default": True}
                accumulated_usage["field_usages"][field_name] = {
                    "error": str(e),
                    "status": "failed",
                    "used_default": True,
                    "default_value": default_value,
                }

        if validation_errors:
            accumulated_usage["validation_errors"] = validation_errors

        try:
            model_instance = model_cls(**data)
            model_dict = model_instance.model_dump()

            class ExtendedJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (datetime, date)):
                        return obj.isoformat()
                    if isinstance(obj, Decimal):
                        return str(obj)
                    return super().default(obj)

            json_string = json.dumps(model_dict, cls=ExtendedJSONEncoder)

            result = {
                "json_string": json_string,
                "json_object": json.loads(json_string),
                "usage": accumulated_usage,
                "field_results": field_results,
            }
            result["model"] = model_instance
            return type(
                "ExtractResult",
                (dict,),
                {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: self["model"]},
            )(result)
        except Exception as e:
            error_msg = f"Model validation error: {e!s}"
            if "validation_errors" not in accumulated_usage:
                accumulated_usage["validation_errors"] = []
            accumulated_usage["validation_errors"].append(error_msg)

            error_result = {
                "json_string": "{}",
                "json_object": {},
                "usage": accumulated_usage,
                "field_results": field_results,
                "error": error_msg,
            }
            return type(
                "ExtractResult",
                (dict,),
                {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: None},
            )(error_result)


def _has_default(field_name: str, field_info: Any, field_definitions: dict[str, Any] | None) -> bool:
    """Check whether a Pydantic field has a usable default value."""
    if field_definitions and field_name in field_definitions:
        fd = field_definitions[field_name]
        if isinstance(fd, dict) and "default" in fd:
            return True
    if hasattr(field_info, "default"):
        val = field_info.default
        if val is not ... and str(val) != "PydanticUndefined":
            return True
    return False
