"""Async core utilities: async versions of all public extraction functions."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, Union

try:
    import toon
except ImportError:
    toon = None

from pydantic import BaseModel

from ..drivers.async_base import AsyncDriver
from ..drivers.async_registry import get_async_driver_for_model
from .core import (
    _calculate_token_savings,
    _dataframe_to_toon,
    _json_to_toon,
    normalize_field_value,
)
from .fields import get_registry_snapshot
from .reasoning import (
    ReasoningStrategyProtocol,
    _strategy_name,
    apply_reasoning_strategy,
    auto_select_reasoning_strategy,
)
from .tools import (
    clean_json_text,
    convert_value,
    get_field_default,
)

logger = logging.getLogger("prompture.async_core")


async def clean_json_text_with_ai(
    driver: AsyncDriver, text: str, model_name: str = "", options: dict[str, Any] | None = None
) -> tuple[str, dict[str, Any]]:
    """Use LLM to fix malformed JSON strings (async version).

    Returns:
        A tuple of (cleaned_text, meta) where cleaned_text is the fixed
        JSON string and meta is the usage metadata dict from the driver.
    """
    if options is None:
        options = {}
    try:
        json.loads(text)
        return text, {}
    except json.JSONDecodeError:
        pass

    prompt = (
        "The following text is supposed to be a single JSON object, but it is malformed. "
        "Please correct it and return only the valid JSON object. Do not add any explanations or markdown. "
        f"The text to correct is:\n\n{text}"
    )
    resp = await driver.generate_with_hooks(prompt, options)
    raw = resp.get("text", "")
    meta = resp.get("meta", {})
    cleaned = clean_json_text(raw)
    return cleaned, meta


async def render_output(
    driver: AsyncDriver,
    content_prompt: str,
    output_format: Literal["text", "html", "markdown"] = "text",
    model_name: str = "",
    options: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Send a prompt and return the raw output in the requested format (async version)."""
    if options is None:
        options = {}
    if output_format not in ("text", "html", "markdown"):
        raise ValueError(f"Unsupported output_format '{output_format}'. Use 'text', 'html', or 'markdown'.")

    instruct = ""
    if output_format == "text":
        instruct = (
            "Return ONLY the raw text content. Do not use markdown formatting, "
            "code fences, or conversational filler. Just the text."
        )
    elif output_format == "html":
        instruct = (
            "Return ONLY valid HTML code. Do not wrap it in markdown code fences "
            "(like ```html ... ```). Do not include conversational filler."
        )
    elif output_format == "markdown":
        instruct = "Return valid markdown content. You may use standard markdown formatting."

    full_prompt = f"{content_prompt}\n\nSYSTEM INSTRUCTION: {instruct}"

    # Use generate_messages when system_prompt is provided
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ]
        resp = await driver.generate_messages_with_hooks(messages, options)
    else:
        resp = await driver.generate_with_hooks(full_prompt, options)
    raw = resp.get("text", "")

    if output_format in ("text", "html"):
        cleaned = raw.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2:
                cleaned = "\n".join(lines[1:-1])
        raw = cleaned

    usage = {
        **resp.get("meta", {}),
        "raw_response": resp,
        "total_tokens": resp.get("meta", {}).get("total_tokens", 0),
        "prompt_tokens": resp.get("meta", {}).get("prompt_tokens", 0),
        "completion_tokens": resp.get("meta", {}).get("completion_tokens", 0),
        "cost": resp.get("meta", {}).get("cost", 0.0),
        "model_name": model_name or getattr(driver, "model", ""),
    }

    return {"text": raw, "usage": usage, "output_format": output_format}


async def ask_for_json(
    driver: AsyncDriver,
    content_prompt: str,
    json_schema: dict[str, Any],
    ai_cleanup: bool = True,
    model_name: str = "",
    options: dict[str, Any] | None = None,
    output_format: Literal["json", "toon"] = "json",
    cache: bool | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Send a prompt and return structured JSON output plus usage metadata (async version)."""
    if options is None:
        options = {}
    if output_format not in ("json", "toon"):
        raise ValueError(f"Unsupported output_format '{output_format}'. Use 'json' or 'toon'.")

    # --- cache lookup ---
    from ..infra.cache import get_cache, make_cache_key

    _cache = get_cache()
    use_cache = cache if cache is not None else _cache.enabled
    _force = cache is True
    cache_key: str | None = None
    if use_cache:
        cache_key = make_cache_key(
            prompt=content_prompt,
            model_name=model_name,
            schema=json_schema,
            options=options,
            output_format=output_format,
        )
        cached = _cache.get(cache_key, force=_force)
        if cached is not None:
            cached["usage"]["cache_hit"] = True
            return cached  # type: ignore[no-any-return]

    schema_string = json.dumps(json_schema, indent=2)
    if output_format == "toon" and toon is None:
        raise RuntimeError(
            "TOON requested but 'python-toon' is not installed. Install it with 'pip install python-toon'."
        )

    # Determine whether to use native JSON mode
    use_json_mode = False
    if json_mode == "on":
        use_json_mode = True
    elif json_mode == "auto":
        use_json_mode = getattr(driver, "supports_json_mode", False)

    if use_json_mode:
        options = {**options, "json_mode": True}
        if getattr(driver, "supports_json_schema", False):
            options["json_schema"] = json_schema

    # Adjust instruction prompt based on JSON mode capabilities
    if use_json_mode and getattr(driver, "supports_json_schema", False):
        # Schema enforced by API — minimal instruction
        instruct = "Extract data matching the requested schema.\nIf a value is unknown use null."
    elif use_json_mode:
        # JSON guaranteed but schema not enforced by API
        instruct = (
            "Return a JSON object that validates against this schema:\n"
            f"{schema_string}\n\n"
            "If a value is unknown use null."
        )
    else:
        # Existing prompt-based enforcement
        instruct = (
            "Return only a single JSON object (no markdown, no extra text) that validates against this JSON schema:\n"
            f"{schema_string}\n\n"
            "If a value is unknown use null. Use double quotes for keys and strings."
        )
    if output_format == "toon":
        instruct += "\n\n(Respond with JSON only; Prompture will convert to TOON.)"

    full_prompt = f"{content_prompt}\n\n{instruct}"

    # Use generate_messages when system_prompt is provided
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ]
        resp = await driver.generate_messages_with_hooks(messages, options)
    else:
        resp = await driver.generate_with_hooks(full_prompt, options)
    raw = resp.get("text", "")
    cleaned = clean_json_text(raw)

    try:
        json_obj = json.loads(cleaned)
        json_string = cleaned
        toon_string = None
        if output_format == "toon":
            toon_string = toon.encode(json_obj)

        usage = {
            **resp.get("meta", {}),
            "raw_response": resp,
            "total_tokens": resp.get("meta", {}).get("total_tokens", 0),
            "prompt_tokens": resp.get("meta", {}).get("prompt_tokens", 0),
            "completion_tokens": resp.get("meta", {}).get("completion_tokens", 0),
            "cost": resp.get("meta", {}).get("cost", 0.0),
            "model_name": model_name or getattr(driver, "model", ""),
        }
        result = {"json_string": json_string, "json_object": json_obj, "usage": usage}
        if toon_string is not None:
            result["toon_string"] = toon_string
            result["output_format"] = "toon"
        else:
            result["output_format"] = "json"

        # --- cache store ---
        if use_cache and cache_key is not None:
            cached_copy = {**result, "usage": {**result["usage"], "raw_response": {}}}
            _cache.set(cache_key, cached_copy, force=_force)

        return result
    except json.JSONDecodeError as e:
        if ai_cleanup:
            cleaned_fixed, cleanup_meta = await clean_json_text_with_ai(driver, cleaned, model_name, options)

            try:
                json_obj = json.loads(cleaned_fixed)

                # Combine usage from original call and cleanup call
                original_meta = resp.get("meta", {})
                combined_usage = {
                    "prompt_tokens": original_meta.get("prompt_tokens", 0) + cleanup_meta.get("prompt_tokens", 0),
                    "completion_tokens": original_meta.get("completion_tokens", 0)
                    + cleanup_meta.get("completion_tokens", 0),
                    "total_tokens": original_meta.get("total_tokens", 0) + cleanup_meta.get("total_tokens", 0),
                    "cost": original_meta.get("cost", 0.0) + cleanup_meta.get("cost", 0.0),
                    "model_name": model_name or options.get("model", getattr(driver, "model", "")),
                    "raw_response": resp,
                    "ai_cleanup_used": True,
                }

                result = {
                    "json_string": cleaned_fixed,
                    "json_object": json_obj,
                    "usage": combined_usage,
                    "output_format": "json" if output_format != "toon" else "toon",
                }
                if output_format == "toon":
                    result["toon_string"] = toon.encode(json_obj)

                # --- cache store (ai cleanup path) ---
                if use_cache and cache_key is not None:
                    cached_copy = {**result, "usage": {**result["usage"], "raw_response": {}}}
                    _cache.set(cache_key, cached_copy, force=_force)

                return result
            except json.JSONDecodeError:
                raise e from None
        else:
            raise e


async def extract_and_jsonify(
    text: str,
    json_schema: dict[str, Any],
    *,
    model_name: str = "",
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    output_format: Literal["json", "toon"] = "json",
    options: dict[str, Any] | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
    reasoning_strategy: str | ReasoningStrategyProtocol | None = None,
) -> dict[str, Any]:
    """Extract structured information using automatic async driver selection (async version)."""
    if options is None:
        options = {}
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")

    actual_model = model_name or options.get("model", "")
    driver = options.pop("driver", None)

    if driver is None:
        if not actual_model:
            raise ValueError("Model name cannot be empty")
        if "/" not in actual_model:
            raise ValueError("Invalid model string format. Expected format: 'provider/model'")
        try:
            driver = get_async_driver_for_model(actual_model)
        except ValueError as e:
            if "Unsupported provider" in str(e):
                raise ValueError(f"Unsupported provider in model name: {actual_model}") from e
            raise

    try:
        provider, model_id = actual_model.split("/", 1)
        if not provider:
            raise ValueError("Provider cannot be empty in model name")
    except ValueError:
        provider = model_id = actual_model

    opts = {**options, "model": model_id}
    content_prompt = f"{instruction_template} {text}"
    if reasoning_strategy == "auto":
        reasoning_strategy = auto_select_reasoning_strategy(text, json_schema)
    content_prompt = apply_reasoning_strategy(content_prompt, reasoning_strategy)

    try:
        result = await ask_for_json(
            driver,
            content_prompt,
            json_schema,
            ai_cleanup,
            model_id,
            opts,
            output_format=output_format,
            json_mode=json_mode,
            system_prompt=system_prompt,
        )
        result["usage"]["reasoning_strategy"] = _strategy_name(reasoning_strategy)
        return result
    except Exception as e:
        if "pytest" in sys.modules:
            import pytest

            pytest.skip(f"Connection error occurred: {e}")
        raise


async def manual_extract_and_jsonify(
    driver: AsyncDriver,
    text: str,
    json_schema: dict[str, Any],
    model_name: str = "",
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    output_format: Literal["json", "toon"] = "json",
    options: dict[str, Any] | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
    reasoning_strategy: str | ReasoningStrategyProtocol | None = None,
) -> dict[str, Any]:
    """Extract structured information using an explicitly provided async driver."""
    if options is None:
        options = {}
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")

    logger.info("[async-manual] Starting async manual extraction")

    opts = dict(options)
    if model_name:
        opts["model"] = model_name

    content_prompt = f"{instruction_template} {text}"
    if reasoning_strategy == "auto":
        reasoning_strategy = auto_select_reasoning_strategy(text, json_schema)
    content_prompt = apply_reasoning_strategy(content_prompt, reasoning_strategy)

    result = await ask_for_json(
        driver,
        content_prompt,
        json_schema,
        ai_cleanup,
        model_name,
        opts,
        output_format=output_format,
        json_mode=json_mode,
        system_prompt=system_prompt,
    )
    result["usage"]["reasoning_strategy"] = _strategy_name(reasoning_strategy)
    return result


async def _async_chunked_extract(
    *,
    chunks: list[Any],
    model_cls: type[BaseModel],
    model_name: str | None,
    instruction_template: str,
    ai_cleanup: bool,
    output_format: Literal["json", "toon"],
    options: dict[str, Any],
    cache: bool | None,
    json_mode: Literal["auto", "on", "off"],
    system_prompt: str | None,
    reasoning_strategy: Any,
) -> dict[str, Any]:
    """Async chunked extraction: extract per chunk and merge results."""
    all_results: list[dict[str, Any]] = []
    total_usage: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "chunks_processed": 0,
    }

    for chunk in chunks:
        chunk_prefix = f"[Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}] "
        chunk_instruction = chunk_prefix + instruction_template

        result = await extract_with_model(
            model_cls,
            chunk.text,
            model_name,
            instruction_template=chunk_instruction,
            ai_cleanup=ai_cleanup,
            output_format=output_format,
            options=options,
            cache=cache,
            json_mode=json_mode,
            system_prompt=system_prompt,
            reasoning_strategy=reasoning_strategy,
        )
        all_results.append(result)

        usage = result.get("usage", {})
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)
        total_usage["cost"] += usage.get("cost", 0.0)
        total_usage["chunks_processed"] += 1

    if not all_results:
        raise ValueError("No chunks to extract from")

    merged_json: dict[str, Any] = {}
    schema = model_cls.model_json_schema()
    schema_properties = schema.get("properties", {})

    for field_name in schema_properties:
        field_schema = schema_properties[field_name]
        field_type = field_schema.get("type", "")
        is_array = field_type == "array"

        if is_array:
            merged_list: list[Any] = []
            for r in all_results:
                val = r.get("json_object", {}).get(field_name)
                if isinstance(val, list):
                    merged_list.extend(val)
            merged_json[field_name] = merged_list
        else:
            for r in all_results:
                val = r.get("json_object", {}).get(field_name)
                if val is not None:
                    merged_json[field_name] = val
                    break
            if field_name not in merged_json:
                merged_json[field_name] = None

    model_instance = model_cls(**merged_json)
    merged_json_str = json.dumps(merged_json, default=str, ensure_ascii=False)

    result_dict: dict[str, Any] = {
        "json_string": merged_json_str,
        "json_object": merged_json,
        "usage": total_usage,
        "model": model_instance,
    }

    return type(  # type: ignore[no-any-return]
        "ExtractResult",
        (dict,),
        {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: self["model"]},
    )(result_dict)


async def extract_with_model(
    model_cls: type[BaseModel],
    text: str | None = None,
    model_name: str | None = None,
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    output_format: Literal["json", "toon"] = "json",
    options: dict[str, Any] | None = None,
    cache: bool | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
    reasoning_strategy: str | ReasoningStrategyProtocol | None = None,
    source: Any = None,
    chunking: Any = None,
) -> dict[str, Any]:
    """Extract structured information into a Pydantic model instance (async version).

    Args:
        source: Path to a document file (str, Path, or DocumentContent) to
            ingest and extract from.  Mutually exclusive with *text*.
        chunking: Chunking configuration for large documents.  Pass ``True``
            for auto-chunking with defaults, a ``ChunkingConfig`` for full
            control, or ``None``/``False`` to disable.
    """
    if options is None:
        options = {}

    # --- Document source ingestion (lazy import) ---
    if source is not None:
        if text is not None and isinstance(text, str) and text.strip():
            raise ValueError("Cannot provide both 'text' and 'source'. Use one or the other.")

        from ..ingestion import ChunkingConfig as _ChunkingConfig
        from ..ingestion import async_ingest as _async_ingest
        from ..ingestion import chunk_document as _chunk_document

        doc = await _async_ingest(source)

        # --- Chunked extraction ---
        if chunking is True:
            chunking = _ChunkingConfig()
        if isinstance(chunking, _ChunkingConfig):
            chunks = _chunk_document(doc, chunking)
            if len(chunks) > 1:
                return await _async_chunked_extract(
                    chunks=chunks,
                    model_cls=model_cls,
                    model_name=model_name,
                    instruction_template=instruction_template,
                    ai_cleanup=ai_cleanup,
                    output_format=output_format,
                    options=options,
                    cache=cache,
                    json_mode=json_mode,
                    system_prompt=system_prompt,
                    reasoning_strategy=reasoning_strategy,
                )

        text = doc.text

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text input cannot be empty")

    # --- cache lookup ---
    from ..infra.cache import get_cache, make_cache_key

    _cache = get_cache()
    use_cache = cache if cache is not None else _cache.enabled
    _force = cache is True
    cache_key: str | None = None
    if use_cache:
        schema_for_key = model_cls.model_json_schema()
        cache_key = make_cache_key(
            prompt=f"{instruction_template} {text}",
            model_name=model_name or "",
            schema=schema_for_key,
            options=options,
            output_format=output_format,
            pydantic_qualname=model_cls.__qualname__,
        )
        cached = _cache.get(cache_key, force=_force)
        if cached is not None:
            cached["usage"]["cache_hit"] = True
            cached["model"] = model_cls(**cached["json_object"])
            return type(  # type: ignore[no-any-return]
                "ExtractResult",
                (dict,),
                {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: self["model"]},
            )(cached)

    logger.info("[async-extract] Starting async extract_with_model")

    schema = model_cls.model_json_schema()

    result = await extract_and_jsonify(
        text=text,
        json_schema=schema,
        model_name=model_name or "",
        instruction_template=instruction_template,
        ai_cleanup=ai_cleanup,
        output_format=output_format,
        options=options,
        json_mode=json_mode,
        system_prompt=system_prompt,
        reasoning_strategy=reasoning_strategy,
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
                json_object[field_name], field_info.annotation or type(json_object[field_name]), field_def
            )

    model_instance = model_cls(**json_object)

    result_dict = {"json_string": result["json_string"], "json_object": result["json_object"], "usage": result["usage"]}

    # --- cache store ---
    if use_cache and cache_key is not None:
        cached_copy = {
            "json_string": result_dict["json_string"],
            "json_object": result_dict["json_object"],
            "usage": {**result_dict["usage"], "raw_response": {}},
        }
        _cache.set(cache_key, cached_copy)

    result_dict["model"] = model_instance

    return type(  # type: ignore[no-any-return]
        "ExtractResult",
        (dict,),
        {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: self["model"]},
    )(result_dict)


async def stepwise_extract_with_model(
    model_cls: type[BaseModel],
    text: str,
    *,
    model_name: str,
    instruction_template: str = "Extract the {field_name} from the following text:",
    ai_cleanup: bool = True,
    fields: list[str] | None = None,
    field_definitions: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
    share_context: bool = False,
    reasoning_strategy: str | ReasoningStrategyProtocol | None = None,
) -> dict[str, Union[str, dict[str, Any]]]:
    """Extract information field-by-field using sequential async LLM calls."""
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")

    # When share_context=True, delegate to AsyncConversation-based extraction
    if share_context:
        from ..agents.async_conversation import AsyncConversation

        conv = AsyncConversation(model_name=model_name, system_prompt=system_prompt, options=options)
        return await conv._stepwise_extract(
            model_cls=model_cls,
            text=text,
            instruction_template=instruction_template,
            ai_cleanup=ai_cleanup,
            fields=fields,
            field_definitions=field_definitions,
            json_mode=json_mode,
        )

    if field_definitions is None:
        field_definitions = get_registry_snapshot()

    data = {}
    validation_errors = []
    field_results = {}
    options = options or {}

    accumulated_usage: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "model_name": model_name,
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

    for field_name, field_info in field_items:
        field_schema = {
            "value": {
                "type": "integer" if field_info.annotation is int else "string",
                "description": field_info.description or f"Value for {field_name}",
            }
        }

        try:
            result = await extract_and_jsonify(
                text=text,
                json_schema=field_schema,
                model_name=model_name,
                instruction_template=instruction_template.format(field_name=field_name),
                ai_cleanup=ai_cleanup,
                options=options,
                json_mode=json_mode,
                system_prompt=system_prompt,
                reasoning_strategy=reasoning_strategy,
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

            field_def: dict[str, Any] = {}
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
            raw_value = normalize_field_value(raw_value, field_info.annotation or type(raw_value), normalize_def)

            try:
                converted_value = convert_value(
                    raw_value, field_info.annotation or type(raw_value), allow_shorthand=True
                )
                data[field_name] = converted_value
                field_results[field_name] = {"status": "success", "used_default": False}
            except ValueError as e:
                error_msg = f"Type conversion failed for {field_name}: {e!s}"
                has_default = _has_default(field_name, field_info, field_definitions)
                if not has_default:
                    validation_errors.append(error_msg)
                default_value = get_field_default(field_name, field_info, field_definitions)
                data[field_name] = default_value
                field_results[field_name] = {"status": "conversion_failed", "error": error_msg, "used_default": True}
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
            def default(self, obj: Any) -> Any:
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
        return type(  # type: ignore[no-any-return]
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
        return type(  # type: ignore[no-any-return]
            "ExtractResult",
            (dict,),
            {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: None},
        )(error_result)


async def extract_from_data(
    data: Union[list[dict[str, Any]], dict[str, Any]],
    question: str,
    json_schema: dict[str, Any],
    *,
    model_name: str,
    data_key: str | None = None,
    instruction_template: str = "Analyze the following data and answer: {question}",
    ai_cleanup: bool = True,
    options: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Extract information from structured data via TOON format (async version)."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    if not json_schema:
        raise ValueError("JSON schema cannot be empty")
    if options is None:
        options = {}

    toon_data = _json_to_toon(data, data_key)

    json_data = json.dumps(data if isinstance(data, list) else data.get(data_key or "", data), indent=2)
    token_savings = _calculate_token_savings(json_data, toon_data)

    content_prompt = instruction_template.format(question=question)
    full_prompt = f"{content_prompt}\n\nData (in TOON format):\n{toon_data}"

    driver = get_async_driver_for_model(model_name)
    result = await ask_for_json(
        driver=driver,  # type: ignore[arg-type]
        content_prompt=full_prompt,
        json_schema=json_schema,
        ai_cleanup=ai_cleanup,
        model_name=model_name.split("/")[-1] if "/" in model_name else model_name,
        options=options,
        output_format="json",
        system_prompt=system_prompt,
    )

    result["toon_data"] = toon_data
    result["token_savings"] = token_savings
    return result


async def extract_from_pandas(
    df: Any,
    question: str,
    json_schema: dict[str, Any],
    *,
    model_name: str,
    instruction_template: str = "Analyze the following data and answer: {question}",
    ai_cleanup: bool = True,
    options: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Extract information from a Pandas DataFrame via TOON format (async version)."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    if not json_schema:
        raise ValueError("JSON schema cannot be empty")
    if options is None:
        options = {}

    toon_data = _dataframe_to_toon(df)

    json_data = df.to_json(indent=2, orient="records")
    token_savings = _calculate_token_savings(json_data, toon_data)

    dataframe_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    content_prompt = instruction_template.format(question=question)
    full_prompt = f"{content_prompt}\n\nData (in TOON format):\n{toon_data}"

    driver = get_async_driver_for_model(model_name)
    result = await ask_for_json(
        driver=driver,  # type: ignore[arg-type]
        content_prompt=full_prompt,
        json_schema=json_schema,
        ai_cleanup=ai_cleanup,
        model_name=model_name.split("/")[-1] if "/" in model_name else model_name,
        options=options,
        output_format="json",
        system_prompt=system_prompt,
    )

    result["toon_data"] = toon_data
    result["token_savings"] = token_savings
    result["dataframe_info"] = dataframe_info
    return result


async def gather_extract(
    text: str,
    json_schema: dict[str, Any],
    model_names: list[str],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Extract from the same text using multiple models concurrently.

    Args:
        text: The raw text to extract information from.
        json_schema: JSON schema defining the expected structure.
        model_names: List of model identifiers (e.g., ``["openai/gpt-4", "claude/claude-3-5-haiku-20241022"]``).
        **kwargs: Extra keyword arguments forwarded to :func:`extract_and_jsonify`.

    Returns:
        A list of result dicts, one per model (order matches *model_names*).
    """
    tasks = [extract_and_jsonify(text=text, json_schema=json_schema, model_name=name, **kwargs) for name in model_names]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
