"""Core utilities: Helpers for requesting JSON from LLM."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from .routing import RoutingConfig

import requests

try:
    import toon
except ImportError:
    toon = None

from pydantic import BaseModel

from .driver import Driver
from .drivers import get_driver_for_model
from .field_definitions import get_registry_snapshot
from .image import ImageInput, make_image
from .tools import (
    clean_json_text,
    convert_value,
    get_field_default,
)

logger = logging.getLogger("prompture.core")


def _record_usage_to_ledger(model_name: str, meta: dict[str, Any]) -> None:
    """Fire-and-forget ledger recording for standalone core functions."""
    from .ledger import _resolve_api_key_hash, record_model_usage

    record_model_usage(
        model_name,
        api_key_hash=_resolve_api_key_hash(model_name),
        tokens=meta.get("total_tokens", 0),
        cost=meta.get("cost", 0.0),
    )


def _build_content_with_images(text: str, images: list[ImageInput] | None = None) -> str | list[dict[str, Any]]:
    """Return plain string when no images, or a list of content blocks."""
    if not images:
        return text
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        ic = make_image(img)
        blocks.append({"type": "image", "source": ic})
    return blocks


def normalize_field_value(value: Any, field_type: type, field_def: dict[str, Any]) -> Any:
    """Normalize invalid values for fields based on their type and nullable status.

    This function handles post-processing of extracted values BEFORE Pydantic validation,
    converting invalid values (like empty strings for booleans) to proper defaults.

    Args:
        value: The extracted value from the LLM
        field_type: The expected Python type for this field
        field_def: The field definition dict containing nullable, default, etc.

    Returns:
        A normalized value suitable for the field type
    """
    nullable = field_def.get("nullable", True)
    default_value = field_def.get("default")

    # Special handling for boolean fields
    if field_type is bool or (hasattr(field_type, "__origin__") and field_type.__origin__ is bool):
        # If value is already a boolean, return it as-is
        if isinstance(value, bool):
            return value

        # For non-nullable booleans
        if not nullable:
            # Any non-empty string should be True, empty/None should be default
            if isinstance(value, str):
                return bool(value.strip()) if value.strip() else (default_value if default_value is not None else False)
            if value in (None, [], {}):
                return default_value if default_value is not None else False
            # Try to coerce other types
            return bool(value) if value else (default_value if default_value is not None else False)
        else:
            # For nullable booleans, preserve None
            if value is None:
                return None
            if isinstance(value, str):
                return bool(value.strip()) if value.strip() else None
            return bool(value) if value else None

    # If the field is nullable and value is None, that's acceptable
    if nullable and value is None:
        return value

    # For non-nullable fields with invalid values, use the default
    if not nullable:
        # Check for invalid values that should be replaced
        invalid_values = (None, "", [], {})

        if value in invalid_values or (isinstance(value, str) and not value.strip()):
            # Use the default value if provided, otherwise use type-appropriate default
            if default_value is not None:
                return default_value

            # Type-specific defaults for non-nullable fields
            if field_type is int or (hasattr(field_type, "__origin__") and field_type.__origin__ is int):
                return 0
            elif field_type is float or (hasattr(field_type, "__origin__") and field_type.__origin__ is float):
                return 0.0
            elif field_type is str or (hasattr(field_type, "__origin__") and field_type.__origin__ is str):
                return ""
            elif field_type is list or (hasattr(field_type, "__origin__") and field_type.__origin__ is list):
                return []
            elif field_type is dict or (hasattr(field_type, "__origin__") and field_type.__origin__ is dict):
                return {}

    return value


def clean_json_text_with_ai(
    driver: Driver, text: str, model_name: str = "", options: dict[str, Any] | None = None
) -> str:
    """Use LLM to fix malformed JSON strings.

    Generates a specialized prompt instructing the LLM to correct the
    provided text into valid JSON.

    Args:
        driver: Active LLM driver used to send the correction request.
        text: Malformed JSON string to be corrected.
        options: Additional options passed to the driver.

    Returns:
        A cleaned string that should contain valid JSON.
    """
    # Check if JSON is already valid - if so, return unchanged
    if options is None:
        options = {}
    try:
        json.loads(text)
        return text  # Already valid, no need for LLM correction
    except json.JSONDecodeError:
        pass  # Invalid, proceed with LLM correction

    prompt = (
        "The following text is supposed to be a single JSON object, but it is malformed. "
        "Please correct it and return only the valid JSON object. Do not add any explanations or markdown. "
        f"The text to correct is:\n\n{text}"
    )
    resp = driver.generate(prompt, options)
    raw = resp.get("text", "")
    cleaned = clean_json_text(raw)
    return cleaned


def render_output(
    driver: Driver,
    content_prompt: str,
    output_format: Literal["text", "html", "markdown"] = "text",
    model_name: str = "",
    options: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    images: list[ImageInput] | None = None,
) -> dict[str, Any]:
    """Sends a prompt to the driver and returns the raw output in the requested format.

    This function is designed for "no fluff" output, instructing the LLM to return
    only the requested content without conversational filler or markdown fences
    (unless markdown is requested).

    Args:
        driver: Adapter that implements generate(prompt, options).
        content_prompt: Main prompt content.
        output_format: Desired format ("text", "html", "markdown").
        model_name: Optional model identifier used in usage metadata.
        options: Additional options to pass to the driver.

    Returns:
        A dictionary containing:
        - text: the raw text output.
        - usage: token usage and cost information from the driver's meta object.
        - output_format: the format of the output.

    Raises:
        ValueError: If an unsupported output format is provided.
    """
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

    # Use generate_messages when system_prompt or images are provided
    user_content = _build_content_with_images(full_prompt, images)
    if system_prompt is not None or images:
        messages = [{"role": "user", "content": user_content}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        resp = driver.generate_messages(messages, options)
    else:
        resp = driver.generate(full_prompt, options)
    raw = resp.get("text", "")

    # Clean up potential markdown fences if the model disobeyed for text/html
    if output_format in ("text", "html"):
        # Simple cleanup for common fences if they appear despite instructions
        cleaned = raw.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            # Remove first line (fence + optional language) and last line (fence)
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

    _record_usage_to_ledger(model_name, resp.get("meta", {}))

    return {"text": raw, "usage": usage, "output_format": output_format}


def ask_for_json(
    driver: Driver,
    content_prompt: str,
    json_schema: dict[str, Any],
    ai_cleanup: bool = True,
    model_name: str = "",
    options: dict[str, Any] | None = None,
    output_format: Literal["json", "toon"] = "json",
    cache: bool | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
    images: list[ImageInput] | None = None,
) -> dict[str, Any]:
    """Sends a prompt to the driver and returns structured output plus usage metadata.

    This function enforces a schema-first approach by requiring a json_schema parameter
    and automatically generating instructions for the LLM to return data that matches it.

    Args:
        driver: Adapter that implements generate(prompt, options).
        content_prompt: Main prompt content (may include examples).
        json_schema: Required JSON schema dictionary defining the expected structure.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        model_name: Optional model identifier used in usage metadata.
        options: Additional options to pass to the driver.
        output_format: Response serialization format ("json" or "toon").
        cache: Override for response caching.  ``True`` forces caching on,
            ``False`` forces it off, ``None`` defers to the global setting.

    Returns:
        A dictionary containing:
        - json_string: the JSON string output.
        - json_object: the parsed JSON object.
        - usage: token usage and cost information from the driver's meta object.

    Raises:
        ValueError: If an unsupported output format is provided.
        RuntimeError: When TOON is requested but the dependency is missing.
        json.JSONDecodeError: If JSON parsing fails and ai_cleanup is False.
        ValueError: If TOON parsing fails.
    """
    if options is None:
        options = {}
    if output_format not in ("json", "toon"):
        raise ValueError(f"Unsupported output_format '{output_format}'. Use 'json' or 'toon'.")

    # --- cache lookup ---
    from .cache import get_cache, make_cache_key

    _cache = get_cache()
    use_cache = cache if cache is not None else _cache.enabled
    _force = cache is True  # explicit per-call override
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
            return cached

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

    # Use generate_messages when system_prompt or images are provided
    user_content = _build_content_with_images(full_prompt, images)
    if system_prompt is not None or images:
        messages = [{"role": "user", "content": user_content}]
        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        resp = driver.generate_messages(messages, options)
    else:
        resp = driver.generate(full_prompt, options)
    raw = resp.get("text", "")
    cleaned = clean_json_text(raw)

    _record_usage_to_ledger(model_name, resp.get("meta", {}))

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
            cleaned_fixed = clean_json_text_with_ai(driver, cleaned, model_name, options)
            try:
                json_obj = json.loads(cleaned_fixed)
                result = {
                    "json_string": cleaned_fixed,
                    "json_object": json_obj,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cost": 0.0,
                        "model_name": options.get("model", getattr(driver, "model", "")),
                        "raw_response": {},
                    },
                    "output_format": "json" if output_format != "toon" else "toon",
                }
                if output_format == "toon":
                    result["toon_string"] = toon.encode(json_obj)

                # --- cache store (ai cleanup path) ---
                if use_cache and cache_key is not None:
                    _cache.set(cache_key, result, force=_force)

                return result
            except json.JSONDecodeError:
                raise e from None
        else:
            raise e


def extract_and_jsonify(
    text: Union[str, Driver],  # Can be either text or driver for backward compatibility
    json_schema: dict[str, Any],
    *,  # Force keyword arguments for remaining params
    model_name: Union[str, dict[str, Any]] = "",  # Can be schema (old) or model name (new)
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    output_format: Literal["json", "toon"] = "json",
    options: dict[str, Any] | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
    images: list[ImageInput] | None = None,
) -> dict[str, Any]:
    """Extracts structured information using automatic driver selection based on model name.

    Args:
        text: The raw text to extract information from.
        json_schema: JSON schema dictionary defining the expected structure.
        model_name: Model identifier in format "provider/model" (e.g., "openai/gpt-4-turbo-preview").
        instruction_template: Instructional text to prepend to the content.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        output_format: Response serialization format ("json" or "toon").
        options: Additional options to pass to the driver.

    Returns:
        A dictionary containing:
        - json_string: the JSON string output.
        - json_object: the parsed JSON object.
        - usage: token usage and cost information from the driver's meta object.

    Raises:
        ValueError: If text is empty or None, or if model_name format is invalid.
        json.JSONDecodeError: If the response cannot be parsed as JSON and ai_cleanup is False.
        pytest.skip: If a ConnectionError occurs during testing (when pytest is running).
    """
    if options is None:
        options = {}
    actual_template = instruction_template
    actual_output_format = output_format
    # Handle legacy format where first argument is driver
    # Validate text input first
    if isinstance(text, Driver):
        driver = text
        actual_text = json_schema
        actual_schema = model_name
        actual_model = options.pop("model", "") or getattr(driver, "model", "")
        options.pop("model_name", None)
    else:
        # New format
        if not isinstance(text, str):
            raise ValueError("Text input must be a string")
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")
        actual_text = text
        actual_schema = json_schema
        actual_model = model_name or options.get("model", "")
        driver = options.pop("driver", None)

    # Get driver if not provided
    if driver is None:
        if not actual_model:
            raise ValueError("Model name cannot be empty")

        # First validate model format
        if "/" not in actual_model:
            raise ValueError("Invalid model string format. Expected format: 'provider/model'")

        try:
            driver = get_driver_for_model(actual_model)
        except ValueError as e:
            if "Unsupported provider" in str(e):
                raise ValueError(f"Unsupported provider in model name: {actual_model}") from e
            raise  # Re-raise any other ValueError

    # Extract model parts for other validation
    try:
        provider, model_id = actual_model.split("/", 1)
        if not provider:
            raise ValueError("Provider cannot be empty in model name")
    except ValueError:
        # If no "/" in model string, use entire string as both provider and model_id
        provider = model_id = actual_model

    opts = {**options, "model": model_id}

    content_prompt = f"{actual_template} {actual_text}"

    try:
        return ask_for_json(
            driver,
            content_prompt,
            actual_schema,
            ai_cleanup,
            model_id,
            opts,
            output_format=actual_output_format,
            json_mode=json_mode,
            system_prompt=system_prompt,
            images=images,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
        if "pytest" in sys.modules:
            import pytest

            pytest.skip(f"Connection error occurred: {e}")
        raise ConnectionError(f"Connection error occurred: {e}") from e


def manual_extract_and_jsonify(
    driver: Driver,
    text: str,
    json_schema: dict[str, Any],
    model_name: str = "",
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    output_format: Literal["json", "toon"] = "json",
    options: dict[str, Any] | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Extracts structured information using an explicitly provided driver.

    This variant is useful when you want to directly control which driver
    is used (e.g., OpenAI, Azure, Ollama, LocalHTTP) and optionally override
    the model per call.

    Args:
        driver: The LLM driver instance to use.
        text: The raw text to extract information from.
        json_schema: JSON schema dictionary defining the expected structure.
        model_name: Optional override of the model name.
        instruction_template: Instructional text to prepend to the content.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        output_format: Response serialization format ("json" or "toon").
        options: Additional options to pass to the driver.

    Returns:
        A dictionary containing:
        - json_string: the JSON string output.
        - json_object: the parsed JSON object.
        - usage: token usage and cost information from the driver's meta object.

    Raises:
        ValueError: If text is empty or None.
        json.JSONDecodeError: If the response cannot be parsed as JSON and ai_cleanup is False.
    """
    if options is None:
        options = {}
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")

    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")

    logger.info("[manual] Starting manual extraction")
    logger.debug(
        "[manual] text_length=%d model_name=%s schema_keys=%s",
        len(text),
        model_name,
        list(json_schema.keys()) if json_schema else [],
    )

    opts = dict(options)
    if model_name:
        opts["model"] = model_name

    content_prompt = f"{instruction_template} {text}"

    logger.debug("[manual] Generated prompt for extraction")

    result = ask_for_json(
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
    logger.debug("[manual] Manual extraction completed successfully")

    return result


def extract_with_model(
    model_cls: Union[type[BaseModel], str],  # Can be model class or model name string for legacy support
    text: Union[str, dict[str, Any]],  # Can be text or schema for legacy support
    model_name: Union[str, dict[str, Any], None] = None,  # Can be model name, text, or None for routing
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    output_format: Literal["json", "toon"] = "json",
    options: dict[str, Any] | None = None,
    cache: bool | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
    images: list[ImageInput] | None = None,
    routing: Union[str, RoutingConfig, None] = None,
    max_retries: int = 1,
    fallback: BaseModel | None = None,
) -> dict[str, Any]:
    """Extracts structured information into a Pydantic model instance.

    Converts the Pydantic model to its JSON schema and uses auto-resolved driver based on model_name
    to extract all fields at once, then validates and returns the model instance.

    Args:
        model_cls: The Pydantic BaseModel class to extract into.
        text: The raw text to extract information from.
        model_name: Model identifier in format "provider/model" (e.g., "openai/gpt-4-turbo-preview").
            Can be None when using routing.
        instruction_template: Instructional text to prepend to the content.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        output_format: Response serialization format ("json" or "toon").
        options: Additional options to pass to the driver.
        cache: Override for response caching.  ``True`` forces caching on,
            ``False`` forces it off, ``None`` defers to the global setting.
        routing: Smart model routing configuration. Can be:
            - A RoutingStrategy string ("cost_optimized", "quality_first", "balanced", "fast")
            - A RoutingConfig instance for full control
            - None to disable routing (requires model_name)
        max_retries: Number of extraction attempts before giving up. Defaults to 1
            (single attempt, preserving existing behavior). Set to 2+ to enable
            automatic retries on extraction/validation failures.
        fallback: Optional Pydantic model instance to return when all retries are
            exhausted. When provided, the result will include ``"fallback_used": True``
            in its usage metadata instead of raising an exception.

    Returns:
        A validated instance of the Pydantic model. When routing is used,
        the result also includes a "routing" key with routing metadata.

    Raises:
        ValueError: If text is empty or None, or if model_name format is invalid.
        ValidationError: If the extracted data doesn't match the model schema
            and no fallback is provided.
    """
    # Import routing types (avoid circular import)
    from .routing import ModelRouter, RoutingConfig, RoutingResult

    # Handle legacy format where first arg is model class
    if options is None:
        options = {}
    if isinstance(model_cls, type) and issubclass(model_cls, BaseModel):
        actual_cls = model_cls
        actual_text = text
        actual_model = model_name
    else:
        # New format where first arg is model name
        actual_model = model_cls
        actual_cls = text
        actual_text = model_name

    if not isinstance(actual_text, str) or not actual_text.strip():
        raise ValueError("Text input cannot be empty")

    # --- Smart Routing ---
    routing_result: RoutingResult | None = None
    if routing is not None:
        # Build routing config
        if isinstance(routing, str):
            routing_config = RoutingConfig(strategy=routing)
        else:
            routing_config = routing

        # Generate schema for routing analysis
        schema = actual_cls.model_json_schema()

        # Select model via router
        router = ModelRouter(routing_config)
        selected_model, routing_result = router.select_model(actual_text, schema)
        actual_model = selected_model
        logger.info("[extract] Routing selected model: %s", selected_model)
    elif actual_model is None or (isinstance(actual_model, str) and not actual_model.strip()):
        raise ValueError("Either model_name or routing must be provided")

    # --- cache lookup ---
    from .cache import get_cache, make_cache_key

    _cache = get_cache()
    use_cache = cache if cache is not None else _cache.enabled
    _force = cache is True
    cache_key: str | None = None
    if use_cache:
        schema_for_key = actual_cls.model_json_schema()
        cache_key = make_cache_key(
            prompt=f"{instruction_template} {actual_text}",
            model_name=actual_model if isinstance(actual_model, str) else "",
            schema=schema_for_key,
            options=options,
            output_format=output_format,
            pydantic_qualname=actual_cls.__qualname__,
        )
        cached = _cache.get(cache_key, force=_force)
        if cached is not None:
            cached["usage"]["cache_hit"] = True
            # Reconstruct Pydantic model instance from cached JSON
            cached["model"] = actual_cls(**cached["json_object"])
            return type(
                "ExtractResult",
                (dict,),
                {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: self["model"]},
            )(cached)

    logger.info("[extract] Starting extract_with_model")
    logger.debug(
        "[extract] model_cls=%s text_length=%d model_name=%s",
        actual_cls.__name__,
        len(actual_text),
        actual_model,
    )

    schema = actual_cls.model_json_schema()
    logger.debug("[extract] Generated JSON schema")

    last_error: Exception | None = None

    for attempt in range(max(1, max_retries)):
        try:
            result = extract_and_jsonify(
                text=actual_text,
                json_schema=schema,
                model_name=actual_model,
                instruction_template=instruction_template,
                ai_cleanup=ai_cleanup,
                output_format=output_format,
                options=options,
                json_mode=json_mode,
                system_prompt=system_prompt,
                images=images,
            )
            logger.debug("[extract] Extraction completed successfully")

            # Post-process the extracted JSON object to normalize invalid values
            json_object = result["json_object"]
            schema_properties = schema.get("properties", {})

            for field_name, field_info in actual_cls.model_fields.items():
                if field_name in json_object and field_name in schema_properties:
                    schema_properties[field_name]
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

                    # Normalize the value
                    json_object[field_name] = normalize_field_value(
                        json_object[field_name], field_info.annotation, field_def
                    )

            # Create model instance for validation
            model_instance = actual_cls(**json_object)

            # Return dictionary with all required fields and backwards compatibility
            result_dict = {
                "json_string": result["json_string"],
                "json_object": result["json_object"],
                "usage": result["usage"],
            }

            # --- Add routing metadata if routing was used ---
            if routing_result is not None:
                result_dict["routing"] = routing_result.to_dict()

            # --- cache store ---
            if use_cache and cache_key is not None:
                cached_copy = {
                    "json_string": result_dict["json_string"],
                    "json_object": result_dict["json_object"],
                    "usage": {**result_dict["usage"], "raw_response": {}},
                }
                _cache.set(cache_key, cached_copy, force=_force)

            # Add backwards compatibility property
            result_dict["model"] = model_instance

            # Return value can be used both as a dict and accessed as model directly
            return type(
                "ExtractResult",
                (dict,),
                {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: self["model"]},
            )(result_dict)

        except Exception as exc:
            last_error = exc
            if attempt < max(1, max_retries) - 1:
                logger.debug("[extract] Attempt %d failed: %s, retrying...", attempt + 1, exc)
                continue

    # All retries exhausted — use fallback or re-raise
    if fallback is not None:
        logger.info("[extract] All %d attempts failed, using fallback", max_retries)
        fallback_dict = fallback.model_dump()
        fallback_json = json.dumps(fallback_dict)
        fallback_result = {
            "json_string": fallback_json,
            "json_object": json.loads(fallback_json),
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "model_name": actual_model if isinstance(actual_model, str) else "",
                "fallback_used": True,
            },
            "model": fallback,
        }
        if routing_result is not None:
            fallback_result["routing"] = routing_result.to_dict()
        return type(
            "ExtractResult",
            (dict,),
            {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: self["model"]},
        )(fallback_result)

    raise last_error  # type: ignore[misc]


def stepwise_extract_with_model(
    model_cls: type[BaseModel],
    text: str,
    *,  # Force keyword arguments for remaining params
    model_name: str,
    instruction_template: str = "Extract the {field_name} from the following text:",
    ai_cleanup: bool = True,
    fields: list[str] | None = None,
    field_definitions: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    json_mode: Literal["auto", "on", "off"] = "auto",
    system_prompt: str | None = None,
    share_context: bool = False,
) -> dict[str, Union[str, dict[str, Any]]]:
    """Extracts structured information into a Pydantic model by processing each field individually.

    For each field in the model, makes a separate LLM call to extract that specific field,
    then combines the results and validates the complete model instance. When extraction
    or conversion fails for individual fields, uses appropriate default values to ensure
    partial results can still be returned.

    Args:
        model_cls: The Pydantic BaseModel class to extract into.
        text: The raw text to extract information from.
        model_name: Model identifier in format "provider/model" (e.g., "openai/gpt-4-turbo-preview").
        instruction_template: Template for instructional text, should include {field_name} placeholder.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        fields: Optional list of field names to extract. If None, extracts all fields.
        field_definitions: Optional field definitions dict for enhanced default handling.
                          If None, automatically uses the global field registry.
        options: Additional options to pass to the driver.

    Returns:
        A dictionary containing:
        - model: A validated instance of the Pydantic model (with defaults for failed extractions).
        - usage: Accumulated token usage and cost information across all field extractions.
        - field_results: Dict tracking success/failure status per field.

    Raises:
        ValueError: If text is empty or None, or if model_name format is invalid.
        KeyError: If a requested field doesn't exist in the model.

    Note:
        This function now gracefully handles extraction failures by falling back to default
        values rather than failing completely. Individual field errors are logged and
        tracked in the usage information.
    """
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")

    # When share_context=True, delegate to Conversation-based extraction
    if share_context:
        from .conversation import Conversation

        conv = Conversation(model_name=model_name, system_prompt=system_prompt, options=options)
        return conv._stepwise_extract(
            model_cls=model_cls,
            text=text,
            instruction_template=instruction_template,
            ai_cleanup=ai_cleanup,
            fields=fields,
            field_definitions=field_definitions,
            json_mode=json_mode,
        )

    logger.info("[stepwise] Starting stepwise extraction")
    logger.debug(
        "[stepwise] model_cls=%s text_length=%d fields=%s",
        model_cls.__name__,
        len(text),
        fields,
    )

    # Auto-use global field registry if no field_definitions provided
    if field_definitions is None:
        field_definitions = get_registry_snapshot()
        logger.debug("[stepwise] Using global field registry")

    data = {}
    validation_errors = []
    field_results = {}  # Track success/failure per field
    options = options or {}

    # Initialize usage accumulator
    accumulated_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "model_name": model_name,  # Use provided model_name directly
        "field_usages": {},
    }

    # Get valid field names from the model
    valid_fields = set(model_cls.model_fields.keys())

    # If fields specified, validate they exist
    if fields is not None:
        invalid_fields = set(fields) - valid_fields
        if invalid_fields:
            raise KeyError(f"Fields not found in model: {', '.join(invalid_fields)}")
        field_items = [(name, model_cls.model_fields[name]) for name in fields]
    else:
        field_items = model_cls.model_fields.items()

    for field_name, field_info in field_items:
        logger.debug("[stepwise] Extracting field: %s", field_name)

        # Create field schema that expects a direct value rather than a dict
        field_schema = {
            "value": {
                "type": "integer" if field_info.annotation is int else "string",
                "description": field_info.description or f"Value for {field_name}",
            }
        }

        try:
            result = extract_and_jsonify(
                text=text,
                json_schema=field_schema,
                model_name=model_name,
                instruction_template=instruction_template.format(field_name=field_name),
                ai_cleanup=ai_cleanup,
                options=options,
                json_mode=json_mode,
                system_prompt=system_prompt,
            )

            logger.debug("[stepwise] Raw extraction result for %s", field_name)

            # Accumulate usage data from this field extraction
            field_usage = result.get("usage", {})
            accumulated_usage["prompt_tokens"] += field_usage.get("prompt_tokens", 0)
            accumulated_usage["completion_tokens"] += field_usage.get("completion_tokens", 0)
            accumulated_usage["total_tokens"] += field_usage.get("total_tokens", 0)
            accumulated_usage["cost"] += field_usage.get("cost", 0.0)
            accumulated_usage["field_usages"][field_name] = field_usage

            # Extract the raw value from the response - handle both dict and direct value formats
            extracted_value = result["json_object"]["value"]
            logger.debug("[stepwise] Raw extracted value for %s: %s", field_name, extracted_value)

            if isinstance(extracted_value, dict) and "value" in extracted_value:
                raw_value = extracted_value["value"]
                logger.debug("[stepwise] Extracted inner value from dict for %s", field_name)
            else:
                raw_value = extracted_value
                logger.debug("[stepwise] Using direct value for %s", field_name)

            # Post-process the raw value to normalize invalid values for non-nullable fields
            field_def = {}
            if field_definitions and field_name in field_definitions:
                field_def = field_definitions[field_name] if isinstance(field_definitions[field_name], dict) else {}

            # Determine nullable status and default value
            nullable = field_def.get("nullable", True)
            default_value = field_def.get("default")
            if (
                default_value is None
                and hasattr(field_info, "default")
                and field_info.default is not ...
                and str(field_info.default) != "PydanticUndefined"
            ):
                default_value = field_info.default

            # Create field_def for normalize_field_value
            normalize_def = {"nullable": nullable, "default": default_value}

            # Normalize the raw value before conversion
            raw_value = normalize_field_value(raw_value, field_info.annotation, normalize_def)
            logger.debug("[stepwise] Normalized value for %s: %s", field_name, raw_value)

            # Convert value using tools.convert_value with logging
            try:
                converted_value = convert_value(raw_value, field_info.annotation, allow_shorthand=True)
                data[field_name] = converted_value
                field_results[field_name] = {"status": "success", "used_default": False}

                logger.debug("[stepwise] Successfully converted %s", field_name)

            except ValueError as e:
                error_msg = f"Type conversion failed for {field_name}: {e!s}"

                # Check if field has a default value (either explicit or from field_definitions)
                has_default = False
                if field_definitions and field_name in field_definitions:
                    field_def = field_definitions[field_name]
                    if isinstance(field_def, dict) and "default" in field_def:
                        has_default = True

                if not has_default and hasattr(field_info, "default"):
                    default_val = field_info.default
                    # Field has default if it's not PydanticUndefined or Ellipsis
                    if default_val is not ... and str(default_val) != "PydanticUndefined":
                        has_default = True

                # Only add to validation_errors if field is required (no default)
                if not has_default:
                    validation_errors.append(error_msg)

                # Use default value (type-appropriate if no explicit default)
                default_value = get_field_default(field_name, field_info, field_definitions)
                data[field_name] = default_value
                field_results[field_name] = {"status": "conversion_failed", "error": error_msg, "used_default": True}

                logger.error("[stepwise] %s", error_msg)
                logger.info("[stepwise] Using default value for %s: %s", field_name, default_value)

        except Exception as e:
            error_msg = f"Extraction failed for {field_name}: {e!s}"

            # Check if field has a default value (either explicit or from field_definitions)
            has_default = False
            if field_definitions and field_name in field_definitions:
                field_def = field_definitions[field_name]
                if isinstance(field_def, dict) and "default" in field_def:
                    has_default = True

            if not has_default and hasattr(field_info, "default"):
                default_val = field_info.default
                # Field has default if it's not PydanticUndefined or Ellipsis
                if default_val is not ... and str(default_val) != "PydanticUndefined":
                    has_default = True

            # Only add to validation_errors if field is required (no default)
            if not has_default:
                validation_errors.append(error_msg)

            # Use default value (type-appropriate if no explicit default)
            default_value = get_field_default(field_name, field_info, field_definitions)
            data[field_name] = default_value
            field_results[field_name] = {"status": "extraction_failed", "error": error_msg, "used_default": True}

            logger.error("[stepwise] %s", error_msg)
            logger.info("[stepwise] Using default value for %s: %s", field_name, default_value)

            # Store error details in field_usages
            accumulated_usage["field_usages"][field_name] = {
                "error": str(e),
                "status": "failed",
                "used_default": True,
                "default_value": default_value,
            }

    if validation_errors:
        logger.warning("[stepwise] Found %d validation errors", len(validation_errors))
        for error in validation_errors:
            logger.error("[stepwise] %s", error)

    # If there are validation errors, include them in the result
    if validation_errors:
        accumulated_usage["validation_errors"] = validation_errors

    try:
        # Create model instance with collected data
        # Create model instance with collected data
        model_instance = model_cls(**data)
        model_dict = model_instance.model_dump()

        # Enhanced DateTimeEncoder to handle both datetime and date objects
        class ExtendedJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                if isinstance(obj, Decimal):
                    return str(obj)
                return super().default(obj)

        # Use enhanced encoder for JSON serialization
        json_string = json.dumps(model_dict, cls=ExtendedJSONEncoder)

        # Also modify return value to use ExtendedJSONEncoder
        if "json_string" in result:
            result["json_string"] = json.dumps(result["json_object"], cls=ExtendedJSONEncoder)

        # Define ExtendedJSONEncoder for handling special types
        class ExtendedJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                if isinstance(obj, Decimal):
                    return str(obj)
                return super().default(obj)

        # Create json string with custom encoder
        json_string = json.dumps(model_dict, cls=ExtendedJSONEncoder)

        # Create result matching extract_with_model format
        result = {
            "json_string": json_string,
            "json_object": json.loads(json_string),  # Re-parse to ensure all values are JSON serializable
            "usage": accumulated_usage,
            "field_results": field_results,
        }

        # Add model instance as property and make callable
        result["model"] = model_instance
        return type(
            "ExtractResult",
            (dict,),
            {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: self["model"]},
        )(result)
    except Exception as e:
        error_msg = f"Model validation error: {e!s}"
        # Add validation error to accumulated usage
        if "validation_errors" not in accumulated_usage:
            accumulated_usage["validation_errors"] = []
        accumulated_usage["validation_errors"].append(error_msg)

        logger.error("[stepwise] %s", error_msg)

        # Create error result with partial data
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
            {
                "__getattr__": lambda self, key: self.get(key),
                "__call__": lambda self: None,  # Return None when called if validation failed
            },
        )(error_result)


def _json_to_toon(data: Union[list[dict[str, Any]], dict[str, Any]], data_key: str | None = None) -> str:
    """Convert JSON array or dict containing array to TOON format.

    Args:
        data: List of dicts (uniform array) or dict containing array under a key
        data_key: If data is a dict, the key containing the array

    Returns:
        TOON formatted string

    Raises:
        ValueError: If TOON conversion fails or data format is invalid
        RuntimeError: If python-toon is not installed
    """
    if toon is None:
        raise RuntimeError(
            "TOON conversion requested but 'python-toon' is not installed. Install it with 'pip install python-toon'."
        )

    # Handle different data formats
    if isinstance(data, list):
        array_data = data
    elif isinstance(data, dict):
        if data_key:
            if data_key not in data:
                raise ValueError(f"Key '{data_key}' not found in data")
            array_data = data[data_key]
        else:
            # Try to find the first array value in the dict
            array_data = None
            for _key, value in data.items():
                if isinstance(value, list) and value:
                    array_data = value
                    break
            if array_data is None:
                raise ValueError("No array found in data. Specify data_key or provide a list directly.")
    else:
        raise ValueError("Data must be a list of dicts or a dict containing an array")

    if not isinstance(array_data, list):
        raise ValueError("Array data must be a list")

    if not array_data:
        raise ValueError("Array data cannot be empty")

    # Validate that all items in array are dicts (uniform structure)
    if not all(isinstance(item, dict) for item in array_data):
        raise ValueError("All items in array must be dictionaries for TOON conversion")

    try:
        return toon.encode(array_data)
    except Exception as e:
        raise ValueError(f"Failed to convert data to TOON format: {e}") from e


def _dataframe_to_toon(df) -> str:
    """Convert Pandas DataFrame to TOON format.

    Args:
        df: Pandas DataFrame to convert

    Returns:
        TOON formatted string

    Raises:
        ValueError: If DataFrame conversion fails
        RuntimeError: If pandas or python-toon is not installed
    """
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError(
            "Pandas DataFrame conversion requested but 'pandas' is not installed. "
            "Install it with 'pip install pandas' or 'pip install prompture[pandas]'."
        ) from None

    if toon is None:
        raise RuntimeError(
            "TOON conversion requested but 'python-toon' is not installed. Install it with 'pip install python-toon'."
        )

    dataframe_type = getattr(pd, "DataFrame", None)
    if isinstance(dataframe_type, type):
        if not isinstance(df, dataframe_type):
            raise ValueError("Input must be a pandas DataFrame")
    else:
        # Duck-type fallback for tests that provide a lightweight mock
        if not hasattr(df, "to_dict") or not hasattr(df, "empty"):
            raise ValueError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("DataFrame cannot be empty")

    try:
        # Convert DataFrame to list of dicts
        data = df.to_dict("records")
        return toon.encode(data)
    except Exception as e:
        raise ValueError(f"Failed to convert DataFrame to TOON format: {e}") from e


def _calculate_token_savings(json_text: str, toon_text: str) -> dict[str, Any]:
    """Calculate estimated token savings between JSON and TOON formats.

    This is a rough estimation based on character count ratios.
    Actual token counts may vary by model and tokenizer.

    Args:
        json_text: JSON formatted text
        toon_text: TOON formatted text

    Returns:
        Dict containing savings statistics
    """
    json_chars = len(json_text)
    toon_chars = len(toon_text)

    # Rough estimation: 4 characters ≈ 1 token (varies by model)
    json_tokens_est = json_chars // 4
    toon_tokens_est = toon_chars // 4

    savings_chars = json_chars - toon_chars
    savings_tokens_est = json_tokens_est - toon_tokens_est

    percentage_saved = (savings_chars / json_chars * 100) if json_chars > 0 else 0

    return {
        "json_characters": json_chars,
        "toon_characters": toon_chars,
        "saved_characters": savings_chars,
        "estimated_json_tokens": json_tokens_est,
        "estimated_toon_tokens": toon_tokens_est,
        "estimated_saved_tokens": savings_tokens_est,
        "percentage_saved": round(percentage_saved, 1),
    }


def extract_from_data(
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
    """Extract information from structured data by converting to TOON format for token efficiency.

    This function takes JSON array data, converts it to TOON format to reduce tokens,
    sends it to the LLM with a question, and returns the JSON response.

    Args:
        data: List of dicts (uniform array) or dict containing array under a key
        question: The question to ask about the data
        json_schema: Expected JSON schema for the response
        model_name: Model identifier in format "provider/model" (e.g., "openai/gpt-4")
        data_key: If data is a dict, the key containing the array (e.g., "products")
        instruction_template: Template with {question} placeholder
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails
        options: Additional options to pass to the driver

    Returns:
        Dict containing:
        - json_object: The parsed JSON response
        - json_string: The JSON string response
        - usage: Token usage and cost information (includes token_savings)
        - toon_data: The TOON formatted input data
        - token_savings: Statistics about token savings vs JSON input

    Raises:
        ValueError: If data format is invalid or conversion fails
        RuntimeError: If required dependencies are missing

    Example:
        >>> products = [
        ...     {"id": 1, "name": "Laptop", "price": 999.99, "category": "electronics"},
        ...     {"id": 2, "name": "Book", "price": 19.99, "category": "books"}
        ... ]
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "average_price": {"type": "number"},
        ...         "total_items": {"type": "integer"}
        ...     }
        ... }
        >>> result = extract_from_data(
        ...     data=products,
        ...     question="What is the average price and total number of items?",
        ...     json_schema=schema,
        ...     model_name="openai/gpt-4"
        ... )
        >>> print(result["json_object"])
        {'average_price': 509.99, 'total_items': 2}
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not json_schema:
        raise ValueError("JSON schema cannot be empty")

    if options is None:
        options = {}

    # Convert data to TOON format
    toon_data = _json_to_toon(data, data_key)

    # Calculate token savings (for comparison with JSON)
    json_data = json.dumps(data if isinstance(data, list) else data.get(data_key, data), indent=2)
    token_savings = _calculate_token_savings(json_data, toon_data)

    # Build the prompt with TOON data
    content_prompt = instruction_template.format(question=question)
    full_prompt = f"{content_prompt}\n\nData (in TOON format):\n{toon_data}"

    # Call the LLM
    result = ask_for_json(
        driver=get_driver_for_model(model_name),
        content_prompt=full_prompt,
        json_schema=json_schema,
        ai_cleanup=ai_cleanup,
        model_name=model_name.split("/")[-1] if "/" in model_name else model_name,
        options=options,
        output_format="json",  # Always return JSON, not TOON
        system_prompt=system_prompt,
    )

    # Add our additional data to the result
    result["toon_data"] = toon_data
    result["token_savings"] = token_savings

    return result


def extract_from_pandas(
    df,  # pandas.DataFrame - optional import
    question: str,
    json_schema: dict[str, Any],
    *,
    model_name: str,
    instruction_template: str = "Analyze the following data and answer: {question}",
    ai_cleanup: bool = True,
    options: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Extract information from Pandas DataFrame by converting to TOON format for token efficiency.

    This function takes a Pandas DataFrame, converts it to TOON format to reduce tokens,
    sends it to the LLM with a question, and returns the JSON response.

    Args:
        df: Pandas DataFrame to analyze
        question: The question to ask about the data
        json_schema: Expected JSON schema for the response
        model_name: Model identifier in format "provider/model" (e.g., "openai/gpt-4")
        instruction_template: Template with {question} placeholder
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails
        options: Additional options to pass to the driver

    Returns:
        Dict containing:
        - json_object: The parsed JSON response
        - json_string: The JSON string response
        - usage: Token usage and cost information (includes token_savings)
        - toon_data: The TOON formatted input data
        - token_savings: Statistics about token savings vs JSON input
        - dataframe_info: Basic info about the original DataFrame

    Raises:
        ValueError: If DataFrame is invalid or conversion fails
        RuntimeError: If required dependencies are missing

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame([
        ...     {"id": 1, "name": "Laptop", "price": 999.99, "category": "electronics"},
        ...     {"id": 2, "name": "Book", "price": 19.99, "category": "books"}
        ... ])
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "highest_priced_item": {"type": "string"},
        ...         "price_range": {"type": "number"}
        ...     }
        ... }
        >>> result = extract_from_pandas(
        ...     df=df,
        ...     question="What is the highest priced item and price range?",
        ...     json_schema=schema,
        ...     model_name="openai/gpt-4"
        ... )
        >>> print(result["json_object"])
        {'highest_priced_item': 'Laptop', 'price_range': 980.0}
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not json_schema:
        raise ValueError("JSON schema cannot be empty")

    if options is None:
        options = {}

    # Convert DataFrame to TOON format
    toon_data = _dataframe_to_toon(df)

    # Calculate token savings (for comparison with JSON)
    json_data = df.to_json(indent=2, orient="records")
    token_savings = _calculate_token_savings(json_data, toon_data)

    # Get basic DataFrame info
    dataframe_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    # Build the prompt with TOON data
    content_prompt = instruction_template.format(question=question)
    full_prompt = f"{content_prompt}\n\nData (in TOON format):\n{toon_data}"

    # Call the LLM
    result = ask_for_json(
        driver=get_driver_for_model(model_name),
        content_prompt=full_prompt,
        json_schema=json_schema,
        ai_cleanup=ai_cleanup,
        model_name=model_name.split("/")[-1] if "/" in model_name else model_name,
        options=options,
        output_format="json",  # Always return JSON, not TOON
        system_prompt=system_prompt,
    )

    # Add our additional data to the result
    result["toon_data"] = toon_data
    result["token_savings"] = token_savings
    result["dataframe_info"] = dataframe_info

    return result
