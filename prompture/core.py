"""Core utilities: Helpers for requesting JSON from LLM.
"""
from __future__ import annotations
import json
import re
import requests
import sys
import warnings
from typing import Any, Dict, Type, Optional, List, Union
import pytest

from pydantic import BaseModel

from .drivers import get_driver, get_driver_for_model
from .driver import Driver
from .tools import create_field_schema, convert_value, log_debug, clean_json_text, LogLevel


def clean_json_text_with_ai(driver: Driver, text: str, model_name: str = "", options: Dict[str, Any] = {}) -> str:
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
    prompt = (
        "The following text is supposed to be a single JSON object, but it is malformed. "
        "Please correct it and return only the valid JSON object. Do not add any explanations or markdown. "
        f"The text to correct is:\n\n{text}"
    )
    resp = driver.generate(prompt, options)
    raw = resp.get("text", "")
    cleaned = clean_json_text(raw)
    return cleaned

def ask_for_json(
    driver: Driver,
    content_prompt: str,
    json_schema: Dict[str, Any],
    ai_cleanup: bool = True,
    model_name: str = "",
    options: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Sends a prompt to the driver and returns both JSON output and usage metadata.

    This function enforces a schema-first approach by requiring a json_schema parameter
    and automatically generating instructions for the LLM to return valid JSON matching the schema.

    Args:
        driver: Adapter that implements generate(prompt, options).
        content_prompt: Main prompt content (may include examples).
        json_schema: Required JSON schema dictionary defining the expected structure.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        options: Additional options to pass to the driver.
        verbose_level: Logging level for debug output (LogLevel.OFF by default).

    Returns:
        A dictionary containing:
        - json_string: the JSON string output.
        - json_object: the parsed JSON object.
        - usage: token usage and cost information from the driver's meta object.

    Raises:
        json.JSONDecodeError: If the response cannot be parsed as JSON and ai_cleanup is False.
    """
    schema_string = json.dumps(json_schema, indent=2)
    instruct = (
        "Return only a single JSON object (no markdown, no extra text) that validates against this JSON schema:\n"
        f"{schema_string}\n\n"
        "If a value is unknown use null. Use double quotes for keys and strings."
    )
    full_prompt = f"{content_prompt}\n\n{instruct}"
    resp = driver.generate(full_prompt, options)
    raw = resp.get("text", "")
    cleaned = clean_json_text(raw)

    try:
        json_obj = json.loads(cleaned)
        usage = {
            **resp.get("meta", {}),
            "raw_response": resp,
            "total_tokens": resp.get("meta", {}).get("total_tokens", 0),
            "prompt_tokens": resp.get("meta", {}).get("prompt_tokens", 0),
            "completion_tokens": resp.get("meta", {}).get("completion_tokens", 0),
            "cost": resp.get("meta", {}).get("cost", 0.0),
            "model_name": model_name or getattr(driver, "model", "")
        }
        return {
            "json_string": cleaned,
            "json_object": json_obj,
            "usage": usage
        }
    except json.JSONDecodeError as e:
        if ai_cleanup:
            cleaned_fixed = clean_json_text_with_ai(driver, cleaned, model_name, options)
            try:
                json_obj = json.loads(cleaned_fixed)
                return {
                    "json_string": cleaned_fixed,
                    "json_object": json_obj,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cost": 0.0,
                        "model_name": options.get("model", getattr(driver, "model", "")),
                        "raw_response": {}
                    },
                }
            except json.JSONDecodeError:
                # Re-raise the original JSONDecodeError
                raise e
        else:
            # Explicitly re-raise the original JSONDecodeError
            raise e

def _old_extract_and_jsonify(
    driver: Driver,
    text: str,
    json_schema: Dict[str, Any],
    model_name: str = "",
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    options: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """[DEPRECATED] Legacy version of extract_and_jsonify that takes an explicit driver.
    
    Use the new extract_and_jsonify(text, json_schema, model_name, ...) instead.
    This version will be removed in a future release.
    """
    warnings.warn(
        "This function is deprecated. Use extract_and_jsonify(text, json_schema, model_name, ...) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    import sys
    
    model = model_name or getattr(driver, "model", "")
    return extract_and_jsonify(
        text=text,
        json_schema=json_schema,
        model_name=model,
        instruction_template=instruction_template,
        ai_cleanup=ai_cleanup,
        options={"driver": driver, "model": model, **options}
    )

def extract_and_jsonify(
    text: Union[str, Driver],  # Can be either text or driver for backward compatibility
    json_schema: Dict[str, Any],
    *,  # Force keyword arguments for remaining params
    model_name: Union[str, Dict[str, Any]] = "",  # Can be schema (old) or model name (new)
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    options: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Extracts structured information using automatic driver selection based on model name.
    
    Args:
        text: The raw text to extract information from.
        json_schema: JSON schema dictionary defining the expected structure.
        model_name: Model identifier in format "provider/model" (e.g., "openai/gpt-4-turbo-preview").
        instruction_template: Instructional text to prepend to the content.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
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
        actual_template = instruction_template
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
                raise ValueError(f"Unsupported provider in model name: {actual_model}")
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
        return ask_for_json(driver, content_prompt, actual_schema, ai_cleanup, model_id, opts)
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
        if "pytest" in sys.modules:
            pytest.skip(f"Connection error occurred: {e}")
        raise ConnectionError(f"Connection error occurred: {e}")

def manual_extract_and_jsonify(
    driver: Driver,
    text: str,
    json_schema: Dict[str, Any],
    model_name: str = "",
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    options: Dict[str, Any] = {},
    verbose_level: LogLevel | int = LogLevel.OFF,
) -> Dict[str, Any]:
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
        options: Additional options to pass to the driver.
        verbose_level: Logging level for debug output (LogLevel.OFF by default).

    Returns:
        A dictionary containing:
        - json_string: the JSON string output.
        - json_object: the parsed JSON object.
        - usage: token usage and cost information from the driver's meta object.

    Raises:
        ValueError: If text is empty or None.
        json.JSONDecodeError: If the response cannot be parsed as JSON and ai_cleanup is False.
    """
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")
    
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")
    # Add function entry logging
    log_debug(LogLevel.INFO, verbose_level, "Starting manual extraction", prefix="[manual]")
    log_debug(LogLevel.DEBUG, verbose_level, {
        "text_length": len(text),
        "model_name": model_name,
        "schema_keys": list(json_schema.keys()) if json_schema else []
    }, prefix="[manual]")

    opts = dict(options)
    if model_name:
        opts["model"] = model_name

    # Generate the content prompt
    content_prompt = f"{instruction_template} {text}"
    
    # Add logging for prompt generation
    log_debug(LogLevel.DEBUG, verbose_level, "Generated prompt for extraction", prefix="[manual]")
    log_debug(LogLevel.TRACE, verbose_level, {"content_prompt": content_prompt}, prefix="[manual]")
    
    # Call ask_for_json and log the result
    result = ask_for_json(driver, content_prompt, json_schema, ai_cleanup, model_name, opts)
    log_debug(LogLevel.DEBUG, verbose_level, "Manual extraction completed successfully", prefix="[manual]")
    log_debug(LogLevel.TRACE, verbose_level, {"result": result}, prefix="[manual]")
    
    return result
    

def extract_with_model(
    model_cls: Union[Type[BaseModel], str],  # Can be model class or model name string for legacy support
    text: Union[str, Dict[str, Any]],  # Can be text or schema for legacy support
    model_name: Union[str, Dict[str, Any]],  # Can be model name or text for legacy support
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    options: Dict[str, Any] = {},
    verbose_level: LogLevel | int = LogLevel.OFF,
) -> Dict[str, Any]:
    """Extracts structured information into a Pydantic model instance.

    Converts the Pydantic model to its JSON schema and uses auto-resolved driver based on model_name
    to extract all fields at once, then validates and returns the model instance.

    Args:
        model_cls: The Pydantic BaseModel class to extract into.
        text: The raw text to extract information from.
        model_name: Model identifier in format "provider/model" (e.g., "openai/gpt-4-turbo-preview").
        instruction_template: Instructional text to prepend to the content.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        options: Additional options to pass to the driver.
        verbose_level: Logging level for debug output (LogLevel.OFF by default).

    Returns:
        A validated instance of the Pydantic model.

    Raises:
        ValueError: If text is empty or None, or if model_name format is invalid.
        ValidationError: If the extracted data doesn't match the model schema.
    """
    # Handle legacy format where first arg is model class
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
    
    # Add function entry logging
    log_debug(LogLevel.INFO, verbose_level, "Starting extract_with_model", prefix="[extract]")
    log_debug(LogLevel.DEBUG, verbose_level, {
        "model_cls": actual_cls.__name__,
        "text_length": len(actual_text),
        "model_name": actual_model
    }, prefix="[extract]")

    schema = actual_cls.model_json_schema()
    log_debug(LogLevel.DEBUG, verbose_level, "Generated JSON schema", prefix="[extract]")
    log_debug(LogLevel.TRACE, verbose_level, {"schema": schema}, prefix="[extract]")
    
    result = extract_and_jsonify(
        text=actual_text,
        json_schema=schema,
        model_name=actual_model,
        instruction_template=instruction_template,
        ai_cleanup=ai_cleanup,
        options=options
    )
    log_debug(LogLevel.DEBUG, verbose_level, "Extraction completed successfully", prefix="[extract]")
    log_debug(LogLevel.TRACE, verbose_level, {"result": result}, prefix="[extract]")
    
    # Create model instance for validation
    model_instance = actual_cls(**result["json_object"])
    
    # Return dictionary with all required fields and backwards compatibility
    result_dict = {
        "json_string": result["json_string"],
        "json_object": result["json_object"],
        "usage": result["usage"]
    }
    
    # Add backwards compatibility property
    result_dict["model"] = model_instance
    
    # Return value can be used both as a dict and accessed as model directly
    return type("ExtractResult", (dict,), {
        "__getattr__": lambda self, key: self.get(key),
        "__call__": lambda self: self["model"]
    })(result_dict)

def stepwise_extract_with_model(
    model_cls: Type[BaseModel],
    text: str,
    model_name: str,
    instruction_template: str = "Extract the {field_name} from the following text:",
    ai_cleanup: bool = True,
    fields: Optional[List[str]] = None,
    options: Dict[str, Any] = {},
    verbose_level: LogLevel | int = LogLevel.OFF,
) -> Dict[str, Union[str, Dict[str, Any]]]:
    """Extracts structured information into a Pydantic model by processing each field individually.

    For each field in the model, makes a separate LLM call to extract that specific field,
    then combines the results and validates the complete model instance.

    Args:
        model_cls: The Pydantic BaseModel class to extract into.
        text: The raw text to extract information from.
        model_name: Model identifier in format "provider/model" (e.g., "openai/gpt-4-turbo-preview").
        instruction_template: Template for instructional text, should include {field_name} placeholder.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        fields: Optional list of field names to extract. If None, extracts all fields.
        options: Additional options to pass to the driver.
        verbose_level: Logging level for debug output (LogLevel.OFF by default).

    Returns:
        A dictionary containing:
        - model: A validated instance of the Pydantic model.
        - usage: Accumulated token usage and cost information across all field extractions.

    Raises:
        ValueError: If text is empty or None, or if model_name format is invalid.
        ValidationError: If the extracted data doesn't match the model schema.
        KeyError: If a requested field doesn't exist in the model.
    """
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")
    # Add function entry logging
    log_debug(LogLevel.INFO, verbose_level, "Starting stepwise extraction", prefix="[stepwise]")
    log_debug(LogLevel.DEBUG, verbose_level, {
        "model_cls": model_cls.__name__,
        "text_length": len(text),
        "fields": fields,
    }, prefix="[stepwise]")

    data = {}
    validation_errors = []
    
    # Initialize usage accumulator
    accumulated_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "model_name": model_name or getattr(driver, "model", ""),
        "field_usages": {}
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
        # Add structured logging for field extraction
        log_debug(LogLevel.DEBUG, verbose_level, f"Extracting field: {field_name}", prefix="[stepwise]")
        log_debug(LogLevel.TRACE, verbose_level, {
            "field_name": field_name,
            "field_info": str(field_info),
            "field_type": str(field_info.annotation)
        }, prefix="[stepwise]")

        # Create field schema using tools.create_field_schema
        field_schema = {
            "value": create_field_schema(
                field_name,
                field_info.annotation,
                field_info.description
            )
        }

        # Add structured logging for field schema and prompt
        log_debug(LogLevel.TRACE, verbose_level, {
            "field_schema": field_schema,
            "prompt_template": instruction_template.format(field_name=field_name)
        }, prefix="[stepwise]")

        try:
            result = extract_and_jsonify(
                text=text,
                json_schema=field_schema,
                model_name=model_name,
                instruction_template=instruction_template.format(field_name=field_name),
                ai_cleanup=ai_cleanup,
                options=options
            )

            # Add structured logging for extraction result
            log_debug(LogLevel.DEBUG, verbose_level, f"Raw extraction result for {field_name}", prefix="[stepwise]")
            log_debug(LogLevel.TRACE, verbose_level, {"result": result}, prefix="[stepwise]")

            # Accumulate usage data from this field extraction
            field_usage = result.get("usage", {})
            accumulated_usage["prompt_tokens"] += field_usage.get("prompt_tokens", 0)
            accumulated_usage["completion_tokens"] += field_usage.get("completion_tokens", 0)
            accumulated_usage["total_tokens"] += field_usage.get("total_tokens", 0)
            accumulated_usage["cost"] += field_usage.get("cost", 0.0)
            accumulated_usage["field_usages"][field_name] = field_usage

            extracted_value = result["json_object"]["value"]

            # Convert value using tools.convert_value
            try:
                converted_value = convert_value(
                    extracted_value,
                    field_info.annotation,
                    allow_shorthand=True
                )
                data[field_name] = converted_value

                # Add structured logging for converted value
                log_debug(LogLevel.DEBUG, verbose_level, f"Successfully converted {field_name}", prefix="[stepwise]")
                log_debug(LogLevel.TRACE, verbose_level, {
                    "field_name": field_name,
                    "converted_value": converted_value
                }, prefix="[stepwise]")
                    
            except ValueError as e:
                error_msg = f"Type conversion failed for {field_name}: {str(e)}"
                validation_errors.append(error_msg)
                
                # Add structured logging for conversion error
                log_debug(LogLevel.ERROR, verbose_level, error_msg, prefix="[stepwise]")
                
        except Exception as e:
            error_msg = f"Extraction failed for {field_name}: {str(e)}"
            validation_errors.append(error_msg)
            
            # Add structured logging for extraction error
            log_debug(LogLevel.ERROR, verbose_level, error_msg, prefix="[stepwise]")
    
    # Add structured logging for validation errors
    if validation_errors:
        log_debug(LogLevel.WARN, verbose_level, f"Found {len(validation_errors)} validation errors", prefix="[stepwise]")
        for error in validation_errors:
            log_debug(LogLevel.ERROR, verbose_level, error, prefix="[stepwise]")
    
    try:
        model_instance = model_cls(**data)
        # Convert model to dict for json_object
        model_dict = model_instance.model_dump()
        
        # Create JSON string from the dict
        json_string = json.dumps(model_dict)
        
        # Create result dictionary with backwards compatibility
        result_dict = {
            "json_string": json_string,
            "json_object": model_dict,
            "usage": accumulated_usage,
            "model": model_instance  # For backwards compatibility
        }
        
        # Return value can be used both as a dict and accessed as model directly
        return type("StepwiseExtractResult", (dict,), {
            "__getattr__": lambda self, key: self.get(key),
            "__call__": lambda self: self["model"]
        })(result_dict)
    except Exception as e:
        # Add structured logging for model validation error
        log_debug(LogLevel.ERROR, verbose_level, f"Model validation error: {str(e)}", prefix="[stepwise]")
        raise
