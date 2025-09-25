"""Core utilities: Helpers for requesting JSON from LLM.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, Type, Optional, List

from pydantic import BaseModel

from .drivers import get_driver
from .driver import Driver
from .tools import create_field_schema, convert_value, log_debug, clean_json_text


def clean_json_text_with_ai(driver: Driver, text: str, options: Dict[str, Any] = {}) -> str:
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
    return clean_json_text(raw)

def ask_for_json(
    driver: Driver,
    content_prompt: str,
    json_schema: Dict[str, Any],
    ai_cleanup: bool = True,
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
        return {
            "json_string": cleaned,
            "json_object": json_obj,
            "usage": resp.get("meta", {}),
        }
    except json.JSONDecodeError:
        if ai_cleanup:
            cleaned_fixed = clean_json_text_with_ai(driver, cleaned, options)
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
                    },
                }
            except json.JSONDecodeError:
                raise
        else:
            raise

def extract_and_jsonify(
    text: str,
    json_schema: Dict[str, Any],
    model_name: str = "",
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    options: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Extracts structured information using the default driver from AI_PROVIDER.

    Automatically initializes the driver based on environment configuration
    and constructs a schema-constrained prompt.

    Args:
        text: The raw text to extract information from.
        json_schema: JSON schema dictionary defining the expected structure.
        instruction_template: Instructional text to prepend to the content.
        model_name: Optional override of the model name.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
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
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")

    driver = get_driver()

    opts = dict(options)
    if model_name:
        opts["model"] = model_name

    content_prompt = f"{instruction_template} {text}"
    return ask_for_json(driver, content_prompt, json_schema, ai_cleanup, opts)

def extract_with_model(
    model_cls: Type[BaseModel],
    text: str,
    driver: Optional[Driver] = None,
    model_name: str = "",
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    options: Dict[str, Any] = {},
) -> BaseModel:
    """Extracts structured information into a Pydantic model instance.

    Converts the Pydantic model to its JSON schema and uses the driver to extract
    all fields at once, then validates and returns the model instance.

    Args:
        model_cls: The Pydantic BaseModel class to extract into.
        text: The raw text to extract information from.
        driver: Optional LLM driver instance. If None, uses get_driver().
        model_name: Optional override of the model name.
        instruction_template: Instructional text to prepend to the content.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        options: Additional options to pass to the driver.

    Returns:
        A validated instance of the Pydantic model.

    Raises:
        ValueError: If text is empty or None.
        ValidationError: If the extracted data doesn't match the model schema.
    """
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")

    if driver is None:
        driver = get_driver()

    schema = model_cls.model_json_schema()
    result = manual_extract_and_jsonify(driver, text, schema, model_name, instruction_template, ai_cleanup, options)
    return model_cls(**result["json_object"])

def stepwise_extract_with_model(
    model_cls: Type[BaseModel],
    text: str,
    driver: Optional[Driver] = None,
    model_name: str = "",
    instruction_template: str = "Extract the {field_name} from the following text:",
    ai_cleanup: bool = True,
    debug: bool = False,
    fields: Optional[List[str]] = None,
    options: Dict[str, Any] = {},
) -> BaseModel:
    """Extracts structured information into a Pydantic model by processing each field individually.

    For each field in the model, makes a separate LLM call to extract that specific field,
    then combines the results and validates the complete model instance.

    Args:
        model_cls: The Pydantic BaseModel class to extract into.
        text: The raw text to extract information from.
        driver: Optional LLM driver instance. If None, uses get_driver().
        model_name: Optional override of the model name.
        instruction_template: Template for instructional text, should include {field_name} placeholder.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
        debug: Whether to print debug information during extraction.
        fields: Optional list of field names to extract. If None, extracts all fields.
        options: Additional options to pass to the driver.

    Returns:
        A validated instance of the Pydantic model.

    Raises:
        ValueError: If text is empty or None.
        ValidationError: If the extracted data doesn't match the model schema.
        KeyError: If a requested field doesn't exist in the model.
    """
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")

    if driver is None:
        driver = get_driver()

    data = {}
    validation_errors = []

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
        if debug:
            print(f"\n=== Extracting field: {field_name} ===")
            print(f"Field info: {field_info}")
            print(f"Field type: {field_info.annotation}")

        # Create field schema using tools.create_field_schema
        field_schema = {
            "value": create_field_schema(
                field_name,
                field_info.annotation,
                field_info.description
            )
        }

        if debug:
            print(f"Field schema: {field_schema}")
            print(f"Prompt template: {instruction_template.format(field_name=field_name)}")

        try:
            result = manual_extract_and_jsonify(
                driver,
                text,
                field_schema,
                model_name,
                instruction_template.format(field_name=field_name),
                ai_cleanup,
                options
            )

            if debug:
                # Make a shallow copy to avoid modifying the original
                safe_result = result.copy()

                if "usage" in safe_result:
                    safe_result["usage"] = {
                        k: v for k, v in safe_result["usage"].items()
                        if k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
                    }

                print(f"Raw extraction result: {safe_result}")

            extracted_value = result["json_object"]["value"]

            # Convert value using tools.convert_value
            try:
                converted_value = convert_value(
                    extracted_value,
                    field_info.annotation,
                    allow_shorthand=True
                )
                data[field_name] = converted_value

                if debug:
                    # Make a shallow copy to avoid modifying the original
                    safe_result = result.copy()

                    if "usage" in safe_result:
                        safe_result["usage"] = {
                            k: v for k, v in safe_result["usage"].items()
                            if k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
                        }

                    print(f"Raw extraction result: {safe_result}")

            except ValueError as e:
                error_msg = f"Type conversion failed for {field_name}: {str(e)}"
                validation_errors.append(error_msg)
                if debug:
                    print(f"Error: {error_msg}")
                
        except Exception as e:
            error_msg = f"Extraction failed for {field_name}: {str(e)}"
            validation_errors.append(error_msg)
            if debug:
                print(f"Error: {error_msg}")
    
    if validation_errors and debug:
        print("\n=== Validation Errors ===")
        for error in validation_errors:
            print(f"- {error}")
    
    try:
        return model_cls(**data)
    except Exception as e:
        if debug:
            print(f"\n=== Model Validation Error ===\n{str(e)}")
        raise

def manual_extract_and_jsonify(
    driver: Driver,
    text: str,
    json_schema: Dict[str, Any],
    model_name: str = "",
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    options: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Extracts structured information using an explicitly provided driver.

    This variant is useful when you want to directly control which driver
    is used (e.g., OpenAI, Azure, Ollama, LocalHTTP) and optionally override
    the model per call.

    Args:
        driver: The LLM driver instance to use.
        text: The raw text to extract information from.
        json_schema: JSON schema dictionary defining the expected structure.
        instruction_template: Instructional text to prepend to the content.
        model_name: Optional override of the model name.
        ai_cleanup: Whether to attempt AI-based cleanup if JSON parsing fails.
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
    if not text or not text.strip():
        raise ValueError("Text input cannot be empty")

    opts = dict(options)
    if model_name:
        opts["model"] = model_name

    content_prompt = f"{instruction_template} {text}"
    return ask_for_json(driver, content_prompt, json_schema, ai_cleanup, opts)
