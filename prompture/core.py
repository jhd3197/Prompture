"""Core utilities: Helpers for requesting JSON from LLM.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict

from .drivers import get_driver
from .driver import Driver


def clean_json_text(text: str) -> str:
    """Attempts to extract a valid JSON object string from text.

    Handles multiple possible formatting issues:
    - Removes <think>...</think> blocks.
    - Strips markdown code fences (```json ... ```).
    - Falls back to first {...} block found.

    Args:
        text: Raw string that may contain JSON plus extra formatting.

    Returns:
        A string that best resembles valid JSON content.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    if text.startswith("```"):
        start_fence = text.find("```")
        if start_fence != -1:
            start_content = text.find("\n", start_fence)
            if start_content != -1:
                end_fence = text.find("```", start_content)
                if end_fence != -1:
                    return text[start_content + 1:end_fence].strip()
                else:
                    return text[start_content + 1 :].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


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
