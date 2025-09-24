"""
Example: Using manual_extract_and_jsonify with Ollama.

This example demonstrates how to:
1. Manually initialize the Ollama driver (ignoring AI_PROVIDER env).
2. Provide a text input and a JSON schema to enforce structured extraction.
3. Call `manual_extract_and_jsonify` with:
   - Default instruction template.
   - A custom instruction template.
4. Override the model per call using `model_name="gpt-oss:20b"`.
5. Inspect and print the JSON output and usage metadata.
"""

import json
from prompture import manual_extract_and_jsonify
from prompture.drivers import get_driver

# 1. Manually get the Ollama driver
ollama_driver = get_driver("ollama")

# 2. Define the raw text and JSON schema
text = "Maria is 32 years old and works as a software developer in New York. She loves hiking and photography."

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "profession": {"type": "string"},
        "city": {"type": "string"},
        "hobbies": {"type": "array", "items": {"type": "string"}}
    }
}

# === FIRST EXAMPLE: Default instruction template ===
print("Extracting information into JSON with default instruction...")

result = manual_extract_and_jsonify(
    driver=ollama_driver,
    text=text,
    json_schema=json_schema,
    model_name="gpt-oss:20b"  # explicit model override
)

json_output = result["json_string"]
json_object = result["json_object"]
usage = result["usage"]

print("\nRaw JSON output from model:")
print(json_output)

print("\nSuccessfully parsed JSON:")
print(json.dumps(json_object, indent=2))

print("\n=== TOKEN USAGE STATISTICS ===")
print(f"Prompt tokens: {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")
print(f"Model name: {usage['model_name']}")


# === SECOND EXAMPLE: Custom instruction template ===
print("\n\n=== SECOND EXAMPLE - CUSTOM INSTRUCTION TEMPLATE ===")
print("Extracting information with custom instruction...")

custom_result = manual_extract_and_jsonify(
    driver=ollama_driver,
    text=text,
    json_schema=json_schema,
    model_name="gpt-oss:20b",  # keep same model
    instruction_template="Parse the biographical details from this text:",
)

custom_json_output = custom_result["json_string"]
custom_json_object = custom_result["json_object"]
custom_usage = custom_result["usage"]

print("\nRaw JSON output with custom instruction:")
print(custom_json_output)

print("\nSuccessfully parsed JSON (custom instruction):")
print(json.dumps(custom_json_object, indent=2))

print("\n=== TOKEN USAGE STATISTICS (Custom Template) ===")
print(f"Prompt tokens: {custom_usage['prompt_tokens']}")
print(f"Completion tokens: {custom_usage['completion_tokens']}")
print(f"Total tokens: {custom_usage['total_tokens']}")
print(f"Model name: {custom_usage['model_name']}")
