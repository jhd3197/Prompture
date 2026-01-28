"""
Example: Using extract_and_jsonify with different drivers and models.

This example demonstrates how to:
1. Initialize a specific driver (OpenAI in this case).
2. Provide a text input and a JSON schema that defines the structure of the output.
3. Use `extract_and_jsonify` to extract structured JSON with:
   - A default instruction template.
   - A custom instruction template.
4. Override the model per call using the `model_name` argument.
5. Inspect and print the raw JSON output, parsed JSON object, and usage metadata.

The `usage` metadata returned includes:
- prompt_tokens: number of tokens used in the input prompt
- completion_tokens: number of tokens used in the model’s output
- total_tokens: sum of prompt and completion tokens
- cost: approximate cost (depends on driver’s pricing table)
- model_name: name of the model used for the request
"""

import json

from prompture import extract_and_jsonify

# 1. Define the raw text to be processed
text = "Maria is 32 years old and works as a software developer in New York. She loves hiking and photography."

# 2. Define the JSON schema
# This schema enforces the structure of the extracted information.
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "profession": {"type": "string"},
        "city": {"type": "string"},
        "hobbies": {"type": "array", "items": {"type": "string"}},
    },
}

# === FIRST EXAMPLE: Default instruction template ===
print("Extracting information into JSON with default instruction...")

# Call extract_and_jsonify with a model override
result = extract_and_jsonify(
    text=text,
    json_schema=json_schema,
    model_name="openai/gpt-4o-mini",  # model is explicitly chosen here
)

# Extract JSON output and metadata
json_output = result["json_string"]
json_object = result["json_object"]
usage = result["usage"]

# Print results
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

custom_result = extract_and_jsonify(
    text=text,
    json_schema=json_schema,
    model_name="openai/gpt-3.5-turbo",  # override with a different model
    instruction_template="Parse the biographical details from this text:",
)

custom_json_output = custom_result["json_string"]
custom_json_object = custom_result["json_object"]
custom_usage = custom_result["usage"]

# Print results
print("\nRaw JSON output from model (custom instruction):")
print(custom_json_output)

print("\nSuccessfully parsed JSON (custom instruction):")
print(json.dumps(custom_json_object, indent=2))

print("\n=== TOKEN USAGE STATISTICS (custom instruction) ===")
print(f"Prompt tokens: {custom_usage['prompt_tokens']}")
print(f"Completion tokens: {custom_usage['completion_tokens']}")
print(f"Total tokens: {custom_usage['total_tokens']}")
print(f"Model name: {custom_usage['model_name']}")
