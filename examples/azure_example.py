"""
Example: Using extract_and_jsonify with Azure OpenAI.

This script demonstrates:
1. Initializing the Azure driver manually (ignoring AI_PROVIDER).
2. Extracting structured information from text into JSON using a schema.
3. Overriding the model per call with `model_name`.
4. Running both a default extraction and a custom-instruction extraction.

Environment variables required:
- AZURE_API_KEY: Your Azure OpenAI API key
- AZURE_API_ENDPOINT: Your Azure OpenAI endpoint URL
- AZURE_DEPLOYMENT_ID: Your deployment ID for the model
- AZURE_API_VERSION: (Optional) Defaults to "2023-07-01-preview"
"""

import json

from prompture import extract_and_jsonify

# 1. Define the raw text to parse
text = "Maria is 32 years old and works as a software developer in New York. She loves hiking and photography."

# 2. Define the JSON schema for expected output
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

# === FIRST EXAMPLE: Default instruction with explicit Azure model ===
print("Extracting information into JSON with default instruction...")

result = extract_and_jsonify(
    text=text,
    json_schema=json_schema,
    model_name="azure/gpt-4o-mini",  # explicitly override to a known Azure deployment
)

json_output = result["json_string"]
json_object = result["json_object"]
usage = result["usage"]

print("\nRaw JSON output from Azure model:")
print(json_output)

print("\nSuccessfully parsed JSON:")
print(json.dumps(json_object, indent=2))

print("\n=== TOKEN USAGE STATISTICS ===")
print(f"Prompt tokens: {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")
print(f"Cost: ${usage['cost']:.6f}")
print(f"Model used: {usage['model_name']}")


# === SECOND EXAMPLE: Custom instruction with a different model ===
print("\n\n=== SECOND EXAMPLE - CUSTOM INSTRUCTION TEMPLATE ===")
print("Extracting information with custom instruction using Azure GPT-4...")

custom_result = extract_and_jsonify(
    text=text,
    json_schema=json_schema,
    instruction_template="Parse the biographical details from this text:",
    model_name="azure/gpt-4",  # override to a different Azure deployment/model
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
print(f"Cost: ${custom_usage['cost']:.6f}")
print(f"Model used: {custom_usage['model_name']}")
