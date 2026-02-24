"""
Example: Using extract_and_jsonify with CachiBot.

This script demonstrates:
1. Extracting structured information via the CachiBot proxy API.
2. CachiBot routes requests to upstream providers (OpenAI, Groq, etc.)
   through a single OpenAI-compatible endpoint.
3. Using different upstream models per call with `model_name`.
4. Running both a default extraction and a custom-instruction extraction.

Setup:
    export CACHIBOT_API_KEY="your-key-here"
    # or add to .env file
"""

import json

from prompture import extract_and_jsonify

# 1. Define the raw text
text = "Maria is 32 years old and works as a software developer in New York. She loves hiking and photography."

# 2. Define the JSON schema
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

# === FIRST EXAMPLE: Default instruction via CachiBot ===
print("Extracting information into JSON with default instruction...")

result = extract_and_jsonify(
    text=text,
    json_schema=json_schema,
    model_name="cachibot/openai/gpt-4o-mini",  # upstream provider/model via CachiBot proxy
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
print(f"Cost: ${usage['cost']:.6f}")
print(f"Model used: {usage['model_name']}")


# === SECOND EXAMPLE: Custom instruction with a different upstream model ===
print("\n\n=== SECOND EXAMPLE - CUSTOM INSTRUCTION & DIFFERENT MODEL ===")
print("Extracting information with custom instruction via CachiBot...")

custom_result = extract_and_jsonify(
    text=text,
    json_schema=json_schema,
    instruction_template="Parse the biographical details from this text:",
    model_name="cachibot/groq/llama-3.3-70b-versatile",  # different upstream provider/model
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
