import json
from prompture import extract_and_jsonify
from prompture.drivers import get_driver

# 1. Instantiate the driver
# Make sure your environment variables are set:
# - CLAUDE_API_KEY: Your Anthropic API key for Claude
# - CLAUDE_MODEL_NAME: (Optional) Your preferred Claude model
#   Available models:
#   - claude-opus-4-1-20250805 (Claude Opus 4.1)
#   - claude-opus-4-20250514 (Claude Opus 4.0)
#   - claude-sonnet-4-20250514 (Claude Sonnet 4.0)
#   - claude-3-7-sonnet-20250219 (Claude Sonnet 3.7)
#   - claude-3-5-haiku-20241022 (Claude Haiku 3.5)
#   Default is "claude-3-5-haiku-20241022"
claude_driver = get_driver("claude")

# You can also specify a different model during initialization:
# claude_driver = get_driver("claude", model="claude-3-7-sonnet-latest")

# 2. Define the raw text and JSON schema
# Raw text containing information to extract
text = "Maria is 32 years old and works as a software developer in New York. She loves hiking and photography."

# 3. Define the JSON schema
# This schema specifies the expected structure for the user information
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

# 4. Call extract_and_jsonify with default instruction
print("Extracting information into JSON with default instruction...")
result = extract_and_jsonify(
    driver=claude_driver,
    text=text,
    json_schema=json_schema
)

# Extract JSON output and usage metadata from the new return type
json_output = result["json_string"]
json_object = result["json_object"]
usage = result["usage"]

# 5. Print and validate the output
print("\nRaw JSON output from model:")
print(json_output)

print("\nSuccessfully parsed JSON:")
print(json.dumps(json_object, indent=2))

# 6. Display token usage and cost information
print("\n=== TOKEN USAGE STATISTICS ===")
print(f"Prompt tokens: {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")
print(f"Cost: ${usage['cost']:.6f}")
print(f"Model used: {usage['model_name']}")

# 7. Example with custom instruction template and different model
print("\n\n=== SECOND EXAMPLE - CUSTOM INSTRUCTION & DIFFERENT MODEL ===")
print("Extracting information with custom instruction using Claude Sonnet 3.7...")
custom_result = extract_and_jsonify(
    driver=claude_driver,
    text=text,
    json_schema=json_schema,
    instruction_template="Parse the biographical details from this text:",
    options={"model": "claude-3-7-sonnet-20250219"}  # Override model for this call
)

# Extract JSON output and usage metadata
custom_json_output = custom_result["json_string"]
custom_json_object = custom_result["json_object"]
custom_usage = custom_result["usage"]

print("\nRaw JSON output with custom instruction:")
print(custom_json_output)

print("\n=== TOKEN USAGE STATISTICS (Custom Template) ===")
print(f"Prompt tokens: {custom_usage['prompt_tokens']}")
print(f"Completion tokens: {custom_usage['completion_tokens']}")
print(f"Total tokens: {custom_usage['total_tokens']}")
print(f"Cost: ${custom_usage['cost']:.6f}")
print(f"Model used: {usage['model_name']}")
