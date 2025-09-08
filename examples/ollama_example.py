import json
from prompture import extract_and_jsonify
from prompture.drivers import get_driver

# 1. Instantiate the driver
# Ollama driver usually works out-of-the-box if Ollama is running locally
# You may need to set OLLAMA_HOST environment variable if Ollama is running on a different host
ollama_driver = get_driver("ollama")

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
    driver=ollama_driver,
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

# 6. Display token usage information
print("\n=== TOKEN USAGE STATISTICS ===")
print(f"Prompt tokens: {usage['prompt_tokens']}")
print(f"Completion tokens: {usage['completion_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")

# 7. Example with custom instruction template
print("\n\n=== SECOND EXAMPLE - CUSTOM INSTRUCTION TEMPLATE ===")
print("Extracting information with custom instruction...")
custom_result = extract_and_jsonify(
    driver=ollama_driver,
    text=text,
    json_schema=json_schema,
    instruction_template="Parse the biographical details from this text:"
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