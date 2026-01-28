"""
auto_example.py

Example: Using extract_and_jsonify with Ollama.

This script demonstrates:
1. How the driver is automatically selected from environment variables.
2. Extracting structured JSON using a simple schema.
3. Running the same extraction across multiple Ollama models.

Environment variables required:
- AI_PROVIDER=ollama
- OLLAMA_ENDPOINT=http://localhost:11434/api/generate
- OLLAMA_MODEL=gpt-oss:20b (default; can be overridden per call)

Because AI_PROVIDER=ollama, the script will automatically use the Ollama driver.
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

# 3. Define llm models to test
llm_models = [
    "ollama/gpt-oss:20b",
    "ollama/qwen2.5:3b",
    "ollama/mistral:latest",
    "azure/gpt-4",
    "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b",
]

# 4. Run extraction for each model
for model in llm_models:
    print(f"\n=== Extracting information using LLM model: {model} ===")
    try:
        result = extract_and_jsonify(
            text=text,
            json_schema=json_schema,
            instruction_template="Extract the biographical details from this text:",
            model_name=model,
        )

        # Access results
        json_output = result["json_string"]
        json_object = result["json_object"]
        usage = result["usage"]

        # Print results
        print("\nRaw JSON output:")
        print(json_output)

        print("\nSuccessfully parsed JSON:")
        print(json.dumps(json_object, indent=2))

        print("\n=== TOKEN USAGE STATISTICS ===")
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        print(f"Model used: {usage['model_name']}")

    except Exception as e:
        print(f"Error running model {model}: {e!s}")
