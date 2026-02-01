---
name: add-example
description: Create a new Prompture usage example script. Follows project conventions for file naming, section structure, docstrings, and output formatting. Use when demonstrating extraction use cases or provider integrations.
metadata:
  author: prompture
  version: "1.1"
---

# Add an Example File

Creates a standalone runnable example in `examples/`.

## Before Starting

Ask the user:
- **Topic / use case** (e.g. "medical record extraction", "product review analysis")
- **Extraction method** — `extract_with_model`, `stepwise_extract_with_model`, `extract_and_jsonify`, `extract_from_data`, or `render_output`
- **Provider/model** — e.g. `openai/gpt-4o`, `moonshot/kimi-k2-0905-preview`, `ollama/llama3.1:8b`

## Two Example Styles

### Provider example (e.g. `moonshot_example.py`, `grok_example.py`)

Shows `extract_and_jsonify` with a specific provider. Two sections:
1. Default instruction with the provider's primary model
2. Custom instruction with the same or a different model

### Use-case example (e.g. `medical_record_example.py`, `resume_cv_example.py`)

Shows extraction for a specific domain using Pydantic models.

## Conventions

- File: `examples/{descriptive_name}_example.py`
- Standalone — no test framework imports
- Always print extracted result and usage metadata (prompt_tokens, completion_tokens, total_tokens, cost, model_name)
- Realistic sample text, not lorem ipsum
- Under 100 lines when possible

## Provider Example Template

```python
"""
Example: Using extract_and_jsonify with {Provider}.

This script demonstrates:
1. Initializing the {Provider} driver manually (ignoring AI_PROVIDER).
2. Extracting structured information from text using a JSON schema.
3. Overriding the {Provider} model per call with `model_name`.
4. Running both a default extraction and a custom-instruction extraction.

Setup:
    export {PROVIDER}_API_KEY="your-key-here"
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

# === FIRST EXAMPLE: Default instruction with {Model} ===
print("Extracting information into JSON with default instruction...")

result = extract_and_jsonify(
    text=text,
    json_schema=json_schema,
    model_name="{provider}/{model}",  # explicitly select model
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


# === SECOND EXAMPLE: Custom instruction with {Alt Model} ===
print("\n\n=== SECOND EXAMPLE - CUSTOM INSTRUCTION & DIFFERENT MODEL ===")
print("Extracting information with custom instruction...")

custom_result = extract_and_jsonify(
    text=text,
    json_schema=json_schema,
    instruction_template="Parse the biographical details from this text:",
    model_name="{provider}/{alt_model}",  # override model here
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
```

## Use-Case Example Template

```python
"""
Example: {Title}

This example demonstrates:
1. {Feature 1}
2. {Feature 2}

Requirements:
    pip install prompture
    # Set up provider credentials in .env
"""

import json

from pydantic import BaseModel, Field

from prompture import extract_with_model

# 1. Define the output model
class MyModel(BaseModel):
    field1: str = Field(description="...")
    field2: int = Field(description="...")

# 2. Input text
text = """
Realistic sample text here.
"""

# 3. Extract
MODEL = "openai/gpt-4o-mini"

result = extract_with_model(
    model_cls=MyModel,
    text=text,
    model_name=MODEL,
)

# 4. Results
print("Extracted model:")
print(result["model"])
print()
print("Usage metadata:")
print(json.dumps(result["usage"], indent=2))
```

## Rules

- Import only from `prompture` public API
- Include docstring header listing features and setup requirements
- If provider-specific, mention the env var in the docstring
- Check available models for the provider: `python -c "from prompture.model_rates import get_all_provider_models; print(get_all_provider_models('{models_dev_name}')[:10])"`
