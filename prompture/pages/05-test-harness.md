# Cross-Model Test Harness

Prompture includes a spec-driven test runner that lets you compare extraction quality, cost, and token usage across multiple LLM providers and models.

## Quick Start

```bash
# Run a built-in spec
prompture test-suite specs/basic_extraction.json

# Specify providers
prompture test-suite specs/basic_extraction.json --providers openai,groq

# Override models
prompture test-suite specs/basic_extraction.json \
    --models openai/gpt-4o-mini,ollama/llama3.1:8b

# Save JSON report
prompture test-suite specs/basic_extraction.json -o report.json
```

## Output

```
Cross-Model Test Results: basic_extraction
===========================================

Test: person-simple
  Model                  Pass  Tokens      Cost
  ---------------------  -----  ------  --------
  openai/gpt-4o-mini      3/3     847   $0.0003
  ollama/llama3.1:8b      2/3    1203   $0.0000

Overall: 5/6 passed, total cost: $0.0003
```

## Spec Format

A spec file is JSON with three sections:

```json
{
  "meta": {
    "project": "my-project",
    "suite": "suite-name",
    "version": "1.0"
  },
  "models": [
    {"id": "openai/gpt-4o-mini", "driver": "openai", "options": {}},
    {"id": "ollama/llama3.1:8b", "driver": "ollama", "options": {}}
  ],
  "tests": [
    {
      "id": "test-name",
      "prompt_template": "Extract info from: {text}",
      "inputs": [
        {"text": "John is 25 years old."}
      ],
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name", "age"]
      }
    }
  ]
}
```

## Built-in Specs

- **`basic_extraction.json`** -- Simple person, contact, and product extraction.
- **`schema_validation.json`** -- Nested objects, enums, arrays, nullable fields.
- **`strategy_comparison.json`** -- Invoice and medical record extraction for comparing strategies.

## Writing Custom Specs

1. Copy a built-in spec as a starting point.
2. Define your schema using standard JSON Schema.
3. Write prompt templates with `{placeholder}` variables.
4. Run with `prompture test-suite your_spec.json`.
