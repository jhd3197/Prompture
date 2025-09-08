# Prompture

`Prompture` is an API-first library for requesting structured **JSON** output from LLMs (or any structure), validating it against a schema, and running comparative tests between models.

## ✨ Features

- ✅ **Structured Output**: Request models to return JSON only
- ✅ **Validation**: Automatic validation with `jsonschema`
- ✅ **Multi-driver**: Run the same specification against multiple drivers (OpenAI, Ollama, Claude, HTTP, mock)
- ✅ **Reports**: Generate JSON reports with results
- ✅ **Usage Tracking**: **NEW** - Automatic token and cost monitoring for all calls

## 🆕 Token and Cost Tracking (New)

Starting with this version, `extract_and_jsonify` and `ask_for_json` automatically include token usage and cost information:

```python
from prompture import extract_and_jsonify
from prompture.drivers import OllamaDriver

driver = OllamaDriver(endpoint="http://localhost:11434/api/generate", model="gemma3")
result = extract_and_jsonify(driver, "Text to process", json_schema)

# Now returns both the response and usage information
json_output = result["json_string"]
usage = result["usage"]

print(f"Tokens used: {usage['total_tokens']}")
print(f"Cost: ${usage['cost']:.6f}")
```

### Return Structure

The main functions now return:
```python
{
    "json_string": str,    # The original JSON string
    "json_object": dict,   # The parsed JSON object
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
        "cost": float      # Cost in USD (0.0 for free models)
    }
}
```

### Supported Drivers

- **OllamaDriver**: Cost = $0.00 (free local models)
- **OpenAIDriver**: Cost automatically calculated based on the model
- **ClaudeDriver**: Cost automatically calculated based on the model

## Batch Running and Testing Prompts

`run_suite_from_spec` enables you to define and run test suites against multiple models using a specification file. This powerful feature allows you to systematically test and compare different models using a consistent set of prompts and validation criteria. Here's how it works:

```python
from prompture import run_suite_from_spec
from prompture.drivers import MockDriver

spec = {
    "meta": {"project": "test"},
    "models": [{"id": "mock1", "driver": "mock", "options": {}}],
    "tests": [
        {
            "id": "t1",
            "prompt_template": "Extract user info: '{text}'",
            "inputs": [{"text": "Juan is 28 and lives in Miami. He likes basketball and coding."}],
            "schema": {"type": "object", "required": ["name", "interests"]}
        }
    ]
}
drivers = {"mock": MockDriver()}
report = run_suite_from_spec(spec, drivers)
print(report)
```

The generated report includes comprehensive results for each test, model, and input combination:
- Validation status for each response
- Usage statistics (tokens, costs) per model
- Execution times
- Generated JSON responses

## Quick Usage (example):

```py
from prompture import run_suite_from_spec, drivers
spec = { ... }
report = run_suite_from_spec(spec, drivers={"mock": drivers.MockDriver()})
print(report)