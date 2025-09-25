# Prompture

`Prompture` is an API-first library for requesting structured **JSON** output from LLMs (or any structure), validating it against a schema, and running comparative tests between models.

## ✨ Features

- ✅ **Structured Output**: Request models to return JSON only
- ✅ **Validation**: Automatic validation with `jsonschema`
- ✅ **Multi-driver**: Run the same specification against multiple drivers (OpenAI, Ollama, Claude, Azure, HTTP, mock)
- ✅ **Reports**: Generate JSON reports with results
- ✅ **Usage Tracking**: **NEW** - Automatic token and cost monitoring for all calls
- ✅ **AI Cleanup**: Automatically fix malformed JSON responses using AI

<br>

> [!TIP]
> Starring this repo helps more developers discover Prompture ✨
> 
>![prompture_no_forks](https://github.com/user-attachments/assets/720f888e-a885-4eb3-970c-ba5809fe2ce7)
> 
>  🔥 Also check out my other project [RepoGif](https://github.com/jhd3197/RepoGif) – the tool I used to generate the GIF above!
<br>


## 🆕 Token and Cost Tracking

Starting with this version, `extract_and_jsonify` and `ask_for_json` automatically include token usage and cost information:

```python
from prompture import extract_and_jsonify

# AI_PROVIDER environment variable should be set to "ollama", "openai", "azure", or "claude"

# Extract JSON with automatic driver selection
result = extract_and_jsonify(
    text="Text to process",
    json_schema=json_schema,
    model_name="gemma3"  # optional model override
)

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
        "model_name": string
    }
}
```

## 🚀 Driver Initialization

Prompture offers two approaches to initialize drivers: automatic (environment-based) and manual (explicit). Each has its own benefits depending on your use case.

### Automatic Initialization

The `extract_and_jsonify()` function provides a convenient way to extract JSON from text without manually initializing a driver. It automatically uses the appropriate driver based on your environment configuration:

```python
from prompture import extract_and_jsonify

# Define your schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

# Extract JSON with automatic driver initialization
result = extract_and_jsonify(
    "John is 28 years old",
    schema,
    instruction_template="Extract the person's information:"
)

# Access the results
json_output = result["json_string"]
json_object = result["json_object"]
usage = result["usage"]

print(f"Extracted data: {json_object}")
print(f"Tokens used: {usage['total_tokens']}")
print(f"Cost: ${usage['cost']:.6f}")
```

### Configuration

The function uses the `AI_PROVIDER` environment variable to determine which driver to use. It supports all the standard features, including:

- Schema validation
- Token usage tracking
- Cost calculation
- AI-based cleanup for malformed JSON

### Parameters

- `text`: The raw text to extract information from
- `json_schema`: JSON schema dictionary defining the expected structure
- `model_name`: Optional model name to override the default for the selected driver
- `instruction_template`: Template string for the extraction instruction (default: "Extract information from the following text:")
- `ai_cleanup`: Whether to attempt AI-based cleanup if JSON parsing fails (default: True)
- `options`: Additional options to pass to the driver

### Return Structure

The function returns:
```python
{
    "json_string": str,    # The original JSON string
    "json_object": dict,   # The parsed JSON object
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
        "cost": float,     # Cost in USD
        "model_name": str
    }
}
```

### Supported Drivers

- **OllamaDriver**: Cost = $0.00 (free local models)
- **OpenAIDriver**: Cost automatically calculated based on the model
- **ClaudeDriver**: Cost automatically calculated based on the model
- **HuggingFaceDriver**: Cost = $0.00 (free local models)
- **AzureDriver**: Cost automatically calculated based on the model
- **LocalHTTPDriver**: Cost = $0.00 (self-hosted models)


### Manual Initialization

The manual approach gives you more explicit control over driver configuration and allows using multiple drivers simultaneously. Use `get_driver()` to initialize a specific driver, then pass it to `manual_extract_and_jsonify()`:

```python
from prompture import manual_extract_and_jsonify
from prompture.drivers import get_driver

# Initialize driver explicitly
ollama_driver = get_driver("ollama")

# Define schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

# Extract JSON with manual driver
result = manual_extract_and_jsonify(
    driver=ollama_driver,
    text="John is 28 years old",
    json_schema=schema,
    model_name="gpt-oss:20b"  # optional model override
)

# Access results (same structure as extract_and_jsonify)
json_output = result["json_string"]
json_object = result["json_object"]
usage = result["usage"]
```

#### Benefits of Manual Initialization

- **Multiple Drivers**: Run different drivers simultaneously (e.g., Ollama and OpenAI)
- **Custom Configuration**: Initialize drivers with specific settings
- **Explicit Control**: More control over which driver handles each request
- **Model Flexibility**: Easily switch between models for the same driver
- **Testing**: Better for testing scenarios where you need specific driver configurations

## 🔎 Pydantic Model Extraction

This document provides a comprehensive comparison between the two extraction modes available in Prompture: `extract_with_model` and `stepwise_extract_with_model`. These functions enable structured data extraction from text using Large Language Models (LLMs) with Pydantic model validation.

### Overview

Prompture offers two distinct approaches for extracting structured data from unstructured text using Pydantic models:

1. **`extract_with_model`** - Single-call extraction that processes all model fields in one LLM request
2. **`stepwise_extract_with_model`** - Multi-call extraction that processes each model field individually

Both functions return validated Pydantic model instances, but they differ significantly in their approach, performance characteristics, and use cases.

### Function Reference

#### `extract_with_model`

**Purpose:** Extracts structured information into a Pydantic model instance using a single LLM call.

**Signature:**
```python
def extract_with_model(
    model_cls: Type[BaseModel],
    text: str,
    driver: Optional[Driver] = None,
    model_name: str = "",
    instruction_template: str = "Extract information from the following text:",
    ai_cleanup: bool = True,
    options: Dict[str, Any] = {},
) -> BaseModel
```

**Parameters:**
- `model_cls`: The Pydantic BaseModel class to extract into
- `text`: The raw text to extract information from
- `driver`: Optional LLM driver instance (uses `get_driver()` if None)
- `model_name`: Optional override of the model name
- `instruction_template`: Instructional text to prepend to the content
- `ai_cleanup`: Whether to attempt AI-based cleanup if JSON parsing fails
- `options`: Additional options to pass to the driver

**Behavior:**
1. Converts the Pydantic model to its JSON schema
2. Makes a single LLM call with the complete schema
3. Parses and validates the response against the model
4. Returns a validated model instance

#### `stepwise_extract_with_model`

**Purpose:** Extracts structured information into a Pydantic model by processing each field individually.

**Signature:**
```python
def stepwise_extract_with_model(
    model_cls: Type[BaseModel],
    text: str,
    driver: Optional[Driver] = None,
    model_name: str = "",
    instruction_template: str = "Extract the {field_name} from the following text:",
    ai_cleanup: bool = True,
    options: Dict[str, Any] = {},
) -> BaseModel
```

**Parameters:**
- `model_cls`: The Pydantic BaseModel class to extract into
- `text`: The raw text to extract information from
- `driver`: Optional LLM driver instance (uses `get_driver()` if None)
- `model_name`: Optional override of the model name
- `instruction_template`: Template for instructional text (must include `{field_name}` placeholder)
- `ai_cleanup`: Whether to attempt AI-based cleanup if JSON parsing fails
- `options`: Additional options to pass to the driver

**Behavior:**
1. Iterates through each field in the model
2. Makes a separate LLM call for each field with a focused prompt
3. Collects all field values
4. Validates the complete model instance
5. Returns a validated model instance

### Comparison Table

| Aspect | `extract_with_model` | `stepwise_extract_with_model` |
|--------|---------------------|------------------------------|
| **LLM Calls** | Single call | Multiple calls (one per field) |
| **Prompt Complexity** | Complex schema-based | Simple field-focused |
| **Token Usage** | Lower (shared context) | Higher (repeated context) |
| **Cost** | Lower | Higher |
| **Speed** | Faster | Slower |
| **Reliability** | Schema coherence | Field-level accuracy |
| **Error Recovery** | All-or-nothing | Per-field recovery |
| **Context Length** | Full text per call | Full text per field |
| **Field Dependencies** | Can leverage relationships | Independent extraction |

### Practical Examples

#### Example Model

```python
from pydantic import BaseModel
from typing import List, Optional

class Person(BaseModel):
    name: str
    age: int
    profession: str
    city: str
    hobbies: List[str]
    education: Optional[str] = None
```

#### Using `extract_with_model`

```python
from prompture import extract_with_model

text = "Maria is 32 years old and works as a software developer in New York. She loves hiking and photography."

person = extract_with_model(Person, text)
print(person.name)  # "Maria"
print(person.profession)  # "software developer"
print(person.hobbies)  # ["hiking", "photography"]
```

#### Using `stepwise_extract_with_model`

```python
from prompture import stepwise_extract_with_model

text = "Maria is 32 years old and works as a software developer in New York. She loves hiking and photography."

person = stepwise_extract_with_model(Person, text)
print(person.name)  # "Maria"
print(person.profession)  # "software developer"
print(person.hobbies)  # ["hiking", "photography"]
```

### When to Choose Each Approach

#### Choose `extract_with_model` when:

- **Cost efficiency is important** - Single LLM call reduces token usage and API costs
- **Speed is critical** - Faster processing with fewer network round trips
- **Fields are interdependent** - The LLM can leverage relationships between fields
- **Simple models** - Fewer fields mean less complex prompts
- **High coherence required** - All fields must be consistent with each other

#### Choose `stepwise_extract_with_model` when:

- **Accuracy per field is paramount** - Each field gets focused attention
- **Complex models** - Many fields where individual focus improves results
- **Field independence** - Fields don't rely on each other for context
- **Error resilience** - Ability to recover from individual field extraction failures
- **Debugging needs** - Easier to identify and fix issues with specific fields
- **Cost is secondary** - When accuracy is more important than efficiency

### Performance and Reliability Considerations

#### Performance Metrics

**Token Efficiency:**
- `extract_with_model`: ~30-50% fewer tokens due to shared context
- `stepwise_extract_with_model`: Higher token usage from repeated text and simpler prompts

**Latency:**
- `extract_with_model`: Single network round trip
- `stepwise_extract_with_model`: Multiple round trips (N fields = N calls)

**Throughput:**
- `extract_with_model`: Better for batch processing
- `stepwise_extract_with_model`: Better for real-time, field-by-field updates

#### Reliability Factors

**Error Handling:**
- `extract_with_model`: Schema validation ensures all fields are present and valid
- `stepwise_extract_with_model`: Individual field validation with potential partial failures

**Context Quality:**
- `extract_with_model`: Full context available for all fields simultaneously
- `stepwise_extract_with_model`: Full context available for each field individually

**Consistency:**
- `extract_with_model`: Better cross-field consistency
- `stepwise_extract_with_model`: Potential for field-level inconsistencies

### Best Practices and Recommendations

#### General Recommendations

1. **Start with `extract_with_model`** for most use cases due to efficiency
2. **Use `stepwise_extract_with_model`** when field accuracy is critical
3. **Profile performance** for your specific use case and model size
4. **Consider cost implications** when choosing between approaches

#### Optimization Tips

**For `extract_with_model`:**
- Keep models focused and not overly complex
- Use clear, descriptive field names and descriptions
- Consider model size limits for very large schemas

**For `stepwise_extract_with_model`:**
- Order fields logically if there are dependencies
- Use descriptive field names in the instruction template
- Consider parallel processing for multiple fields if supported

#### Error Handling

```python
from prompture import extract_with_model, stepwise_extract_with_model
from pydantic import ValidationError

def safe_extract(model_cls, text, use_stepwise=False):
    try:
        if use_stepwise:
            return stepwise_extract_with_model(model_cls, text)
        else:
            return extract_with_model(model_cls, text)
    except ValidationError as e:
        print(f"Validation failed: {e}")
        # Handle validation errors
    except Exception as e:
        print(f"Extraction failed: {e}")
        # Handle other errors
```

#### Monitoring and Debugging

**Track key metrics:**
- Extraction success rate
- Average processing time
- Token usage per extraction
- Cost per extraction

**Debugging approaches:**
- `extract_with_model`: Check the complete schema and response
- `stepwise_extract_with_model`: Debug individual field extractions

### Conclusion

Both extraction modes serve important roles in the Prompture ecosystem. `extract_with_model` provides efficiency and coherence for most use cases, while `stepwise_extract_with_model` offers precision and resilience when individual field accuracy is paramount.

Choose based on your specific requirements for accuracy, performance, cost, and complexity. Consider prototyping with both approaches to determine which works best for your particular use case and data characteristics.

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
```

## Ollama Model Comparison Example

This example demonstrates how to compare different Ollama models using a specific script located at `examples/ollama_models_comparison.py`.

| Model            | Success | Prompt | Completion | Total | Fields | Validation | Name                | Price    | Variants | Screen Size | Warranty | Is New |
|------------------|---------|--------|------------|-------|--------|------------|---------------------|----------|----------|-------------|----------|--------|
| gpt-oss:20b      | True    | 801    | 945        | 1746  | 8      | ✓          | GalaxyFold Ultra    | 1299.99  | 9        | 6.9         | 3        | True   |
| deepseek-r1:latest | True  | 757    | 679        | 1436  | 8      | ✗          | GalaxyFold Ultra    | 1299.99  | 3        | 6.9         | None     | True   |
| llama3.1:8b      | True    | 746    | 256        | 1002  | 8      | ✓          | GalaxyFold Ultra    | 1299.99  | 3        | 6.9         | 3        | True   |
| gemma3:latest    | True    | 857    | 315        | 1172  | 8      | ✗          | GalaxyFold Ultra    | 1299.99  | 3        | 6.9         | None     | True   |
| qwen2.5:1.5b     | True    | 784    | 236        | 1020  | 8      | ✓          | GalaxyFold Ultra    | 1299.99  | 3        | 6.9         | 3        | True   |
| qwen2.5:3b       | True    | 784    | 273        | 1057  | 9      | ✓          | GalaxyFold Ultra    | 1299.99  | 3        | 6.9         | 3        | True   |
| mistral:latest   | True    | 928    | 337        | 1265  | 8      | ✓          | GalaxyFold Ultra    | 1299.99  | 3        | 6.9         | 3        | True   |

> **Successful models (7):** gpt-oss:20b, deepseek-r1:latest, llama3.1:8b, gemma3:latest, qwen2.5:1.5b, qwen2.5:3b, mistral:latest

You can run this comparison yourself with:
`python examples/ollama_models_comparison.py`

This example script compares multiple Ollama models on a complex task of extracting structured information from a smartphone description using a detailed JSON schema. The purpose of this example is to illustrate how `Prompture` can be used to test and compare different models on the same structured output task, showing their success rates, token usage, and validation results.

**Location:** `examples/ollama_models_comparison.py`
