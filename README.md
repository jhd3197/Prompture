<p align="center">
  <h1 align="center">Prompture</h1>
  <p align="center">Structured JSON extraction from any LLM. Schema-enforced, Pydantic-native, multi-provider.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/prompture/"><img src="https://badge.fury.io/py/prompture.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/prompture/"><img src="https://img.shields.io/pypi/pyversions/prompture.svg" alt="Python versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://pepy.tech/project/prompture"><img src="https://static.pepy.tech/badge/prompture" alt="Downloads"></a>
  <a href="https://github.com/jhd3197/prompture"><img src="https://img.shields.io/github/stars/jhd3197/prompture?style=social" alt="GitHub stars"></a>
</p>

---

**Prompture** is a Python library that turns LLM responses into validated, structured data. Define a schema or Pydantic model, point it at any provider, and get typed output back — with token tracking, cost calculation, and automatic JSON repair built in.

```python
from pydantic import BaseModel
from prompture import extract_with_model

class Person(BaseModel):
    name: str
    age: int
    profession: str

person = extract_with_model(Person, "Maria is 32, a developer in NYC.", model_name="openai/gpt-4")
print(person.name)  # Maria
```

## Key Features

- **Structured output** — JSON schema enforcement and direct Pydantic model population
- **12 providers** — OpenAI, Claude, Google, Groq, Grok, Azure, Ollama, LM Studio, OpenRouter, HuggingFace, AirLLM, and generic HTTP
- **TOON input conversion** — 45-60% token savings when sending structured data via [Token-Oriented Object Notation](https://github.com/jhd3197/python-toon)
- **Stepwise extraction** — Per-field prompts with smart type coercion (shorthand numbers, multilingual booleans, dates)
- **Field registry** — 50+ predefined extraction fields with template variables and Pydantic integration
- **Conversations** — Stateful multi-turn sessions with sync and async support
- **Tool use** — Function calling and streaming across supported providers, with automatic prompt-based simulation for models without native tool support
- **Caching** — Built-in response cache with memory, SQLite, and Redis backends
- **Plugin system** — Register custom drivers via entry points
- **Usage tracking** — Token counts and cost calculation on every call
- **Auto-repair** — Optional second LLM pass to fix malformed JSON
- **Batch testing** — Spec-driven suites to compare models side by side

## Installation

```bash
pip install prompture
```

Optional extras:

```bash
pip install prompture[redis]     # Redis cache backend
pip install prompture[serve]     # FastAPI server mode
pip install prompture[airllm]    # AirLLM local inference
```

## Configuration

Set API keys for the providers you use. Prompture reads from environment variables or a `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
GROK_API_KEY=...
OPENROUTER_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
```

Local providers (Ollama, LM Studio) work out of the box with no keys required.

## Providers

Model strings use `"provider/model"` format. The provider prefix routes to the correct driver automatically.

| Provider | Example Model | Cost |
|---|---|---|
| `openai` | `openai/gpt-4` | Automatic |
| `claude` | `claude/claude-3` | Automatic |
| `google` | `google/gemini-1.5-pro` | Automatic |
| `groq` | `groq/llama2-70b-4096` | Automatic |
| `grok` | `grok/grok-4-fast-reasoning` | Automatic |
| `azure` | `azure/deployed-name` | Automatic |
| `openrouter` | `openrouter/anthropic/claude-2` | Automatic |
| `ollama` | `ollama/llama3.1:8b` | Free (local) |
| `lmstudio` | `lmstudio/local-model` | Free (local) |
| `huggingface` | `hf/model-name` | Free (local) |
| `http` | `http/self-hosted` | Free |

## Usage

### One-Shot Pydantic Extraction

Single LLM call, returns a validated Pydantic instance:

```python
from typing import List, Optional
from pydantic import BaseModel
from prompture import extract_with_model

class Person(BaseModel):
    name: str
    age: int
    profession: str
    city: str
    hobbies: List[str]
    education: Optional[str] = None

person = extract_with_model(
    Person,
    "Maria is 32, a software developer in New York. She loves hiking and photography.",
    model_name="openai/gpt-4"
)
print(person.model_dump())
```

### Stepwise Extraction

One LLM call per field. Higher accuracy, per-field error recovery:

```python
from prompture import stepwise_extract_with_model

result = stepwise_extract_with_model(
    Person,
    "Maria is 32, a software developer in New York. She loves hiking and photography.",
    model_name="openai/gpt-4"
)
print(result["model"].model_dump())
print(result["usage"])  # per-field and total token usage
```

| Aspect | `extract_with_model` | `stepwise_extract_with_model` |
|---|---|---|
| LLM calls | 1 | N (one per field) |
| Speed / cost | Faster, cheaper | Slower, higher |
| Accuracy | Good global coherence | Higher per-field accuracy |
| Error handling | All-or-nothing | Per-field recovery |

### JSON Schema Extraction

For raw JSON output with full control:

```python
from prompture import ask_for_json

schema = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

result = ask_for_json(
    content_prompt="Extract the person's info from: John is 28 and lives in Miami.",
    json_schema=schema,
    model_name="openai/gpt-4"
)
print(result["json_object"])  # {"name": "John", "age": 28}
print(result["usage"])        # token counts and cost
```

### TOON Input — Token Savings

Analyze structured data with automatic TOON conversion for 45-60% fewer tokens:

```python
from prompture import extract_from_data

products = [
    {"id": 1, "name": "Laptop", "price": 999.99, "rating": 4.5},
    {"id": 2, "name": "Book", "price": 19.99, "rating": 4.2},
    {"id": 3, "name": "Headphones", "price": 149.99, "rating": 4.7},
]

result = extract_from_data(
    data=products,
    question="What is the average price and highest rated product?",
    json_schema={
        "type": "object",
        "properties": {
            "average_price": {"type": "number"},
            "highest_rated": {"type": "string"}
        }
    },
    model_name="openai/gpt-4"
)

print(result["json_object"])
# {"average_price": 389.99, "highest_rated": "Headphones"}

print(f"Token savings: {result['token_savings']['percentage_saved']}%")
```

Works with Pandas DataFrames via `extract_from_pandas()`.

### Field Definitions

Use the built-in field registry for consistent extraction across models:

```python
from pydantic import BaseModel
from prompture import field_from_registry, stepwise_extract_with_model

class Person(BaseModel):
    name: str = field_from_registry("name")
    age: int = field_from_registry("age")
    email: str = field_from_registry("email")
    occupation: str = field_from_registry("occupation")

result = stepwise_extract_with_model(
    Person,
    "John Smith, 25, software engineer at TechCorp, john@example.com",
    model_name="openai/gpt-4"
)
```

Register custom fields with template variables:

```python
from prompture import register_field

register_field("document_date", {
    "type": "str",
    "description": "Document creation date",
    "instructions": "Use {{current_date}} if not specified",
    "default": "{{current_date}}",
    "nullable": False
})
```

### Conversations

Stateful multi-turn sessions:

```python
from prompture import Conversation

conv = Conversation(model_name="openai/gpt-4")
conv.add_message("system", "You are a helpful assistant.")
response = conv.send("What is the capital of France?")
follow_up = conv.send("What about Germany?")  # retains context
```

### Tool Use

Register Python functions as tools the LLM can call during a conversation:

```python
from prompture import Conversation, ToolRegistry

registry = ToolRegistry()

@registry.tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 22 {units}"

conv = Conversation("openai/gpt-4", tools=registry)
result = conv.ask("What's the weather in London?")
```

For models without native function calling (Ollama, LM Studio, etc.), Prompture automatically simulates tool use by describing tools in the prompt and parsing structured JSON responses:

```python
# Auto-detect: uses native tool calling if available, simulation otherwise
conv = Conversation("ollama/llama3.1:8b", tools=registry, simulated_tools="auto")

# Force simulation even on capable models
conv = Conversation("openai/gpt-4", tools=registry, simulated_tools=True)

# Disable tool use entirely
conv = Conversation("openai/gpt-4", tools=registry, simulated_tools=False)
```

The simulation loop describes tools in the system prompt, asks the model to respond with JSON (`tool_call` or `final_answer`), executes tools, and feeds results back — all transparent to the caller.

### Model Discovery

Auto-detect available models from configured providers:

```python
from prompture import get_available_models

models = get_available_models()
for model in models:
    print(model)  # "openai/gpt-4", "ollama/llama3:latest", ...
```

### Logging and Debugging

```python
import logging
from prompture import configure_logging

configure_logging(logging.DEBUG)
```

### Response Shape

All extraction functions return a consistent structure:

```python
{
    "json_string": str,       # raw JSON text
    "json_object": dict,      # parsed result
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
        "cost": float,
        "model_name": str
    }
}
```

## CLI

```bash
prompture run <spec-file>
```

Run spec-driven extraction suites for cross-model comparison.

## Development

```bash
# Install with dev dependencies
pip install -e ".[test,dev]"

# Run tests
pytest

# Run integration tests (requires live LLM access)
pytest --run-integration

# Lint and format
ruff check .
ruff format .
```

## Contributing

PRs welcome. Please add tests for new functionality and examples under `examples/` for new drivers or patterns.

## License

[MIT](https://opensource.org/licenses/MIT)
