# Prompture

[![PyPI version](https://badge.fury.io/py/prompture.svg)](https://badge.fury.io/py/prompture)
[![Python Versions](https://img.shields.io/pypi/pyversions/prompture.svg)](https://pypi.org/project/prompture/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/prompture)](https://pepy.tech/project/prompture)
![GitHub Repo stars](https://img.shields.io/github/stars/jhd3197/prompture?style=social)


**Prompture** is an API-first library for getting **structured JSON** (or any structure) from LLMs, validating it, and benchmarking multiple models with one spec.

## ✨ Features

- ✅ **Structured output** → JSON schema enforcement, or direct **Pydantic** instances
- ✅ **Stepwise extraction** → Per-field prompts, with smart type conversion (incl. shorthand numbers)
- ✅ **Multi-driver** → OpenAI, Azure, Claude, Ollama, LM Studio, Google, Groq, OpenRouter, Grok, HTTP, Mock, HuggingFace (via `get_driver()`)
- ✅ **Usage & cost** → Token + $ tracking on every call (`usage` from driver meta)
- ✅ **AI cleanup** → Optional LLM pass to fix malformed JSON
- ✅ **Batch testing** → Define suites and compare models (spec-driven)
- 🧪 **Experimental TOON output** → Request Token-Oriented Object Notation when you need ultra-compact text (see [analysis](toon_token_analysis.md))

<br>

> [!TIP]
> Starring this repo helps more developers discover Prompture ✨
> 
>![prompture_no_forks](https://github.com/user-attachments/assets/720f888e-a885-4eb3-970c-ba5809fe2ce7)
> 
>  🔥 Also check out my other project [RepoGif](https://github.com/jhd3197/RepoGif) – the tool I used to generate the GIF above!
<br>


---

## Installation

```bash
pip install prompture
````

---

## Configure a Provider

Model names now support provider prefixes (e.g., "ollama/llama3.1:8b"). The `get_driver_for_model()` function automatically selects the appropriate driver based on the provider prefix.

You can configure providers either through environment variables or by using provider-prefixed model names:

```bash
# Environment variable approach:
export AI_PROVIDER=ollama  # One of: ollama | openai | azure | claude | google | groq | openrouter | grok | lmstudio | http | huggingface

# Only if the provider needs them:
export OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export GROQ_API_KEY=...
export OPENROUTER_API_KEY=...
export GROK_API_KEY=...
export LMSTUDIO_ENDPOINT=...
```

| Provider | Example models                        | Cost calc       |
| -------- | ------------------------------------- | --------------- |
| `ollama` | `ollama/llama3.1:8b`, `ollama/qwen2.5:3b` | `$0.00` (local) |
| `openai` | `openai/gpt-4`, `openai/gpt-3.5-turbo` | Automatic       |
| `azure`  | `azure/deployed-name`                  | Automatic       |
| `claude` | `claude/claude-3`                      | Automatic       |
| `google` | `google/gemini-1.5-pro`, `google/gemini-1.5-flash` | Automatic |
| `groq` | `groq/llama2-70b-4096`, `groq/mixtral-8x7b-32768` | Automatic |
| `openrouter` | `openrouter/openai/gpt-3.5-turbo`, `openrouter/anthropic/claude-2` | Automatic |
| `grok` | `grok/grok-4-fast-reasoning`, `grok/grok-3-mini` | Automatic |
| `lmstudio` | `lmstudio/local-model` | `$0.00` (local) |
| `huggingface` | `hf/local-or-endpoint`            | `$0.00` (local) |
| `http`   | `http/self-hosted`                     | `$0.00`         |

---

## Quickstart: Pydantic in one line (auto driver)

Use `extract_with_model` for a single LLM call that fills your Pydantic model.

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

text = "Maria is 32, a software developer in New York. She loves hiking and photography."

# Uses get_driver_for_model() internally based on model name prefix
person = extract_with_model(Person, text, model_name="ollama/gpt-oss:20b")
print(person.dict())
```

**Why start here?** It's fast (one call), cost-efficient, and returns a validated Pydantic instance.

---

## 📋 Field Definitions

Prompture includes a powerful **field definitions system** that provides a centralized registry of structured data extraction fields. This system enables consistent, reusable field configurations across your data extraction workflows with built-in fields for common use cases like personal info, contact details, professional data, and more.

**Key benefits:**
- 🎯 Pre-configured fields with descriptions and extraction instructions
- 🔄 Template variables like `{{current_year}}`, `{{current_date}}`, `{{current_datetime}}`
- 🔌 Seamless Pydantic integration via `field_from_registry()`
- ⚙️ Easy custom field registration

### Using Built-in Fields

```python
from pydantic import BaseModel
from prompture import field_from_registry, stepwise_extract_with_model

class Person(BaseModel):
    name: str = field_from_registry("name")
    age: int = field_from_registry("age")
    email: str = field_from_registry("email")
    occupation: str = field_from_registry("occupation")
    company: str = field_from_registry("company")

# Built-in fields include: name, age, email, phone, address, city, country,
# occupation, company, education_level, salary, and many more!

result = stepwise_extract_with_model(
    Person,
    "John Smith is 25 years old, software engineer at TechCorp, john@example.com",
    model_name="openai/gpt-4"
)
```

### Registering Custom Fields

```python
from prompture import register_field, field_from_registry

# Register a custom field with template variables
register_field("document_date", {
    "type": "str",
    "description": "Document creation or processing date",
    "instructions": "Use {{current_date}} if not specified in document",
    "default": "{{current_date}}",
    "nullable": False
})

# Use custom field in your model
class Document(BaseModel):
    title: str = field_from_registry("name")
    created_date: str = field_from_registry("document_date")
```

📚 **[View Full Field Definitions Reference →](https://prompture.readthedocs.io/en/latest/field_definitions_reference.html)**

---

## JSON-first (low-level primitives)

When you want raw JSON with a schema and full control, use `ask_for_json` or `extract_and_jsonify`.

```python
from prompture.drivers import get_driver
from prompture import ask_for_json, extract_and_jsonify

schema = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

# 1) ask_for_json: you provide the full content prompt
resp1 = ask_for_json(
    content_prompt="Extract the person's info from: John is 28 and lives in Miami.",
    json_schema=schema,
    model_name="google/gemini-1.5-pro"
)
print(resp1["json_object"], resp1["usage"])

# 2) extract_and_jsonify: you provide text & an instruction template; it builds the prompt
resp2 = extract_and_jsonify(
    text="John is 28 and lives in Miami.",
    json_schema=schema,
    model_name="groq/mixtral-8x7b-32768",
    instruction_template="Extract the person's information:"
)
print(resp2["json_object"], resp2["usage"])
```

### Experimental TOON output

Prompture can ask for TOON (Token-Oriented Object Notation) instead of JSON by setting `output_format="toon"` on `ask_for_json`, `extract_and_jsonify`, `manual_extract_and_jsonify`, or `extract_with_model`. Responses are parsed back into Python dicts, so your downstream code still receives JSON-compatible structures.

```python
result = extract_and_jsonify(
    text="Alice Johnson is a 30-year-old data scientist...",
    json_schema=schema,
    model_name="lmstudio/deepseek/deepseek-r1-0528-qwen3-8b",
    output_format="toon",
)
print(result["json_string"])  # TOON text
print(result["json_object"])  # regular dict
```

> [!IMPORTANT]
> TOON output is **experimental**. General-purpose models often emit more verbose completions when asked for TOON, so total token usage can increase (see `toon_token_analysis.md`). Treat it as an opt-in mode until TOON-aware fine-tunes or adapters are available.

### Return shape (JSON helpers)

```python
{
  "json_string": str,
  "json_object": dict,
  "usage": {
    "prompt_tokens": int,
    "completion_tokens": int,
    "total_tokens": int,
    "cost": float,
    "model_name": str
  }
}
```

> If the model returns malformed JSON and `ai_cleanup=True`, a second LLM pass tries to fix it.

---

## Pydantic: one-shot vs stepwise

Prompture supports two Pydantic extraction modes:

* **`extract_with_model`** → Single call; global context; best cost/latency; coherent fields
* **`stepwise_extract_with_model`** → One call per field; higher per-field accuracy; resilient

| Aspect         | `extract_with_model` (one-shot)        | `stepwise_extract_with_model` (per-field) |
| -------------- | -------------------------------------- | ----------------------------------------- |
| LLM calls      | 1                                      | N (one per field)                         |
| Speed & cost   | **Faster / cheaper**                   | Slower / higher                           |
| Accuracy       | Good global coherence                  | **Higher per-field accuracy**             |
| Error handling | All-or-nothing                         | **Per-field recovery**                    |
| Best when      | Fields are related; throughput matters | Correctness per field is critical         |

### Examples

```python
from prompture import extract_with_model, stepwise_extract_with_model

person1 = extract_with_model(Person, text, model_name="openrouter/anthropic/claude-2")
print(person1.dict())

res = stepwise_extract_with_model(Person, text, model_name="grok/grok-4-fast-reasoning")
print(res["model"].dict())
print(res["usage"])  # includes per-field usage and totals
```

**Stepwise extras:** internally uses `tools.create_field_schema` + `tools.convert_value` (with `allow_shorthand=True`) so values like `"3.4m"`, `"2k"`, `"1.2b"` can be converted to typed fields where appropriate.

---

## Manual control with logging

`manual_extract_and_jsonify` is like `extract_and_jsonify` but adds structured debug logging.

```python
from prompture import manual_extract_and_jsonify
from prompture.drivers import get_driver
from prompture.tools import LogLevel

driver = get_driver("ollama")
res = manual_extract_and_jsonify(
    driver=driver,
    text="Maria works as a software developer in New York.",
    json_schema={
      "type": "object",
      "required": ["city", "profession"],
      "properties": {"city": {"type": "string"}, "profession": {"type": "string"}}
    },
    model_name="llama3.1:8b",
    options={"temperature": 0.2},
    verbose_level=LogLevel.DEBUG  # TRACE for full prompts/results
)
print(res["json_object"])
```

---


**Example output (Ollama comparison)** — see `examples/ollama_models_comparison.py` for a richer comparison table.

---


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

---

## Error handling notes

* With `ai_cleanup=True`, a second LLM pass attempts to fix malformed JSON; on success, `usage` may be a minimal stub.
* `extract_and_jsonify` will **skip tests** under `pytest` if there’s a local server connection error (e.g., Ollama), instead of failing the suite.
* All functions raise `ValueError` for empty text.

---

## Tips & Best Practices

* Add `description` to schema fields (or Pydantic field metadata) for better extractions.
* Start with **one-shot Pydantic**; switch specific fields to **stepwise** if they’re noisy.
* Track usage/cost before scaling; tweak `temperature` in `options` if consistency wobbles.
* Use `verbose_level=TRACE` in dev to see prompts/results and tighten your specs.

---

## Contributing

PRs welcome! Add tests and—if adding drivers or patterns—drop an example under `examples/`.
