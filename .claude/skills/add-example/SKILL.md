---
name: add-example
description: Create a new Prompture usage example script. Follows project conventions for file naming, section structure, docstrings, and output formatting. Use when demonstrating extraction use cases or provider integrations.
metadata:
  author: prompture
  version: "1.0"
---

# Add an Example File

Creates a standalone runnable example in `examples/`.

## Before Starting

Ask the user:
- **Topic / use case** (e.g. "medical record extraction", "product review analysis")
- **Extraction method** — `extract_with_model`, `stepwise_extract_with_model`, `extract_and_jsonify`, `extract_from_data`, or `render_output`
- **Provider/model** — default: `ollama/gpt-oss:20b`

## Conventions

- File: `examples/{descriptive_name}_example.py`
- Standalone — no test framework imports
- Section dividers: `# ── N. Title ──`
- Always print extracted result and usage metadata
- Realistic sample text, not lorem ipsum
- Under 80 lines when possible

## Template

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

# ── 1. Define the output model ──────────────────────────

class MyModel(BaseModel):
    field1: str = Field(description="...")
    field2: int = Field(description="...")

# ── 2. Input text ───────────────────────────────────────

text = """
Realistic sample text here.
"""

# ── 3. Extract ──────────────────────────────────────────

MODEL = "ollama/gpt-oss:20b"

result = extract_with_model(
    model_cls=MyModel,
    text=text,
    model_name=MODEL,
)

# ── 4. Results ──────────────────────────────────────────

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
