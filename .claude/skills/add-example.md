# Skill: Add an Example File

When the user asks to create a new usage example, follow this template and conventions.

## Information to Gather

- **Topic / use case** (e.g. "medical record extraction", "product review analysis"): [ASK]
- **Which extraction method** to demonstrate: `extract_with_model`, `stepwise_extract_with_model`, `extract_and_jsonify`, `extract_from_data`, or `render_output`: [ASK if unclear]
- **Which provider/model** to target (default: `ollama/gpt-oss:20b`): [ASK or use default]

## File Conventions

- Location: `examples/{descriptive_name}_example.py`
- Filename: lowercase, underscores, ends with `_example.py`
- Must be a standalone runnable script (no test framework dependency)

## Template

```python
"""
Example: {Title}

This example demonstrates:
1. {Feature 1}
2. {Feature 2}
3. {Feature 3}

Requirements:
    pip install prompture
    # Set up your provider credentials in .env
"""

import json
from pydantic import BaseModel, Field
from prompture import extract_with_model  # or whichever function

# ── 1. Define the output model ──────────────────────────

class MyModel(BaseModel):
    field1: str = Field(description="...")
    field2: int = Field(description="...")

# ── 2. Input text ───────────────────────────────────────

text = """
Paste realistic sample text here.
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

- Use section comments with the `# ── N. Title ──` format to divide the script
- Always print both the extracted result and the usage metadata
- Use realistic sample text, not placeholder lorem ipsum
- Import only from `prompture` public API (never internal modules)
- Include a docstring header listing what the example demonstrates and any setup requirements
- If the example needs a specific provider, mention the env var in the docstring
- Keep it under 80 lines if possible — examples should be concise
