---
name: scaffold-extraction
description: Scaffold a complete extraction pipeline for a new domain — Pydantic model, field definitions, example script, and tests. Use when building a new extraction use case like medical records, invoices, or reviews.
metadata:
  author: prompture
  version: "1.0"
---

# Scaffold an Extraction Pipeline

Builds the full pipeline for a new extraction domain: model, fields, example, and tests.

## Before Starting

Ask the user:
- **Domain / use case** — what kind of data to extract
- **Fields** — list with types, or infer from a sample text
- **Provider/model** — default `ollama/gpt-oss:20b`
- **Method** — one-shot or stepwise (see selection guide below)

## Method Selection Guide

| Scenario | Method |
|----------|--------|
| Simple model, < 8 fields | `extract_with_model` (1 LLM call) |
| Complex model, 8+ fields | `stepwise_extract_with_model` (N calls, more accurate) |
| No Pydantic model, raw schema | `extract_and_jsonify` |
| Structured input (CSV, JSON) | `extract_from_data` (TOON, 45-60% token savings) |
| DataFrame input | `extract_from_pandas` |
| Non-JSON output | `render_output` |

## Deliverables

### 1. Pydantic Model

```python
from pydantic import BaseModel, Field

class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Company that issued the invoice")
    total_amount: float = Field(description="Total amount due")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
```

Rules:
- `Field(description=...)` on every field — these become LLM instructions
- Type-appropriate defaults: `""`, `0`, `0.0`, `[]`
- `Optional[T] = None` only when a field genuinely might not exist

### 2. Field Definitions (optional)

Register in `prompture/field_definitions.py` only if fields are general-purpose. Use the `add-field` skill.

### 3. Example Script

Create `examples/{domain}_extraction_example.py`. Use the `add-example` skill.

### 4. Tests

Unit test for the Pydantic model + integration test for end-to-end extraction. Use the `add-test` skill.

## Sample Call

```python
from prompture import extract_with_model

result = extract_with_model(
    model_cls=InvoiceData,
    text=invoice_text,
    model_name="ollama/gpt-oss:20b",
    instruction_template="Extract all invoice details from the following document:",
)

invoice = result["model"]    # Pydantic instance
usage = result["usage"]      # Token counts and cost
```
