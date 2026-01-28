# Skill: Scaffold an Extraction Pipeline

When the user wants to build a new extraction use case (e.g. "extract medical records", "parse invoices", "analyze reviews"), scaffold the full pipeline: Pydantic model, field definitions, extraction call, and test.

## Information to Gather

- **Domain / use case**: What kind of data are we extracting? [ASK]
- **Fields**: List of fields with types, or let me infer from a sample text [ASK]
- **Provider/model**: Which model to target (default: `ollama/gpt-oss:20b`) [ASK or use default]
- **Extraction method**: One-shot (`extract_with_model`) or stepwise (`stepwise_extract_with_model`) [ASK or recommend based on complexity]

## Output Files

### 1. Pydantic Model (add to existing or new module)

```python
from pydantic import BaseModel, Field

class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Company or person that issued the invoice")
    invoice_number: str = Field(description="Unique invoice identifier")
    total_amount: float = Field(description="Total amount due in the invoice currency")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    line_items: list = Field(default_factory=list, description="List of individual items")
```

Rules for the model:
- Use `Field(description=...)` on every field — these become LLM instructions
- Use type-appropriate defaults: `""` for str, `0` for int, `0.0` for float, `[]` for list
- Use `Optional[T] = None` only when a field genuinely might not exist in the source

### 2. Field Definitions (optional, if reuse is desired)

Register fields in `prompture/field_definitions.py` using the add-field skill pattern. Only do this if the fields are general-purpose enough to reuse across projects.

### 3. Example Script

Create `examples/{domain}_extraction_example.py` following the add-example skill template.

### 4. Tests

Create tests following the add-test skill pattern:
- Unit test: Validate the Pydantic model accepts expected data shapes
- Integration test: End-to-end extraction from sample text

## Extraction Method Selection Guide

| Scenario | Recommended Method |
|----------|-------------------|
| Simple model, < 8 fields | `extract_with_model` (one-shot, 1 LLM call) |
| Complex model, 8+ fields | `stepwise_extract_with_model` (per-field, N calls but more accurate) |
| No Pydantic model, raw schema | `extract_and_jsonify` |
| Structured input data (CSV, JSON) | `extract_from_data` (TOON input, saves 45-60% tokens) |
| DataFrame input | `extract_from_pandas` |
| Non-JSON output (text, HTML, markdown) | `render_output` |

## Sample Extraction Call

```python
from prompture import extract_with_model

result = extract_with_model(
    model_cls=InvoiceData,
    text=invoice_text,
    model_name="ollama/gpt-oss:20b",
    instruction_template="Extract all invoice details from the following document:",
)

invoice = result["model"]       # Pydantic model instance
usage = result["usage"]         # Token counts and cost
```
