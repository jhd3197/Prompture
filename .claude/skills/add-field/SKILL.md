---
name: add-field
description: Add predefined field definitions to the Prompture field registry. Handles field structure, categories, template variables, enum support, and thread-safe registration. Use when adding reusable extraction fields.
metadata:
  author: prompture
  version: "1.0"
---

# Add Field Definitions

Adds new fields to the global `BASE_FIELD_DEFINITIONS` registry in `prompture/field_definitions.py`.

## Before Starting

Ask the user for each field:
- **Field name** — lowercase, underscore-separated (e.g. `linkedin_url`, `blood_type`)
- **Category** — Person, Contact, Professional, Financial, Location, Education, Demographic, Social Media, Task Management, or a new one
- **Type** — `str`, `int`, `float`, `bool`, `list`, `dict`
- **Description** — what this field represents
- **Instructions** — how the LLM should extract or compute the value
- **Default** — type-appropriate: `""` for str, `0` for int, `0.0` for float, `[]` for list, `False` for bool
- **Nullable** — `True` if the field can legitimately be absent
- **Enum values** (optional) — list of allowed string values

## Field Structure

```python
"field_name": {
    "type": str,
    "description": "What this field represents.",
    "instructions": "How the LLM should extract or compute this value.",
    "default": "",
    "nullable": False,
},
```

### Optional keys

- `"enum"`: `["low", "medium", "high"]`
- Template variables in `instructions`: `{{current_year}}`, `{{current_date}}`, `{{current_datetime}}`, `{{current_month}}`, `{{current_day}}`, `{{current_weekday}}`, `{{current_iso_week}}`

## Steps

### 1. Edit `prompture/field_definitions.py`

Add fields to `BASE_FIELD_DEFINITIONS` under the right category comment. If the category is new, add a comment header:

```python
    # ── Medical Fields ──────────────────────────────────
```

Alphabetical order within each category.

### 2. Verify

```bash
python -c "from prompture.field_definitions import get_field_definition; print(get_field_definition('field_name'))"
pytest tests/test_field_definitions.py -x -q
```
