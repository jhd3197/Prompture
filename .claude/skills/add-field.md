# Skill: Add Field Definitions

When the user asks to add new field definitions to the registry, follow this process.

## Information to Gather

Ask the user:
- **Field name(s)** — lowercase, underscore-separated (e.g. `linkedin_url`, `blood_type`)
- **Category** — which logical group (Person, Contact, Professional, Financial, etc.) or a new category
- **For each field**: type, description, instructions, default value, nullable, and optionally enum values

## Where Fields Live

All predefined fields are in `prompture/field_definitions.py` inside the `BASE_FIELD_DEFINITIONS` dict, organized by category comments.

## Field Definition Structure

Every field follows this exact shape:

```python
"field_name": {
    "type": str,           # Python type: str, int, float, bool, list, dict
    "description": "What this field represents.",
    "instructions": "How the LLM should extract or compute this value.",
    "default": "",         # Type-appropriate default (0 for int, "" for str, [] for list, False for bool)
    "nullable": False,     # True if the field can be None/null
},
```

### Optional keys

- `"enum"`: list of allowed string values (e.g. `["low", "medium", "high"]`)
- Template variables in `instructions`: `{{current_year}}`, `{{current_date}}`, `{{current_datetime}}`, `{{current_month}}`, `{{current_day}}`, `{{current_weekday}}`, `{{current_iso_week}}`

## Steps

### 1. Edit `prompture/field_definitions.py`

Add the new field(s) to `BASE_FIELD_DEFINITIONS` under the appropriate category comment block. If the category is new, add a clear comment header like:

```python
    # ── Medical Fields ──────────────────────────────────
```

Place the entries in alphabetical order within their category.

### 2. Verify

Run:
```bash
python -c "from prompture.field_definitions import get_field_definition; print(get_field_definition('field_name'))"
```

Then run the field definitions tests:
```bash
pytest tests/test_field_definitions.py -x -q
```

## Example

Adding a `blood_type` field:

```python
"blood_type": {
    "type": str,
    "description": "Blood type classification (ABO system with Rh factor).",
    "instructions": "Extract the blood type. Use standard notation: A+, A-, B+, B-, AB+, AB-, O+, O-.",
    "default": "",
    "nullable": True,
    "enum": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
},
```
