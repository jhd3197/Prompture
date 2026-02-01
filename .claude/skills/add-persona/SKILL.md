---
name: add-persona
description: Create reusable persona system prompts for Prompture. Covers Persona dataclass, template variables, composition, trait registry, global registry, serialization, and Conversation integration. Use when defining system prompts for extraction or agent behavior.
metadata:
  author: prompture
  version: "1.0"
---

# Add a Persona

Creates reusable, composable system prompt definitions with template variables and a global registry.

## Before Starting

Ask the user for:
- **Persona name** (lowercase_with_underscores, used as registry key)
- **Purpose** — what the persona should do (extract data, analyze, summarize, etc.)
- **System prompt** — the core instruction text
- **Template variables** — any `{{variable}}` placeholders and their defaults
- **Constraints** — rules the persona must follow
- **Model hint** — suggested model string (optional, e.g. `"openai/gpt-4o"`)
- **Settings** — default driver options (optional, e.g. `{"temperature": 0.0}`)

## Key Concepts

### Persona Dataclass (frozen/immutable)

```python
from prompture.persona import Persona

persona = Persona(
    name="invoice_extractor",
    system_prompt=(
        "You are a financial document processor. "
        "Extract invoice data from the provided text and return structured JSON. "
        "Use {{currency}} as the default currency when not specified."
    ),
    description="Invoice data extraction with configurable currency.",
    traits=("precise_output",),          # Trait names from trait registry
    variables={"currency": "USD"},       # Default template variable values
    constraints=[                        # Appended as ## Constraints section
        "Output ONLY valid JSON.",
        "Use null for unknown values.",
        "Amounts must be numeric, not strings.",
    ],
    model_hint="openai/gpt-4o-mini",     # Suggested model
    settings={"temperature": 0.0},       # Default driver options
)
```

### Persona Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique identifier (required) |
| `system_prompt` | `str` | Template text with `{{variable}}` placeholders (required) |
| `description` | `str` | Human-readable purpose description |
| `traits` | `tuple[str, ...]` | Trait names resolved from trait registry during `render()` |
| `variables` | `dict[str, Any]` | Default template variable values |
| `constraints` | `list[str]` | Rules appended as a `## Constraints` section |
| `model_hint` | `str \| None` | Suggested `"provider/model"` string |
| `settings` | `dict[str, Any]` | Default driver options (temperature, etc.) |

### Template Variables

Placeholders use `{{variable}}` syntax. Built-in variables are always available:

| Variable | Example Value |
|----------|---------------|
| `{{current_year}}` | `2026` |
| `{{current_date}}` | `2026-02-01` |
| `{{current_month}}` | `February` |
| `{{current_day_of_week}}` | `Sunday` |

Custom variables are set via `variables` dict or at render time:

```python
rendered = persona.render(currency="EUR", max_items="10")
```

**Variable precedence (highest wins):**
1. `kwargs` passed to `render()`
2. `self.variables` (defaults)
3. Built-in template variables

### Rendering

```python
prompt = persona.render()                          # Uses defaults
prompt = persona.render(currency="EUR")            # Override variable
prompt = persona.render(custom_var="custom_value") # Additional variable
```

### Composition

Personas are immutable. Composition returns new instances:

```python
# Extend — append additional instructions
specialized = persona.extend("Focus specifically on line items and tax calculations.")

# Add constraints
stricter = persona.with_constraints([
    "All dates must be in ISO 8601 format.",
    "Round amounts to 2 decimal places.",
])

# Merge two personas with + operator
# Right-side values win on variable/settings conflicts
combined = base_persona + domain_persona
# Result: name="base+domain", prompts concatenated, constraints merged,
#         variables merged (right wins), traits deduped
```

### Trait Registry

Reusable prompt fragments that can be shared across personas:

```python
from prompture.persona import register_trait, get_trait, get_trait_names

# Register traits
register_trait("precise_output", "Always output precisely formatted data with no extraneous text.")
register_trait("multilingual", "Support input in any language. Always respond in {{output_language}}.")

# Use in persona
persona = Persona(
    name="my_persona",
    system_prompt="...",
    traits=("precise_output", "multilingual"),
    variables={"output_language": "English"},
)
# Traits are resolved and appended during render()
```

### Global Persona Registry

```python
from prompture.persona import (
    register_persona, get_persona, get_persona_names,
    clear_persona_registry, reset_persona_registry,
    PERSONAS,  # Dict-like proxy
)

# Register
register_persona(persona)

# Retrieve
p = get_persona("invoice_extractor")       # Returns Persona or None
p = PERSONAS["invoice_extractor"]          # Returns Persona or raises KeyError

# List
names = get_persona_names()                # ["json_extractor", "data_analyst", ...]

# Dict-like access via PERSONAS proxy
PERSONAS["my_persona"] = persona           # Register
del PERSONAS["my_persona"]                 # Unregister
"my_persona" in PERSONAS                   # Check existence
len(PERSONAS)                              # Count

# Reset to built-in only
reset_persona_registry()
```

### Built-in Personas

5 personas are registered on import:

| Name | Description |
|------|-------------|
| `json_extractor` | Precise JSON extraction (temperature=0.0) |
| `data_analyst` | Quantitative analysis with citations |
| `text_summarizer` | Configurable summarization (`{{max_sentences}}` variable) |
| `code_reviewer` | Structured code review (Summary/Issues/Suggestions) |
| `concise_assistant` | Brief, no-elaboration responses |

### Serialization

```python
# JSON
persona.save_json("personas/invoice_extractor.json")
loaded = Persona.load_json("personas/invoice_extractor.json")

# YAML (requires pyyaml)
persona.save_yaml("personas/invoice_extractor.yaml")
loaded = Persona.load_yaml("personas/invoice_extractor.yaml")

# Dict round-trip
d = persona.to_dict()      # {"version": 1, "name": ..., "system_prompt": ..., ...}
p = Persona.from_dict(d)

# Bulk load from directory (registers all loaded personas)
from prompture.persona import load_personas_from_directory
personas = load_personas_from_directory("personas/")
```

### Conversation Integration

```python
from prompture import Conversation

conv = Conversation(
    driver=driver,
    system_prompt=persona.render(),
    options=persona.settings,
)
result = conv.ask("Extract data from this invoice: ...")
```

Or use `model_hint` to select the driver:

```python
from prompture.drivers import get_driver_for_model

if persona.model_hint:
    driver = get_driver_for_model(persona.model_hint)
```

## Implementation Steps

### 1. Define the persona

Create in the appropriate module or a personas directory:

```python
from prompture.persona import Persona, register_persona

my_persona = Persona(
    name="my_persona",
    system_prompt="...",
    description="...",
    constraints=["..."],
    settings={"temperature": 0.0},
)
register_persona(my_persona)
```

### 2. Add traits if needed

```python
from prompture.persona import register_trait

register_trait("my_trait", "Trait text with {{variables}}.")
```

### 3. Write tests

```python
def test_persona_render():
    p = Persona(name="test", system_prompt="Hello {{name}}")
    assert "Hello World" == p.render(name="World")

def test_persona_constraints():
    p = Persona(name="test", system_prompt="Base", constraints=["Rule 1"])
    rendered = p.render()
    assert "## Constraints" in rendered
    assert "Rule 1" in rendered

def test_persona_composition():
    a = Persona(name="a", system_prompt="A")
    b = Persona(name="b", system_prompt="B")
    c = a + b
    assert c.name == "a+b"
    assert "A" in c.render()
    assert "B" in c.render()
```

## Best Practices

- **Use template variables** for anything that might change between uses (language, limits, format preferences)
- **Keep system prompts focused** — one clear role per persona
- **Use constraints for rules** — they're appended as a structured section, easy to compose
- **Use traits for shared fragments** — DRY across personas (e.g., "output JSON only" trait)
- **Set `temperature: 0.0`** for extraction personas — deterministic output is preferred
- **Use `model_hint`** to suggest the best model for the persona's task
- **Serialize to YAML** for human-editable persona libraries
- **Use `load_personas_from_directory()`** to load persona libraries at startup
