# Migration Guide

## Response Shape

All extraction functions return a dict with these **stable** keys:

```python
result = extract_and_jsonify(...)
result["json_string"]   # raw JSON string
result["json_object"]   # parsed dict/list
result["usage"]         # token counts and cost
```

Use `.get()` for optional keys to stay forward-compatible:

```python
strategy_used = result["usage"].get("strategy", "unknown")
```

## Strategy Parameter

The `strategy` parameter gives explicit control over how structured output is obtained.

```python
from prompture.extraction.strategy import StructuredOutputStrategy

result = extract_with_model(
    MyModel, text,
    model_name="openai/gpt-4o",
    strategy=StructuredOutputStrategy.TOOL_CALL,
)
```

The default is `"auto"`, which picks the best available strategy. Existing code that doesn't pass `strategy` continues to work unchanged.

## Cost and Pricing

`usage["cost"]` is calculated from:

1. **models.dev live cache** -- queried at runtime for up-to-date pricing.
2. **Local rate files** in `prompture/infra/rates/` -- fallback when models.dev is unavailable.

## Driver Interface

The `Driver` base class provides a stable `generate(prompt, options)` contract. Custom drivers that accept `**kwargs` are forward-compatible:

```python
class MyDriver(Driver):
    def generate(self, prompt, options=None, **kwargs):
        ...
```

## Import Path Changes

| Old Path | New Path | Status |
|----------|----------|--------|
| `prompture.bridges.tukuy_backend` | `prompture.infra.tukuy_backend` | Shimmed |
| `prompture.integrations.tukuy_bridge` | `prompture.extraction.tukuy_bridge` | Shimmed |

Update your imports to the new paths. Old paths will be removed in a future major version.
