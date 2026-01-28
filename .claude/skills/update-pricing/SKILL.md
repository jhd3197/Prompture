---
name: update-pricing
description: Update LLM model pricing tables in Prompture driver files. Checks provider pricing pages, updates MODEL_PRICING dicts, adds new models, and removes discontinued ones. Use when model prices change or new models launch.
metadata:
  author: prompture
  version: "1.0"
---

# Update Model Pricing

## Where Pricing Lives

Each driver has a `MODEL_PRICING` class variable:

```python
class OpenAIDriver(Driver):
    MODEL_PRICING = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "default": {"prompt": 0.002, "completion": 0.002},
    }
```

Files with pricing:
- `prompture/drivers/openai_driver.py`
- `prompture/drivers/claude_driver.py`
- `prompture/drivers/google_driver.py`
- `prompture/drivers/groq_driver.py`
- `prompture/drivers/grok_driver.py`
- `prompture/drivers/openrouter_driver.py`
- `prompture/drivers/azure_driver.py`

Local/free drivers (ollama, lmstudio, local_http, airllm) use `0.0`.

## Steps

1. **Search the web** for the provider's current pricing page
2. **Read** the current `MODEL_PRICING` dict in the driver file
3. **Update** prices, add new models, remove discontinued ones
4. **Keep** the `"default"` fallback entry
5. **Preserve** extra keys like `"tokens_param"` or `"supports_temperature"`
6. **Run tests**: `pytest tests/ -x -q`

## Format

- Values: **cost per 1,000 tokens** in USD
- Both `"prompt"` and `"completion"` keys required
- Always include `"default"` as fallback

## Side Effects

`prompture/discovery.py` reads `MODEL_PRICING` keys to list available models. Adding or removing models from pricing automatically updates discovery results.
