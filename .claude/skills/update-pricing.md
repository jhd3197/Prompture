# Skill: Update Model Pricing

When the user asks to update pricing for a provider's models, follow this process.

## Where Pricing Lives

Each driver has a `MODEL_PRICING` class variable — a dict mapping model names to cost-per-1K-token values:

```python
class OpenAIDriver(Driver):
    MODEL_PRICING = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "default": {"prompt": 0.002, "completion": 0.002},
    }
```

Files to check:
- `prompture/drivers/openai_driver.py`
- `prompture/drivers/claude_driver.py`
- `prompture/drivers/google_driver.py`
- `prompture/drivers/groq_driver.py`
- `prompture/drivers/grok_driver.py`
- `prompture/drivers/openrouter_driver.py`
- `prompture/drivers/azure_driver.py`

Local/free drivers (ollama, lmstudio, local_http, airllm) all use `0.0`.

## Steps

1. **Search the web** for the latest pricing page for the provider (e.g. "OpenAI API pricing 2026")
2. **Read** the current `MODEL_PRICING` dict in the driver file
3. **Update** prices, add new models, remove discontinued models
4. **Verify** the `"default"` entry still makes sense as a fallback
5. **Run tests**: `pytest tests/ -x -q`

## Pricing Format

- Values are **cost per 1,000 tokens** in USD
- Always include both `"prompt"` and `"completion"` keys
- Always keep a `"default"` entry as fallback
- Some drivers have extra keys like `"tokens_param"` or `"supports_temperature"` — preserve those

## Also check `prompture/discovery.py`

The discovery module reads `MODEL_PRICING` keys to list available models. Adding/removing models from pricing automatically updates discovery.
