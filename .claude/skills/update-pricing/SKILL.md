---
name: update-pricing
description: Update LLM model pricing in Prompture. Pricing now comes from two sources — models.dev live cache (primary) and hardcoded MODEL_PRICING dicts (fallback). Use when model prices change, new models launch, or models.dev data is stale.
metadata:
  author: prompture
  version: "2.0"
---

# Update Model Pricing

## How Pricing Works (Two-Tier System)

Pricing is resolved by `CostMixin._calculate_cost()` in this order:

1. **models.dev live rates** (primary) — cached from `https://models.dev/api.json`
2. **Hardcoded `MODEL_PRICING`** (fallback) — per-driver class variable
3. **Zero** — if neither source has data

Most newer drivers (moonshot, zai, modelscope) have `MODEL_PRICING = {}` because their pricing comes entirely from models.dev. Older drivers (openai, claude, google, groq, grok, openrouter, azure) have hardcoded pricing as a fallback.

### models.dev Integration

The mapping from prompture provider names to models.dev provider names lives in `prompture/model_rates.py`:

```python
PROVIDER_MAP = {
    "openai": "openai",
    "claude": "anthropic",
    "google": "google",
    "groq": "groq",
    "grok": "xai",
    "azure": "azure",
    "openrouter": "openrouter",
    "moonshot": "moonshotai",
    "zai": "zai",
}
```

The cache is stored at `~/.prompture/cache/models_dev.json` with a TTL configured by `settings.model_rates_ttl_days` (default 7 days).

## When to Update What

| Scenario | Action |
|----------|--------|
| New model from an existing provider | Usually nothing — models.dev updates automatically |
| models.dev has wrong/outdated prices | Force refresh: see below |
| Provider not on models.dev | Update hardcoded `MODEL_PRICING` in the driver file |
| New provider added to models.dev | Add entry to `PROVIDER_MAP` in `model_rates.py` |
| Model pricing unit changed (per-1K vs per-1M) | Check `_PRICING_UNIT` on the driver class |

## Refreshing models.dev Cache

```python
from prompture.model_rates import refresh_rates_cache
refresh_rates_cache(force=True)  # Fetch fresh data regardless of TTL
```

Or delete the cache file:
```bash
rm ~/.prompture/cache/models_dev.json
```

## Updating Hardcoded MODEL_PRICING (Fallback)

Only needed for drivers where models.dev doesn't have pricing data, or as a safety fallback.

### Files with hardcoded pricing:

| File | Provider | Unit | Notes |
|------|----------|------|-------|
| `prompture/drivers/openai_driver.py` | OpenAI | per 1K tokens | Also has `tokens_param`, `supports_temperature` per model |
| `prompture/drivers/claude_driver.py` | Anthropic | per 1K tokens | |
| `prompture/drivers/google_driver.py` | Google | per 1M chars | Uses `_PRICING_UNIT = 1_000_000` |
| `prompture/drivers/groq_driver.py` | Groq | per 1K tokens | |
| `prompture/drivers/grok_driver.py` | xAI | per 1M tokens | Uses `_PRICING_UNIT = 1_000_000` |
| `prompture/drivers/openrouter_driver.py` | OpenRouter | per 1K tokens | |
| `prompture/drivers/azure_driver.py` | Azure | per 1K tokens | |

### Drivers with NO hardcoded pricing (models.dev only):

| File | Provider | models.dev name |
|------|----------|-----------------|
| `prompture/drivers/moonshot_driver.py` | Moonshot (Kimi) | `moonshotai` |
| `prompture/drivers/zai_driver.py` | Z.ai (Zhipu) | `zai` |
| `prompture/drivers/modelscope_driver.py` | ModelScope | — (not in PROVIDER_MAP) |

### Free/local drivers (always $0):

`ollama_driver.py`, `lmstudio_driver.py`, `local_http_driver.py`, `airllm_driver.py`, `hugging_driver.py`

## Steps for Hardcoded Updates

1. **Search the web** for the provider's current pricing page
2. **Read** the current `MODEL_PRICING` dict in the driver file
3. **Update** prices, add new models, remove discontinued ones
4. **Preserve** extra keys like `"tokens_param"` or `"supports_temperature"` — these control API behavior, not just pricing
5. **Check the unit**: most drivers use per-1K tokens, but check `_PRICING_UNIT` on the class
6. **Run tests**: `pytest tests/ -x -q`

### Format

```python
MODEL_PRICING = {
    "model-name": {
        "prompt": 0.005,        # cost per unit (see _PRICING_UNIT)
        "completion": 0.015,
        "tokens_param": "max_completion_tokens",  # API parameter name (optional)
        "supports_temperature": True,              # override for this model (optional)
    },
}
```

Both `"prompt"` and `"completion"` keys are required. Extra keys are optional and model-specific.

## Side Effects

- `prompture/discovery.py` reads `MODEL_PRICING` keys to list available models (static detection)
- `PROVIDER_MAP` entries also feed discovery via `get_all_provider_models()` (models.dev detection)
- Adding a model to `MODEL_PRICING` makes it appear in `get_available_models()` even without an API key configured
- `CostMixin._get_model_config()` reads `tokens_param` and `supports_temperature` from `MODEL_PRICING` as fallback when models.dev data is unavailable

## Verification

```bash
# Check live rates for a model
python -c "from prompture.model_rates import get_model_rates; print(get_model_rates('openai', 'gpt-4o'))"

# Check capabilities for a model
python -c "from prompture.model_rates import get_model_capabilities; print(get_model_capabilities('moonshot', 'kimi-k2.5'))"

# Check cache age
python -c "from prompture.model_rates import _META_FILE; import json; print(json.load(open(_META_FILE)))"

# Run tests
pytest tests/ -x -q
```
