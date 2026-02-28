---
name: update-pricing
description: Update LLM model pricing in Prompture. Pricing comes from two sources — models.dev live cache (primary) and JSON rate files in prompture/infra/rates/ (capabilities KB). Use when model prices change, new models launch, or models.dev data is stale.
metadata:
  author: prompture
  version: "3.0"
---

# Update Model Pricing

## How Pricing Works

Pricing is resolved by `CostMixin._calculate_cost()` using **models.dev live rates only** (per 1M tokens). If no live rates are found, cost is $0.00.

Model configuration (`tokens_param`, `supports_temperature`) is resolved by `CostMixin._get_model_config()` from the **capabilities knowledge base** (JSON rate files in `prompture/infra/rates/`) overlaid onto models.dev live data.

### models.dev Integration

The mapping from prompture provider names to models.dev provider names lives in `prompture/infra/model_rates.py`:

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

### Capabilities Knowledge Base (JSON rate files)

Per-provider JSON files in `prompture/infra/rates/` define model capabilities:

| File | Provider | Notes |
|------|----------|-------|
| `rates/openai.json` | OpenAI | `tokens_param` varies by model |
| `rates/anthropic.json` | Anthropic | All use `max_tokens` |
| `rates/google.json` | Google | All use `max_tokens` |
| `rates/groq.json` | Groq | All use `max_tokens` |
| `rates/xai.json` | xAI/Grok | All use `max_tokens` |
| `rates/azure.json` | Azure | Mixed OpenAI/Claude/Mistral models |
| `rates/openrouter.json` | OpenRouter | All use `max_tokens` |
| `rates/moonshot.json` | Moonshot | All use `max_tokens` |

Each entry contains:
```json
{
  "model-name": {
    "supports_temperature": true,
    "supports_tool_use": true,
    "supports_structured_output": true,
    "supports_vision": true,
    "is_reasoning": false,
    "context_window": 128000,
    "max_output_tokens": 16384,
    "modalities_input": ["text", "image"],
    "modalities_output": ["text"],
    "api_type": "openai",
    "tokens_param": "max_completion_tokens"
  }
}
```

### Free/local drivers (always $0):

`ollama_driver.py`, `lmstudio_driver.py`, `local_http_driver.py`, `airllm_driver.py`, `hugging_driver.py`

## When to Update What

| Scenario | Action |
|----------|--------|
| New model from an existing provider | Add entry to the provider's JSON rate file in `rates/` |
| models.dev has wrong/outdated prices | Force refresh: see below |
| New provider | Add `rates/{provider}.json`, entry to `PROVIDER_MAP`, discovery integration |
| Model pricing changed | Usually nothing — models.dev updates automatically |

## Refreshing models.dev Cache

```python
from prompture.model_rates import refresh_rates_cache
refresh_rates_cache(force=True)  # Fetch fresh data regardless of TTL
```

Or delete the cache file:
```bash
rm ~/.prompture/cache/models_dev.json
```

## Steps for Adding/Updating Models

1. **Edit** the appropriate JSON file in `prompture/infra/rates/`
2. **Set** `tokens_param` correctly — `"max_completion_tokens"` for newer OpenAI models (gpt-5.x, gpt-4o, o-series), `"max_tokens"` for everything else
3. **Set** `api_type` — `"openai"`, `"anthropic"`, `"google"`, or `"openai-compatible"`
4. **Run tests**: `pytest tests/ -x -q`

## Side Effects

- `prompture/infra/discovery.py` reads KB model IDs via `get_kb_models_for_provider()` to list available models (static detection)
- `PROVIDER_MAP` entries also feed discovery via `get_all_provider_models()` (models.dev detection)
- Adding a model to the KB makes it appear in `get_available_models()` even without an API key configured
- `CostMixin._get_model_config()` reads `tokens_param` and `supports_temperature` from the KB

## Verification

```bash
# Check live rates for a model
python -c "from prompture.model_rates import get_model_rates; print(get_model_rates('openai', 'gpt-4o'))"

# Check capabilities for a model
python -c "from prompture.model_rates import get_model_capabilities; print(get_model_capabilities('openai', 'gpt-5.1'))"

# Check cache age
python -c "from prompture.model_rates import _META_FILE; import json; print(json.load(open(_META_FILE)))"

# Run tests
pytest tests/ -x -q
```
