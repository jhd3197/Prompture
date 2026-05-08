---
name: update-pricing
description: Update LLM model pricing in Prompture. Pricing resolves through a pluggable source registry — by default local KB JSON files (primary, curated) then models.dev (fallback). Use when model prices change, new models launch, models.dev data is stale, or you need to add a custom pricing source.
metadata:
  author: prompture
  version: "4.0"
---

# Update Model Pricing

## How Pricing Resolves

Pricing flows through a **pluggable source registry** at `prompture/infra/pricing.py`. `get_model_rates(provider, model_id)` walks the registered sources in priority order (lowest first) and returns the first dict that includes both `input` and `output` per-1M rates.

Default registration (controlled by `Settings.pricing_source`):

| Setting value | Sources registered (priority order) |
|---------------|-------------------------------------|
| `local_first` (default) | `LocalKBPricingSource` (priority 0) → `ModelsDevPricingSource` (priority 100) |
| `local_only` | `LocalKBPricingSource` only |
| `models_dev_only` | `ModelsDevPricingSource` only |

**Use `local_first` (default)** when you want curated rates with automatic fallback. **`local_only`** for offline/locked-down envs or when you don't trust models.dev. **`models_dev_only`** as a temporary opt-out while local files are unmaintained.

`CostMixin._calculate_cost()` consumes the resolved dict — `input`, `output`, optional `cache_read`, `cache_write`, `reasoning` rates per 1M tokens. Cached prompt tokens are billed at `cache_read` (falls back to `input` rate when not published).

## Local KB layout (the primary source)

Per-provider JSON files in `prompture/infra/rates/` carry both **capabilities** and **pricing** for each model. Adding a `cost` block makes that model use local pricing; omitting it falls through to models.dev.

```json
{
  "gpt-5.5": {
    "supports_temperature": false,
    "supports_tool_use": true,
    "supports_structured_output": true,
    "supports_vision": true,
    "is_reasoning": true,
    "context_window": 1050000,
    "max_output_tokens": 128000,
    "modalities_input": ["text", "image"],
    "modalities_output": ["text"],
    "api_type": "openai",
    "tokens_param": "max_completion_tokens",
    "cost": {
      "input": 5.00,
      "output": 30.00,
      "cache_read": 0.50
    }
  }
}
```

**`cost` block keys** (USD per 1M tokens):
- `input` — required for cost to compute
- `output` — required
- `cache_read` — optional. Discount rate for cached prompt tokens (Anthropic / OpenAI / Kimi all support this).
- `cache_write` — optional. Anthropic only — premium rate for tokens written to prompt cache.
- `reasoning` — optional. For models that bill reasoning tokens separately.

## Vendor pricing pages (verify before editing)

**Always cross-check rates against the vendor's official page**, not just models.dev — feeds get stale.

| Provider | Pricing page |
|----------|--------------|
| OpenAI | https://developers.openai.com/api/docs/pricing |
| Anthropic (Claude) | https://platform.claude.com/docs/en/about-claude/pricing |
| Moonshot (Kimi) | https://platform.kimi.ai/docs (search "Model Pricing") |
| xAI (Grok) | https://docs.x.ai/docs/models |
| Google (Gemini) | https://ai.google.dev/gemini-api/docs/pricing |
| Groq | https://groq.com/pricing |
| OpenRouter | https://openrouter.ai/models |
| Moonshot K2 | https://platform.moonshot.ai/docs/pricing/chat |
| Z.ai (GLM) | https://docs.z.ai/guides/llm/pricing |

When a vendor publishes "cache hit" / "cache read" pricing, encode it as `cache_read`. When they publish a "cache write" / "cache creation" rate (Anthropic only), encode it as `cache_write`.

## When to update what

| Scenario | Action |
|----------|--------|
| Vendor changed prices for a model already in the KB | Update the `cost` block in `rates/{provider}.json` |
| New model from existing provider | Add entry with capabilities + `cost` block |
| New provider | Create `rates/{provider}.json`, add to `PROVIDER_MAP` in `model_rates.py`, wire descriptor in `provider_descriptors.py` |
| Model not in KB but in models.dev | Nothing — fallback already covers it |
| You want to override what models.dev says | Add the `cost` block to local KB; it wins |
| You want to lock pricing to local-only | Set `pricing_source=local_only` in env / `.env` |

## Adding a custom pricing source

For internal price feeds, vendor APIs the registry doesn't cover, or per-environment overrides:

```python
from prompture.infra.pricing import register_pricing_source

class MyInternalFeed:
    name = "internal_feed"
    priority = 50  # between local_kb (0) and models_dev (100)

    def get_rates(self, provider: str, model_id: str) -> dict[str, float] | None:
        # Return {"input": ..., "output": ..., "cache_read": ...} or None
        return _query_my_internal_service(provider, model_id)

register_pricing_source(MyInternalFeed())
```

Sources are ordered by `(priority, name)`. The first to return a dict containing both `input` and `output` wins. Exceptions inside `get_rates` are swallowed and the next source is tried.

## Refreshing the models.dev cache

```python
from prompture.infra.model_rates import refresh_rates_cache
refresh_rates_cache(force=True)
```

Or delete the cache file:
```bash
rm ~/.prompture/cache/models_dev.json
```

## Provider name → models.dev key mapping

Lives in `prompture/infra/model_rates.py` (`PROVIDER_MAP`):

```python
{"openai": "openai", "claude": "anthropic", "google": "google", "groq": "groq",
 "grok": "xai", "azure": "azure", "openrouter": "openrouter",
 "moonshot": "moonshotai", "zai": "zai"}
```

## Free / local drivers (always $0)

`ollama_driver.py`, `lmstudio_driver.py`, `local_http_driver.py`, `airllm_driver.py`, `hugging_driver.py`.

## Steps for adding / updating a model price

1. **Open** the vendor's pricing page (table above) and copy the current rates.
2. **Edit** `prompture/infra/rates/{provider}.json` — find the model entry (or add a new one with capabilities) and add/update the `cost` block.
3. **Verify** with `python -c "from prompture.infra.model_rates import get_model_rates; print(get_model_rates('openai', 'gpt-5.5'))"`. The dict should include exactly the keys you wrote.
4. **Run tests**: `pytest tests/test_pricing.py tests/test_cached_tokens.py -q`
5. **For long-running processes** that imported the LocalKBPricingSource before the edit: the in-memory KB is loaded once at startup. Either restart, or call `LocalKBPricingSource().reload()` on the registered instance.

## Verification commands

```bash
# Check resolved rates for a model
python -c "from prompture.infra.model_rates import get_model_rates; print(get_model_rates('openai', 'gpt-5.5'))"

# Inspect which sources are registered
python -c "from prompture.infra.pricing import get_pricing_sources; print([(s.name, s.priority) for s in get_pricing_sources()])"

# Check that capabilities still load
python -c "from prompture.infra.model_rates import get_model_capabilities; print(get_model_capabilities('openai', 'gpt-5.5'))"

# Run pricing + cost tests
pytest tests/test_pricing.py tests/test_cached_tokens.py tests/test_model_rates.py -q
```

## Side effects of editing the KB

- `prompture/infra/discovery.py` reads KB model IDs via `get_kb_models_for_provider()` for static detection (no API key needed).
- `CostMixin._get_model_config()` still reads `tokens_param` / `supports_temperature` from the KB capabilities.
- A model with a local `cost` block now appears in cost calculations regardless of models.dev cache state — useful when models.dev lags vendor announcements.
