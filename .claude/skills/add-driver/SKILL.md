---
name: add-driver
description: Scaffold a new LLM provider driver for Prompture. Creates the driver class, registers it in the driver registry, adds settings, env template, setup.py extras, and package exports. Use when adding support for a new LLM provider.
metadata:
  author: prompture
  version: "1.0"
---

# Add a New LLM Driver

Scaffolds all files needed to integrate a new LLM provider into Prompture.

## Before Starting

Ask the user for:
- **Provider name** (lowercase, used as registry key and `provider/model` prefix)
- **SDK package name** on PyPI and minimum version
- **Default model ID**
- **Authentication** — API key env var name, endpoint URL, or both
- **Model pricing** — cost per 1K tokens for prompt/completion, or `0.0` for free/local
- **Lazy or eager import** — lazy if SDK is optional, eager if it's in `install_requires`

## Files to Create or Modify (7 total)

### 1. NEW: `prompture/drivers/{provider}_driver.py`

See [references/driver-template.md](references/driver-template.md) for the full skeleton.

Key rules:
- Subclass `Driver` from `..driver`
- `generate()` returns `{"text": str, "meta": dict}`
- `meta` MUST contain: `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost`, `raw_response`, `model_name`
- Optional SDK: wrap import in try/except, raise `ImportError` pointing to `pip install prompture[{provider}]`

### 2. `prompture/drivers/__init__.py`

- Add import: `from .{provider}_driver import {Provider}Driver`
- Add to `DRIVER_REGISTRY`:
  ```python
  "{provider}": lambda model=None: {Provider}Driver(
      api_key=settings.{provider}_api_key,
      model=model or settings.{provider}_model
  ),
  ```
- Add `"{Provider}Driver"` to `__all__`

### 3. `prompture/__init__.py`

- Add `{Provider}Driver` to the `.drivers` import line
- Add `"{Provider}Driver"` to `__all__` under `# Drivers`

### 4. `prompture/settings.py`

Add inside `Settings` class:
```python
# {Provider}
{provider}_api_key: Optional[str] = None
{provider}_model: str = "default-model"
```

### 5. `setup.py`

If optional: add `"{provider}": ["{sdk}>={version}"]` to `extras_require`.
If required: add to `install_requires`.

### 6. `.env.copy`

Add section:
```
# {Provider} Configuration
{PROVIDER}_API_KEY=your-api-key-here
{PROVIDER}_MODEL=default-model
```

### 7. `CLAUDE.md`

Add `{provider}` to the driver list in the Module Layout bullet.

## Verification

```bash
python -c "from prompture import {Provider}Driver; print('OK')"
python -c "from prompture.drivers import get_driver_for_model; d = get_driver_for_model('{provider}/test'); print(d.model)"
pytest tests/ -x -q
```
