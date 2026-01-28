# Skill: Add a New LLM Driver

When the user asks to add a new driver (provider), follow this checklist exactly. Ask the user for any values marked **[ASK]** before writing code.

## Information to Gather

- **Provider name** (lowercase, used as registry key and `provider/model` prefix): [ASK]
- **SDK package name** on PyPI (e.g. `openai`, `anthropic`): [ASK]
- **Minimum SDK version**: [ASK]
- **Default model ID**: [ASK]
- **Authentication**: API key env var name, or endpoint URL, or both: [ASK]
- **Model pricing**: dict of model names to `{"prompt": cost_per_1k, "completion": cost_per_1k}`, or `0.0` for free/local: [ASK]
- **Lazy or eager import**: Use lazy import (try/except inside methods) if the SDK is an optional dependency. Use eager import if the SDK is in `install_requires`.

## Files to Touch (in order)

### 1. `prompture/drivers/{provider}_driver.py` (NEW)

Follow this exact skeleton — match the style of existing drivers:

```python
import os
import logging
from ..driver import Driver
from typing import Any, Dict

logger = logging.getLogger(__name__)


class {Provider}Driver(Driver):
    MODEL_PRICING = {
        # "model-name": {"prompt": 0.00, "completion": 0.00},
        "default": {"prompt": 0.0, "completion": 0.0},
    }

    def __init__(self, api_key: str | None = None, model: str = "default-model"):
        self.api_key = api_key or os.getenv("{PROVIDER}_API_KEY")
        self.model = model
        self.options: Dict[str, Any] = {}

    def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        # --- provider-specific call here ---

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": round(total_cost, 6),
            "raw_response": raw,
            "model_name": self.model,
        }
        return {"text": text, "meta": meta}
```

Key rules:
- Subclass `Driver` from `..driver`.
- `generate()` returns `{"text": str, "meta": dict}`.
- `meta` MUST contain: `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost`, `raw_response`, `model_name`.
- If the SDK is optional, wrap its import inside `generate()` or `__init__` in try/except and raise a clear `ImportError` pointing to `pip install prompture[{provider}]`.

### 2. `prompture/drivers/__init__.py`

- Add import at top: `from .{provider}_driver import {Provider}Driver`
- Add entry to `DRIVER_REGISTRY`:
```python
"{provider}": lambda model=None: {Provider}Driver(
    api_key=settings.{provider}_api_key,
    model=model or settings.{provider}_model
),
```
- Add `"{Provider}Driver"` to `__all__`.

### 3. `prompture/__init__.py`

- Add `{Provider}Driver` to the import line from `.drivers`.
- Add `"{Provider}Driver"` to `__all__` in the `# Drivers` section.

### 4. `prompture/settings.py`

Add settings fields inside the `Settings` class, following the existing pattern:

```python
# {Provider}
{provider}_api_key: Optional[str] = None
{provider}_model: str = "default-model"
```

Add endpoint fields too if the provider uses a configurable URL.

### 5. `setup.py`

If the SDK is an optional dependency, add to `extras_require`:
```python
"{provider}": ["{sdk-package}>={min-version}"],
```

If the SDK should always be installed, add to `install_requires` instead.

### 6. `.env.copy`

Add a section at the end:
```
# {Provider} Configuration
{PROVIDER}_API_KEY=your-api-key-here
{PROVIDER}_MODEL=default-model
```

### 7. `CLAUDE.md`

Add `{provider}` to the driver list in the **Module Layout** bullet for `prompture/drivers/`.

## Verification

After all files are written:
1. Run `python -c "from prompture import {Provider}Driver; print('OK')"` to confirm clean import.
2. Run `python -c "from prompture.drivers import get_driver_for_model; d = get_driver_for_model('{provider}/test'); print(d.model)"` to confirm registry resolution.
3. Run `pytest tests/ -x -q` to confirm no regressions.
