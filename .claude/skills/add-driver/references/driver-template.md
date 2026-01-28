# Driver Template

Every Prompture driver follows this skeleton:

```python
import os
import logging
from ..driver import Driver
from typing import Any, Dict

logger = logging.getLogger(__name__)


class {Provider}Driver(Driver):
    MODEL_PRICING = {
        # "model-name": {"prompt": cost_per_1k, "completion": cost_per_1k},
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

        # --- provider-specific API call here ---
        # text = ...
        # prompt_tokens = ...
        # completion_tokens = ...
        # raw = ...

        # Compute cost
        pricing = self.MODEL_PRICING.get(self.model, self.MODEL_PRICING["default"])
        total_cost = (
            (prompt_tokens / 1000) * pricing["prompt"]
            + (completion_tokens / 1000) * pricing["completion"]
        )

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

## Lazy Import Pattern (for optional SDKs)

```python
def __init__(self, ...):
    self._client = None
    # defer import

def _ensure_client(self):
    if self._client is not None:
        return
    try:
        from some_sdk import Client
    except ImportError:
        raise ImportError(
            "The 'some-sdk' package is required. "
            "Install with: pip install prompture[provider]"
        )
    self._client = Client(api_key=self.api_key)
```

## Existing Drivers for Reference

| Driver | File | SDK | Auth |
|--------|------|-----|------|
| OpenAI | `openai_driver.py` | `openai` | API key |
| Claude | `claude_driver.py` | `anthropic` | API key |
| Google | `google_driver.py` | `google-generativeai` | API key |
| Groq | `groq_driver.py` | `groq` | API key |
| Ollama | `ollama_driver.py` | `requests` | Endpoint URL |
| AirLLM | `airllm_driver.py` | `airllm` (lazy) | None (local) |
