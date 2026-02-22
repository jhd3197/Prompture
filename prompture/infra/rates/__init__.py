"""Per-provider model capabilities loaded from JSON files.

Each ``*.json`` file in this directory represents one models.dev provider
(e.g. ``openai.json`` → provider ``"openai"``).  The JSON maps model IDs to
capability dicts which are converted into :class:`ModelCapabilities` instances
at import time.
"""

import json
from pathlib import Path

from ..model_rates import ModelCapabilities

_RATES_DIR = Path(__file__).resolve().parent


def _load_capabilities() -> dict[tuple[str, str], ModelCapabilities]:
    """Load all ``*.json`` files and build the capabilities KB."""
    kb: dict[tuple[str, str], ModelCapabilities] = {}
    for json_file in sorted(_RATES_DIR.glob("*.json")):
        provider = json_file.stem
        raw: dict[str, dict] = json.loads(json_file.read_text(encoding="utf-8"))
        for model_id, entry in raw.items():
            kb[(provider, model_id)] = ModelCapabilities(
                supports_temperature=entry.get("supports_temperature"),
                supports_tool_use=entry.get("supports_tool_use"),
                supports_structured_output=entry.get("supports_structured_output"),
                supports_vision=entry.get("supports_vision"),
                is_reasoning=entry.get("is_reasoning"),
                context_window=entry.get("context_window"),
                max_output_tokens=entry.get("max_output_tokens"),
                modalities_input=tuple(entry.get("modalities_input", ())),
                modalities_output=tuple(entry.get("modalities_output", ())),
                api_type=entry.get("api_type"),
            )
    return kb


CAPABILITIES_KB: dict[tuple[str, str], ModelCapabilities] = _load_capabilities()
