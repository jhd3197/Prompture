"""Where Ollama reads a request's sampling and length controls.

Ollama's /api/chat takes temperature, top_p, top_k, the reply length
(``num_predict``) and the context size inside an ``options`` object; at the top
level they are silently ignored. ``think`` (reasoning models) is top level.
Without this a ``max_tokens`` never reached the model, and a reasoning model
asked for structured output could think until the request timed out.
"""

from __future__ import annotations

from typing import Any

_SAMPLING = ("temperature", "top_p", "top_k", "num_ctx", "seed", "repeat_penalty", "min_p")


def apply_sampling(payload: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """Copy sampling and length controls from `options` into `payload` the way
    Ollama expects them. Returns the payload."""
    sampling = {key: options[key] for key in _SAMPLING if options.get(key) is not None}
    limit = options.get("num_predict", options.get("max_tokens"))
    if limit is not None:
        sampling["num_predict"] = int(limit)
    if sampling:
        payload["options"] = {**(payload.get("options") or {}), **sampling}
    if options.get("think") is not None:
        payload["think"] = options["think"]
    return payload
