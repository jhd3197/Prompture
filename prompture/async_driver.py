"""Async driver base class for LLM adapters."""

from __future__ import annotations

from typing import Any


class AsyncDriver:
    """Async adapter base. Implement ``async generate(prompt, options)``
    returning ``{"text": ..., "meta": {...}}``.

    The ``meta`` dict follows the same contract as :class:`Driver`:

    .. code-block:: python

        {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "cost": float,
            "raw_response": dict,
        }
    """

    supports_json_mode: bool = False
    supports_json_schema: bool = False

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
