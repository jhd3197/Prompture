"""Async driver base class for LLM adapters."""

from __future__ import annotations

from typing import Any

from .driver import Driver


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
    supports_messages: bool = False

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        """Generate a response from a list of conversation messages (async).

        Default implementation flattens the messages into a single prompt
        and delegates to :meth:`generate`.  Drivers that natively support
        message arrays should override this and set
        ``supports_messages = True``.
        """
        prompt = Driver._flatten_messages(messages)
        return await self.generate(prompt, options)

    # Re-export the static helper for convenience
    _flatten_messages = staticmethod(Driver._flatten_messages)
