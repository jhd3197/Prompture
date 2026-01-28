"""Callback hooks for driver-level observability.

Provides :class:`DriverCallbacks`, a lightweight container for functions
that are invoked before/after every driver call, giving full visibility
into request/response payloads and errors without modifying driver code.

Usage::

    from prompture import DriverCallbacks

    def log_request(info: dict) -> None:
        print(f"-> {info['driver']} prompt length={len(info.get('prompt', ''))}")

    def log_response(info: dict) -> None:
        print(f"<- {info['driver']} {info['elapsed_ms']:.0f}ms")

    callbacks = DriverCallbacks(on_request=log_request, on_response=log_response)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Type aliases for callback signatures.
# Each callback receives a single ``dict[str, Any]`` payload and returns nothing.
OnRequestCallback = Callable[[dict[str, Any]], None]
OnResponseCallback = Callable[[dict[str, Any]], None]
OnErrorCallback = Callable[[dict[str, Any]], None]


@dataclass
class DriverCallbacks:
    """Optional callbacks fired around every driver call.

    Payload shapes:

    ``on_request``
        ``{prompt, messages, options, driver}``

    ``on_response``
        ``{text, meta, driver, elapsed_ms}``

    ``on_error``
        ``{error, prompt, messages, options, driver}``
    """

    on_request: OnRequestCallback | None = field(default=None)
    on_response: OnResponseCallback | None = field(default=None)
    on_error: OnErrorCallback | None = field(default=None)
