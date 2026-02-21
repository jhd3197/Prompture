"""Prompture exception hierarchy.

Provides a structured set of exceptions for different failure modes.
All exceptions inherit from :class:`PromptureError` which itself
inherits from ``Exception``.  Where appropriate, exceptions also
inherit from the stdlib exception they replace (e.g. ``DriverError``
extends ``NotImplementedError``) for backward compatibility.
"""

from typing import Any


class PromptureError(Exception):
    """Base exception for all Prompture errors."""


class DriverError(PromptureError, NotImplementedError):
    """Raised when a driver operation fails or is not implemented."""


class ExtractionError(PromptureError, ValueError):
    """Raised when JSON extraction or parsing fails."""


class ConfigurationError(PromptureError, ValueError):
    """Raised for invalid configuration (missing keys, bad settings)."""


class ValidationError(PromptureError):
    """Raised when extracted data fails validation (not Pydantic's own)."""


class BudgetExceededError(PromptureError, RuntimeError):
    """Raised when a cost or token budget has been exceeded."""

    def __init__(self, message: str = "", *, budget_state: Any = None) -> None:
        self.budget_state = budget_state
        super().__init__(message)
