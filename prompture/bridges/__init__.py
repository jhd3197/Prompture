"""Bridges between Prompture and external frameworks.

.. deprecated::
    This package has been consolidated into ``prompture.infra``.
    Import from ``prompture.infra`` instead.
"""

try:
    from ..infra.tukuy_backend import TukuyLLMBackend, create_tukuy_backend
except ImportError:  # tukuy not installed
    TukuyLLMBackend = None  # type: ignore[assignment,misc]
    create_tukuy_backend = None  # type: ignore[assignment]

__all__ = ["TukuyLLMBackend", "create_tukuy_backend"]
