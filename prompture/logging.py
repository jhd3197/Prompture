"""Logging configuration for the Prompture library.

Provides a structured JSON formatter and a convenience function for users
to enable Prompture's internal logging with a single call.

Usage::

    from prompture import configure_logging
    import logging

    # Simple: enable DEBUG-level output to stderr
    configure_logging(logging.DEBUG)

    # Structured JSON lines (useful for log aggregation)
    configure_logging(logging.DEBUG, json_format=True)

    # Provide your own handler
    fh = logging.FileHandler("prompture.log")
    configure_logging(logging.INFO, handler=fh)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Fields always present: ``timestamp``, ``level``, ``logger``, ``message``.
    If the caller passes ``extra={"prompture_data": ...}`` the value is
    included under the ``data`` key.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        data = getattr(record, "prompture_data", None)
        if data is not None:
            payload["data"] = data
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: int = logging.DEBUG,
    handler: logging.Handler | None = None,
    json_format: bool = False,
) -> None:
    """Set up Prompture's library logger for application-level visibility.

    Args:
        level: Minimum severity to emit (e.g. ``logging.DEBUG``).
        handler: Custom :class:`logging.Handler`.  When *None*, a
            :class:`logging.StreamHandler` writing to *stderr* is created.
        json_format: When *True*, messages are formatted as JSON lines
            via :class:`JSONFormatter`.
    """
    logger = logging.getLogger("prompture")
    logger.setLevel(level)

    if handler is None:
        handler = logging.StreamHandler()

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    handler.setLevel(level)

    # Avoid adding duplicate handlers when called multiple times.
    logger.handlers = [h for h in logger.handlers if h is not handler]
    logger.addHandler(handler)
