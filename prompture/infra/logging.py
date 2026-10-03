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

import contextlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from .._internal.json_encoder import PromptureJSONEncoder


class SecretScrubbingFilter(logging.Filter):
    """Scrub URL credentials, API keys and bearer tokens from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        from ..security.redaction import scrub_secrets

        try:
            message = record.getMessage()
        except Exception:  # malformed %-args: leave the record alone
            return True
        scrubbed = scrub_secrets(message)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = None
        # Formatters append the traceback after the message, and exception
        # text often carries the very URL or key we just scrubbed. Render it
        # now and store the scrubbed copy; formatters reuse ``exc_text``.
        if record.exc_info and not record.exc_text:
            with contextlib.suppress(Exception):
                record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = scrub_secrets(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub_secrets(record.stack_info)
        return True


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
        return json.dumps(payload, cls=PromptureJSONEncoder, ensure_ascii=False)


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
    if not any(isinstance(f, SecretScrubbingFilter) for f in handler.filters):
        handler.addFilter(SecretScrubbingFilter())

    # Avoid adding duplicate handlers when called multiple times.
    logger.handlers = [h for h in logger.handlers if h is not handler]
    logger.addHandler(handler)
