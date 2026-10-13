"""Debug logging must not raise when a provider reports a metric as null.

A rotating file handler formats a record while deciding whether to roll
over, outside logging's own error guard, so a bad %-format reached the
caller and failed the generation.
"""

import logging

from prompture.drivers.base import Driver


class NullCostDriver(Driver):
    def generate(self, prompt, options):
        return {"text": "ok", "meta": {"cost": None, "total_tokens": None}}

    def generate_messages(self, messages, options):
        return self.generate("", options)


class Formatting(logging.Handler):
    """Formats every record eagerly, as a rotating file handler does."""

    def emit(self, record):
        self.format(record)


def test_null_metrics_are_logged_without_raising():
    logger = logging.getLogger("prompture.driver")
    handler = Formatting(level=logging.DEBUG)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        resp = NullCostDriver().generate_messages_with_hooks([{"role": "user", "content": "hi"}], {})
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    assert resp["text"] == "ok"
