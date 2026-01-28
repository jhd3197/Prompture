"""Tests for prompture.logging module."""

from __future__ import annotations

import json
import logging

from prompture.logging import JSONFormatter, configure_logging


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_format(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="prompture.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "prompture.test"
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed

    def test_format_with_data(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="prompture.test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="with data",
            args=(),
            exc_info=None,
        )
        record.prompture_data = {"key": "value", "count": 42}
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["data"] == {"key": "value", "count": 42}

    def test_format_without_data(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="prompture.test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="no data",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "data" not in parsed


class TestConfigureLogging:
    """Tests for configure_logging."""

    def teardown_method(self):
        # Clean up handlers after each test
        logger = logging.getLogger("prompture")
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def test_default_configuration(self):
        configure_logging(logging.DEBUG)
        logger = logging.getLogger("prompture")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_json_format(self):
        configure_logging(logging.INFO, json_format=True)
        logger = logging.getLogger("prompture")
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    def test_custom_handler(self):
        handler = logging.StreamHandler()
        configure_logging(logging.WARNING, handler=handler)
        logger = logging.getLogger("prompture")
        assert handler in logger.handlers

    def test_no_duplicate_handlers(self):
        """Calling configure_logging twice with same handler shouldn't duplicate."""
        handler = logging.StreamHandler()
        configure_logging(logging.DEBUG, handler=handler)
        configure_logging(logging.DEBUG, handler=handler)
        logger = logging.getLogger("prompture")
        count = sum(1 for h in logger.handlers if h is handler)
        assert count == 1
