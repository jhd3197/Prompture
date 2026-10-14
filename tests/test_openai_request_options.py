"""Per-request options the OpenAI driver forwards to ``chat.completions.create``.

``timeout`` used to be dropped: the Responses path and the Claude driver took it, but a chat
completion always waited the SDK's own 10 minutes per try, with retries, on a stalled
connection. A reasoning model at high effort on a dead socket held a caller for half an hour.
"""

from __future__ import annotations

from prompture.drivers.openai_driver import _apply_openai_reporting_options


def test_timeout_reaches_the_request() -> None:
    kwargs: dict = {}
    _apply_openai_reporting_options(kwargs, {"timeout": 900, "reasoning_effort": "xhigh", "temperature": 1})
    assert kwargs == {"timeout": 900, "reasoning_effort": "xhigh"}


def test_nothing_is_added_when_not_asked_for() -> None:
    kwargs: dict = {"model": "gpt-6-astra"}
    _apply_openai_reporting_options(kwargs, {})
    assert kwargs == {"model": "gpt-6-astra"}
