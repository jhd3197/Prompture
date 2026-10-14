"""One reasoning-effort option ("effort") reaches every provider in its own field."""

from __future__ import annotations

import pytest

from prompture.drivers._effort import apply_effort, google_thinking_level, requested_effort
from prompture.drivers._openai_responses import build_request
from prompture.drivers.claude_driver import _apply_anthropic_reporting_options
from prompture.drivers.openai_driver import _apply_openai_reporting_options


def test_effort_and_reasoning_effort_are_the_same_request():
    assert requested_effort({"effort": "high"}) == "high"
    assert requested_effort({"reasoning_effort": "low"}) == "low"
    assert requested_effort({"effort": "max", "reasoning_effort": "low"}) == "max"
    assert requested_effort({}) is None
    assert requested_effort(None) is None


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("openai", {"reasoning_effort": "xhigh"}),
        ("openrouter", {"reasoning": {"effort": "xhigh"}}),
        ("google", {"thinking_config": {"thinking_level": "high"}}),
    ],
)
def test_each_style_puts_effort_where_its_api_reads_it(style, expected):
    request: dict = {}
    apply_effort(request, {"effort": "xhigh"}, style)
    assert request == expected
    untouched: dict = {}
    apply_effort(untouched, {"temperature": 0}, style)
    assert untouched == {}


def test_an_explicit_provider_field_wins():
    request = {"reasoning_effort": "low"}
    apply_effort(request, {"effort": "max"}, "openai")
    assert request == {"reasoning_effort": "low"}
    with pytest.raises(ValueError):
        apply_effort({}, {"effort": "low"}, "carrier-pigeon")


def test_gemini_has_two_levels():
    assert [google_thinking_level(e) for e in ("none", "low", "medium", "high", "xhigh", "max")] == [
        "low",
        "low",
        "low",
        "high",
        "high",
        "high",
    ]


def test_claude_takes_openai_s_name_for_it():
    kwargs: dict = {}
    _apply_anthropic_reporting_options(kwargs, {"reasoning_effort": "medium"})
    assert kwargs["extra_body"]["output_config"]["effort"] == "medium"


def test_openai_takes_the_neutral_name():
    kwargs: dict = {}
    _apply_openai_reporting_options(kwargs, {"effort": "high"})
    assert kwargs == {"reasoning_effort": "high"}


def test_the_responses_api_takes_it_as_reasoning_effort():
    class FakeDriver:
        model = "gpt-6-luna"

    request = build_request(FakeDriver(), [{"role": "user", "content": "hi"}], {"effort": "low"})
    assert request["reasoning"] == {"effort": "low"}
