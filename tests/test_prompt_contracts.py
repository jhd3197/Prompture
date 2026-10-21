"""Small prompt contracts: JSON repair, faithfulness verdicts, deferred tool loading."""

from __future__ import annotations

from typing import Any

from prompture.drivers.base import Driver
from prompture.eval.faithfulness import _parse_verdict
from prompture.extraction.core import clean_json_text_with_ai, json_repair_prompt


class _RecordingDriver(Driver):
    def __init__(self, reply: str) -> None:
        self.model = "mock/recorder"
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        return {"text": self.reply, "meta": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class TestJsonRepair:
    def test_prompt_carries_schema_and_error(self):
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        prompt = json_repair_prompt('{"age": 3,', json_schema=schema, error="Expecting property name")
        assert "Parser error: Expecting property name" in prompt
        assert '"age": {"type": "integer"}' in prompt
        assert '<malformed>\n{"age": 3,\n</malformed>' in prompt
        assert "do not invent values" in prompt

    def test_prompt_without_schema_omits_it(self):
        assert "schema" not in json_repair_prompt("{oops")

    def test_cleanup_sends_schema(self):
        driver = _RecordingDriver('{"age": 3}')
        fixed, _ = clean_json_text_with_ai(driver, '{"age": 3,', json_schema={"type": "object"}, error="bad")
        assert fixed == '{"age": 3}'
        assert '{"type": "object"}' in driver.prompts[0]

    def test_valid_json_skips_the_call(self):
        driver = _RecordingDriver("{}")
        assert clean_json_text_with_ai(driver, '{"a": 1}')[0] == '{"a": 1}'
        assert driver.prompts == []


class TestFaithfulnessVerdict:
    def test_bare_label_still_parses(self):
        assert _parse_verdict("SUPPORTED") == "SUPPORTED"

    def test_last_label_wins_over_quote(self):
        text = "QUOTE: the plan was not supported by the board\nCONTRADICTED"
        assert _parse_verdict(text) == "CONTRADICTED"

    def test_label_without_evidence_is_silent(self):
        assert _parse_verdict("QUOTE: NONE\nSUPPORTED") == "SILENT"

    def test_missing_label(self):
        assert _parse_verdict("QUOTE: something") is None


def test_load_tools_description_says_to_load_before_calling():
    from prompture.agents.tool_context import ToolLoader
    from prompture.agents.tools_schema import ToolRegistry

    registry = ToolRegistry()

    def lookup(q: str) -> str:
        """Look something up."""
        return q

    registry.register(lookup)
    loader = ToolLoader(defer=True)
    tools = {t.name: t for t in loader.meta_tools(lambda: registry)}
    assert "instead of guessing" in tools["load_tools"].description
    assert "lookup" in tools["search_tools"].description
