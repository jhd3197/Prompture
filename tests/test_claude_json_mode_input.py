"""Forced-tool JSON mode: Claude sometimes string-encodes a structured field.

Observed with claude-sonnet-5 and a schema whose only property is an array of
objects (a dubbing translator's batch): the tool input came back as
``{"translations": "<the whole answer as a JSON string>"}`` three times out of
three, so every caller validating against its schema rejected a correct answer.
"""

from __future__ import annotations

import json

from prompture.drivers.claude_driver import _json_mode_input

SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
        "note": {"type": "string"},
        "meta": {"type": "object", "properties": {"n": {"type": "integer"}}},
    },
}
ITEMS = [{"segment_id": 1, "text": "Oye, ¿han visto mi auto?"}]


def test_the_whole_answer_wrapped_in_its_first_field_is_unwrapped():
    wrapped = {"translations": json.dumps({"translations": ITEMS}, ensure_ascii=False)}
    assert _json_mode_input(wrapped, SCHEMA) == {"translations": ITEMS}


def test_a_string_encoded_array_is_decoded():
    assert _json_mode_input({"translations": json.dumps(ITEMS)}, SCHEMA) == {"translations": ITEMS}


def test_a_string_encoded_object_is_decoded_but_not_unwrapped():
    meta = {"n": 2, "meta": "kept"}
    assert _json_mode_input({"meta": json.dumps(meta)}, SCHEMA) == {"meta": meta}


def test_well_formed_input_and_string_fields_are_untouched():
    good = {"translations": ITEMS, "note": '["looks", "like", "json"]'}
    assert _json_mode_input(good, SCHEMA) == good


def test_undecodable_strings_and_missing_schema_pass_through():
    assert _json_mode_input({"translations": "not json"}, SCHEMA) == {"translations": "not json"}
    assert _json_mode_input({"translations": "[]"}, None) == {"translations": "[]"}
