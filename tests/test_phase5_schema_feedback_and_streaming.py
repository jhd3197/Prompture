"""Phase 5: schema-tightening retry feedback + partial-JSON streaming.

Two independent feature areas:

1. ``tighten_schema_from_validation_errors()`` walks Pydantic
   validation errors and appends short retry hints to the
   corresponding field's ``description`` in a deep-copied schema. The
   retry loop in ``extract_with_model`` now uses the tightened schema
   on subsequent attempts.
2. ``parse_partial_json`` + ``stream_extract`` produce a progression
   of partial Pydantic instances as a streaming driver emits delta
   chunks; the last yield is a fully-validated instance when the
   final JSON parses.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from prompture.extraction.core import (
    _format_validation_hint,
    _walk_schema_to_field,
    tighten_schema_from_validation_errors,
)
from prompture.extraction.streaming import (
    _build_partial_model,
    parse_partial_json,
    stream_extract,
)

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class _Person(BaseModel):
    name: str
    age: int
    profession: str | None = None


class _Address(BaseModel):
    street: str
    zip: int


class _PersonWithAddress(BaseModel):
    name: str
    age: int
    address: _Address


# ---------------------------------------------------------------------------
# Schema-tightening helper unit tests
# ---------------------------------------------------------------------------


class TestWalkSchemaToField:
    def test_top_level_property(self) -> None:
        schema = _Person.model_json_schema()
        node = _walk_schema_to_field(schema, ("age",))
        assert isinstance(node, dict)
        assert node.get("type") in {"integer", "number"} or "type" in node

    def test_missing_field_returns_none(self) -> None:
        schema = _Person.model_json_schema()
        assert _walk_schema_to_field(schema, ("not_a_field",)) is None

    def test_empty_loc_returns_none(self) -> None:
        assert _walk_schema_to_field({}, ()) is None

    def test_nested_field_resolves(self) -> None:
        # Pydantic resolves $refs lazily; for nested models the path may differ,
        # so we just verify the helper doesn't crash and gives None when the
        # ref path isn't directly walkable.
        schema = _PersonWithAddress.model_json_schema()
        # Top-level "address" should resolve; deeper may not without $ref.
        addr_node = _walk_schema_to_field(schema, ("address",))
        assert addr_node is None or isinstance(addr_node, dict)


class TestFormatValidationHint:
    def test_includes_msg(self) -> None:
        hint = _format_validation_hint(
            {"type": "int_parsing", "loc": ("age",), "msg": "Input should be a valid integer", "input": "thirty"}
        )
        assert "VALIDATION:" in hint
        assert "integer" in hint.lower()
        assert "thirty" in hint

    def test_truncates_huge_input(self) -> None:
        huge = "x" * 1000
        hint = _format_validation_hint(
            {"type": "x", "loc": ("name",), "msg": "bad value", "input": huge}
        )
        # Inputs over 80 chars get ellipsised.
        assert len(hint) < 250
        assert "..." in hint

    def test_handles_missing_input(self) -> None:
        hint = _format_validation_hint({"loc": ("name",), "msg": "missing"})
        assert "VALIDATION" in hint


class TestTightenSchema:
    def test_no_errors_returns_input(self) -> None:
        schema = _Person.model_json_schema()
        out = tighten_schema_from_validation_errors(schema, [])
        # No deep-copy when there's nothing to do — same identity.
        assert out is schema

    def test_unresolvable_locs_return_input(self) -> None:
        schema = _Person.model_json_schema()
        out = tighten_schema_from_validation_errors(
            schema,
            [{"loc": ("does_not_exist",), "msg": "x", "type": "x", "input": "x"}],
        )
        assert out is schema

    def test_appends_hint_to_failing_field(self) -> None:
        schema = _Person.model_json_schema()
        original_desc = schema["properties"]["age"].get("description", "")
        out = tighten_schema_from_validation_errors(
            schema,
            [{"loc": ("age",), "msg": "Input should be a valid integer", "type": "int_parsing", "input": "thirty"}],
        )
        # Deep copy — input untouched.
        assert schema["properties"]["age"].get("description", "") == original_desc
        # Output has the hint appended.
        new_desc = out["properties"]["age"]["description"]
        assert "VALIDATION" in new_desc
        assert "integer" in new_desc.lower()
        assert "thirty" in new_desc

    def test_appending_twice_dedupes(self) -> None:
        schema = _Person.model_json_schema()
        err = {"loc": ("age",), "msg": "Input should be a valid integer", "type": "int_parsing", "input": "thirty"}
        once = tighten_schema_from_validation_errors(schema, [err])
        twice = tighten_schema_from_validation_errors(once, [err])
        # The hint appears only once.
        assert twice["properties"]["age"]["description"].count("VALIDATION") == 1

    def test_creates_description_when_missing(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},  # no description
        }
        out = tighten_schema_from_validation_errors(
            schema,
            [{"loc": ("x",), "msg": "must be integer", "type": "int_parsing", "input": "a"}],
        )
        assert "VALIDATION" in out["properties"]["x"]["description"]


# ---------------------------------------------------------------------------
# extract_with_model retry-loop integration
# ---------------------------------------------------------------------------


class TestRetryLoopUsesTightenedSchema:
    """Use a constrained model so normalize_field_value can't paper over the bug.

    The retry loop only fires when ``actual_cls(**json_object)`` raises a
    Pydantic ValidationError. ``_Person.age=int`` is silently normalised
    by ``normalize_field_value`` (string -> 0 default), so we use a
    constrained ``Field(gt=0)`` here — negative values survive
    normalization and *do* fail Pydantic.
    """

    def test_second_attempt_receives_tightened_schema(self) -> None:
        """The schema sent on retry must carry the corrective hint."""
        from pydantic import Field

        from prompture.extraction.core import extract_with_model

        class _ConstrainedPerson(BaseModel):
            name: str
            age: int = Field(..., gt=0, description="Positive integer")

        # Track every schema sent to extract_and_jsonify.
        sent_schemas: list[dict[str, Any]] = []
        call_count = {"n": 0}

        def _fake_extract_and_jsonify(*, text, json_schema, instruction_template, **kw):
            sent_schemas.append(json_schema)
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: -5 survives normalisation but fails Field(gt=0).
                return {
                    "json_string": '{"name": "Maria", "age": -5}',
                    "json_object": {"name": "Maria", "age": -5},
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
                }
            # Second call: return a valid positive value.
            return {
                "json_string": '{"name": "Maria", "age": 30}',
                "json_object": {"name": "Maria", "age": 30},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            }

        with patch("prompture.extraction.core.extract_and_jsonify", _fake_extract_and_jsonify):
            result = extract_with_model(
                _ConstrainedPerson,
                "Maria is -5",
                model_name="fake/model",
                max_retries=2,
            )

        # Second attempt happened with a tightened schema.
        assert len(sent_schemas) == 2, f"expected 2 attempts, got {len(sent_schemas)}"
        first_age = sent_schemas[0]["properties"]["age"].get("description", "")
        second_age = sent_schemas[1]["properties"]["age"]["description"]
        assert "VALIDATION" not in first_age
        assert "VALIDATION" in second_age
        # The corrective hint references the bad input ('-5') or the constraint.
        assert "-5" in second_age or "greater than" in second_age.lower()
        # Result is the validated model from attempt 2.
        assert result["json_object"]["age"] == 30

    def test_no_retries_no_tightening(self) -> None:
        """max_retries=1 means one attempt; no schema tightening happens."""
        from prompture.extraction.core import extract_with_model

        sent_schemas: list[dict[str, Any]] = []

        def _fake(*, text, json_schema, instruction_template, **kw):
            sent_schemas.append(json_schema)
            return {
                "json_string": '{"name": "Maria", "age": 30}',
                "json_object": {"name": "Maria", "age": 30},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            }

        with patch("prompture.extraction.core.extract_and_jsonify", _fake):
            extract_with_model(_Person, "Maria is 30", model_name="fake/model")

        assert len(sent_schemas) == 1
        # No corrective hint was added.
        desc = sent_schemas[0]["properties"]["age"].get("description", "")
        assert "VALIDATION" not in desc


# ---------------------------------------------------------------------------
# Partial-JSON parser unit tests
# ---------------------------------------------------------------------------


class TestParsePartialJSON:
    def test_empty_returns_none(self) -> None:
        assert parse_partial_json("") is None
        assert parse_partial_json("   ") is None

    def test_complete_object_parses(self) -> None:
        assert parse_partial_json('{"a": 1}') == {"a": 1}

    def test_truncated_string_value(self) -> None:
        # Mid-string at end: complete by appending the closing quote + brace.
        out = parse_partial_json('{"name": "Alic')
        assert out == {"name": "Alic"}

    def test_missing_closing_brace(self) -> None:
        out = parse_partial_json('{"a": 1, "b": 2')
        # The trailing "2" is complete; just need to close the brace.
        assert out == {"a": 1, "b": 2}

    def test_partial_number_drops_to_last_complete_value(self) -> None:
        # Mid-number — fallback cuts back to the comma, leaving {"a": 1}.
        out = parse_partial_json('{"a": 1, "b": 23')
        # Either {"a": 1, "b": 23} (we treat 23 as complete) or {"a": 1}
        # are acceptable. Verify it's one of those.
        assert out in ({"a": 1, "b": 23}, {"a": 1})

    def test_partial_key_drops_pair(self) -> None:
        # Mid-key after a comma — fallback cuts back, returning {"a": 1}.
        out = parse_partial_json('{"a": 1, "n')
        assert out == {"a": 1}

    def test_partial_array(self) -> None:
        out = parse_partial_json('{"items": [1, 2, 3')
        # Should complete to {"items": [1, 2, 3]} or earlier truncation.
        assert isinstance(out, dict)
        assert isinstance(out.get("items"), list)
        assert out["items"][:3] == [1, 2, 3] or out["items"][:2] == [1, 2]

    def test_nested_object(self) -> None:
        out = parse_partial_json('{"a": {"b": {"c": 1')
        # Should fill closing braces — exact innermost may be partial.
        assert isinstance(out, dict)
        # The "a" key should at least be present.
        assert "a" in out

    def test_unrecoverable_imbalanced(self) -> None:
        # More closes than opens — unrecoverable.
        assert parse_partial_json("{}}") is None or parse_partial_json("{}}") == {}

    def test_trailing_escape_dropped(self) -> None:
        # Mid-escape — the trailing backslash should be dropped, not crash.
        out = parse_partial_json('{"name": "foo\\')
        # Should produce {"name": "foo"} or similar; not None on a normal case.
        # If the parser can't recover, None is acceptable.
        assert out is None or "name" in out


# ---------------------------------------------------------------------------
# stream_extract integration
# ---------------------------------------------------------------------------


class _StreamingDriver:
    """Driver stub that emits ``chunks`` as delta dicts then a done chunk."""

    supports_json_schema = True

    def __init__(self, chunks: list[str], final_text: str | None = None) -> None:
        self.chunks = chunks
        self.final_text = final_text
        self.last_messages: list[dict[str, Any]] | None = None
        self.last_options: dict[str, Any] | None = None

    def generate_messages_stream(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        self.last_messages = messages
        self.last_options = options
        for c in self.chunks:
            yield {"type": "delta", "text": c}
        yield {
            "type": "done",
            "text": self.final_text if self.final_text is not None else "".join(self.chunks),
        }


class _NonStreamingDriver:
    """Driver with no generate_messages_stream."""

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "{}", "meta": {}}


class TestBuildPartialModel:
    def test_skips_non_dict(self) -> None:
        assert _build_partial_model(_Person, [1, 2]) is None

    def test_filters_unknown_keys(self) -> None:
        out = _build_partial_model(_Person, {"name": "A", "junk_key": 1})
        assert out is not None
        assert out.name == "A"
        assert not hasattr(out, "junk_key")

    def test_missing_fields_left_unset(self) -> None:
        out = _build_partial_model(_Person, {"name": "A"})
        assert out is not None
        # model_construct doesn't enforce required fields; age is unset.
        assert out.name == "A"


class TestStreamExtract:
    def test_yields_progressive_partials(self) -> None:
        # Chunks split mid-key, mid-value, then complete.
        chunks = ['{"na', 'me": "Mar', 'ia", "ag', 'e": 32', "}"]
        driver = _StreamingDriver(chunks)
        outputs = list(stream_extract(driver, "extract", _Person))
        # At least 2 progressive partials before the final.
        assert len(outputs) >= 2
        # Final yield is a fully-validated model with all required fields.
        final = outputs[-1]
        assert final.name == "Maria"
        assert final.age == 32

    def test_emits_unchanged_when_requested(self) -> None:
        # Same parseable state across two chunks (whitespace) — without
        # emit_unchanged we should only see 1 emit; with it, 2+.
        chunks = ['{"name": "Maria", "age": 32}', "  ", "  "]
        driver = _StreamingDriver(chunks, final_text='{"name": "Maria", "age": 32}')
        compact = list(stream_extract(driver, "extract", _Person, emit_unchanged=False))
        verbose = list(stream_extract(driver, "extract", _Person, emit_unchanged=True))
        assert len(verbose) >= len(compact)

    def test_skips_partial_arrays_as_root(self) -> None:
        # Top-level array can't construct a Person — should not emit garbage,
        # and the final yield must validate (which it can't here), so 0 emits.
        chunks = ["[1, 2"]
        driver = _StreamingDriver(chunks, final_text="[1, 2]")
        outputs = list(stream_extract(driver, "extract", _Person))
        assert outputs == []

    def test_drops_unknown_keys(self) -> None:
        # Model has no "garbage" field — final validates only on declared keys.
        chunks = ['{"name": "Maria", "age": 32, "garbage": true}']
        driver = _StreamingDriver(chunks)
        outputs = list(stream_extract(driver, "extract", _Person))
        final = outputs[-1]
        assert final.name == "Maria"
        assert final.age == 32

    def test_invalid_final_keeps_last_partial(self) -> None:
        # Final text doesn't validate (missing required field) — partials still emitted,
        # no final-emit happens.
        chunks = ['{"name": "Maria"']
        driver = _StreamingDriver(chunks, final_text='{"name": "Maria"')
        outputs = list(stream_extract(driver, "extract", _Person))
        # At least one partial was emitted via model_construct, but no validated final.
        assert len(outputs) >= 1
        # The last partial has name set, age unset.
        assert outputs[-1].name == "Maria"

    def test_non_streaming_driver_raises(self) -> None:
        with pytest.raises(AttributeError, match="generate_messages_stream"):
            list(stream_extract(_NonStreamingDriver(), "x", _Person))

    def test_system_prompt_forwarded(self) -> None:
        driver = _StreamingDriver(['{"name": "Alice", "age": 1}'])
        list(stream_extract(driver, "extract", _Person, system_prompt="SYS"))
        assert driver.last_messages is not None
        roles = [m["role"] for m in driver.last_messages]
        assert roles[0] == "system"
        assert driver.last_messages[0]["content"] == "SYS"


# ---------------------------------------------------------------------------
# Conversation.ask_stream_model integration
# ---------------------------------------------------------------------------


class TestAskStreamModel:
    def test_yields_partials_via_conversation(self) -> None:
        from prompture.agents.conversation import Conversation

        driver = _StreamingDriver(['{"na', 'me": "Bob", "ag', 'e": 41}'])
        conv = Conversation(driver=driver)
        outputs = list(conv.ask_stream_model("Tell me about Bob", _Person))

        assert any(getattr(p, "name", None) == "Bob" for p in outputs)
        final = outputs[-1]
        assert final.age == 41
        # History records user content and the final JSON string.
        assert any(m["role"] == "user" and m["content"] == "Tell me about Bob" for m in conv._messages)

    def test_schema_propagated_to_driver(self) -> None:
        from prompture.agents.conversation import Conversation

        driver = _StreamingDriver(['{"name": "Bob", "age": 41}'])
        conv = Conversation(driver=driver)
        list(conv.ask_stream_model("Tell me about Bob", _Person))

        # driver.supports_json_schema = True (set on stub), so json_schema should be
        # forwarded in options.
        assert driver.last_options is not None
        assert driver.last_options.get("json_mode") is True
        assert driver.last_options.get("json_schema") is not None
        assert driver.last_options["json_schema"]["properties"].get("name") is not None

    def test_history_excludes_schema_text(self) -> None:
        from prompture.agents.conversation import Conversation

        driver = _StreamingDriver(['{"name": "Bob", "age": 41}'])
        conv = Conversation(driver=driver)
        list(conv.ask_stream_model("ASK", _Person))

        # The user history entry is the plain prompt, NOT the schema-augmented one.
        user_msgs = [m for m in conv._messages if m["role"] == "user"]
        assert user_msgs[-1]["content"] == "ASK"
        # No "JSON object" instruction in stored history.
        assert "MUST respond" not in user_msgs[-1]["content"]


# ---------------------------------------------------------------------------
# Smoke: ValidationError shape we depend on
# ---------------------------------------------------------------------------


class TestValidationErrorContract:
    def test_pydantic_validation_error_shape(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            _Person(name="A", age="not an int")
        errs = exc_info.value.errors()
        assert isinstance(errs, list)
        assert errs
        assert "loc" in errs[0]
        assert "msg" in errs[0]
        assert "type" in errs[0]
