"""Anthropic structured output refuses some JSON Schema keywords; the driver drops them."""

from __future__ import annotations

from pydantic import BaseModel, Field

from prompture.drivers.claude_driver import _structured_output_format


class Hit(BaseModel):
    confidence: float = Field(default=0.5, ge=0, le=1, description="how sure")
    name: str = Field(min_length=1, max_length=40)
    tags: list[str] = Field(default_factory=list, min_length=2)


class Hits(BaseModel):
    hits: list[Hit]


def test_limits_move_from_the_schema_into_the_description():
    fmt = _structured_output_format(Hits.model_json_schema())
    hit = fmt["schema"]["$defs"]["Hit"]["properties"]
    assert "minimum" not in hit["confidence"] and "maximum" not in hit["confidence"]
    assert hit["confidence"]["description"] == "how sure (minimum 0, maximum 1)"
    assert "minLength" not in hit["name"] and "maxLength" not in hit["name"]
    assert "minItems" not in hit["tags"] and "minItems 2" in hit["tags"]["description"]
    assert fmt["schema"]["$defs"]["Hit"]["additionalProperties"] is False


def test_small_min_items_stays():
    schema = {"type": "object", "properties": {"x": {"type": "array", "items": {"type": "string"}, "minItems": 1}}}
    assert _structured_output_format(schema)["schema"]["properties"]["x"]["minItems"] == 1
