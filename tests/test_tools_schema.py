"""Tests for ToolRegistry filtering methods (subset, filter, exclude)."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, Optional, TypedDict, Union

import pytest
from pydantic import BaseModel

from prompture.agents.tools_schema import (
    ToolDefinition,
    ToolRegistry,
    _python_type_to_json_schema,
    tool_from_function,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_registry() -> ToolRegistry:
    """Create a registry with three tools for testing."""
    reg = ToolRegistry()

    @reg.tool
    def file_read(path: str) -> str:
        """Read a file."""
        return path

    @reg.tool
    def file_write(path: str, content: str) -> str:
        """Write a file."""
        return f"{path}: {content}"

    @reg.tool
    def python_execute(code: str) -> str:
        """Execute Python code."""
        return code

    return reg


# ---------------------------------------------------------------------------
# subset()
# ---------------------------------------------------------------------------


class TestSubset:
    def test_subset_returns_correct_tools(self):
        reg = _build_registry()
        sub = reg.subset({"file_read", "python_execute"})
        assert len(sub) == 2
        assert "file_read" in sub
        assert "python_execute" in sub
        assert "file_write" not in sub

    def test_subset_returns_new_registry(self):
        reg = _build_registry()
        sub = reg.subset({"file_read"})
        assert sub is not reg
        assert len(reg) == 3  # original unchanged

    def test_subset_with_list(self):
        reg = _build_registry()
        sub = reg.subset(["file_read"])
        assert len(sub) == 1
        assert "file_read" in sub

    def test_subset_unknown_name_raises_keyerror(self):
        reg = _build_registry()
        with pytest.raises(KeyError, match="Unknown tools"):
            reg.subset({"file_read", "nonexistent"})

    def test_subset_all_unknown_raises_keyerror(self):
        reg = _build_registry()
        with pytest.raises(KeyError, match="Unknown tools"):
            reg.subset({"foo", "bar"})


# ---------------------------------------------------------------------------
# filter()
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filter_with_predicate(self):
        reg = _build_registry()
        sub = reg.filter(lambda td: "file_" in td.name)
        assert len(sub) == 2
        assert "file_read" in sub
        assert "file_write" in sub
        assert "python_execute" not in sub

    def test_filter_returns_new_registry(self):
        reg = _build_registry()
        sub = reg.filter(lambda td: True)
        assert sub is not reg
        assert len(sub) == 3

    def test_filter_no_match(self):
        reg = _build_registry()
        sub = reg.filter(lambda td: False)
        assert len(sub) == 0

    def test_filter_by_description(self):
        reg = _build_registry()
        sub = reg.filter(lambda td: "Execute" in td.description)
        assert len(sub) == 1
        assert "python_execute" in sub


# ---------------------------------------------------------------------------
# exclude()
# ---------------------------------------------------------------------------


class TestExclude:
    def test_exclude_removes_named_tools(self):
        reg = _build_registry()
        sub = reg.exclude({"python_execute"})
        assert len(sub) == 2
        assert "python_execute" not in sub
        assert "file_read" in sub
        assert "file_write" in sub

    def test_exclude_returns_new_registry(self):
        reg = _build_registry()
        sub = reg.exclude({"file_read"})
        assert sub is not reg
        assert len(reg) == 3

    def test_exclude_missing_name_is_silent(self):
        reg = _build_registry()
        sub = reg.exclude({"nonexistent"})
        assert len(sub) == 3

    def test_exclude_all(self):
        reg = _build_registry()
        sub = reg.exclude({"file_read", "file_write", "python_execute"})
        assert len(sub) == 0

    def test_exclude_with_list(self):
        reg = _build_registry()
        sub = reg.exclude(["file_read", "file_write"])
        assert len(sub) == 1
        assert "python_execute" in sub


# ---------------------------------------------------------------------------
# Tool execution on filtered registries
# ---------------------------------------------------------------------------


class TestFilteredExecution:
    def test_subset_tools_are_executable(self):
        reg = _build_registry()
        sub = reg.subset({"file_read"})
        result = sub.execute("file_read", {"path": "/tmp/test"})
        assert result == "/tmp/test"


    def test_exclude_tools_are_executable(self):
        reg = _build_registry()
        sub = reg.exclude({"file_read"})
        result = sub.execute("file_write", {"path": "/tmp/f", "content": "data"})
        assert result == "/tmp/f: data"


# ---------------------------------------------------------------------------
# metadata (host annotations, e.g. is_write for a confirmation gate)
# ---------------------------------------------------------------------------


class TestMetadata:
    def _reg(self) -> ToolRegistry:
        reg = ToolRegistry()

        def read_battery() -> str:
            """Read the battery."""
            return "100%"

        def wipe(path: str) -> str:
            """Delete a path."""
            return f"wiped {path}"

        reg.register(read_battery, metadata={"is_write": False})
        reg.register(wipe, metadata={"is_write": True, "category": "fs"})
        return reg

    def test_metadata_is_stored(self):
        reg = self._reg()
        assert reg.get("wipe").metadata == {"is_write": True, "category": "fs"}

    def test_metadata_defaults_to_empty_and_not_shared(self):
        reg = ToolRegistry()
        a = reg.register(lambda: "a", name="a")
        b = reg.register(lambda: "b", name="b")
        assert a.metadata == {} and b.metadata == {}
        a.metadata["x"] = 1
        assert b.metadata == {}  # no shared mutable default

    def test_filter_by_metadata_preserves_schema_and_exec(self):
        reg = self._reg()
        read_only = reg.filter(lambda td: not td.metadata.get("is_write"))
        assert read_only.names == ["read_battery"]
        assert "wipe" not in read_only
        # schema + metadata survive the filter
        assert reg.get("wipe").parameters["properties"].keys() == {"path"}
        assert read_only.execute("read_battery", {}) == "100%"
# ---------------------------------------------------------------------------
# Unannotated parameters are unconstrained, not silently "string"
# ---------------------------------------------------------------------------


class TestUnannotatedParameters:
    """An absent annotation means "type unknown", so the schema must not
    constrain it.  Claiming ``{"type": "string"}`` both misinforms the model
    and — now that arguments are validated against the schema — rejects every
    correct non-string call."""

    def test_missing_annotation_yields_empty_schema(self):
        import inspect

        assert _python_type_to_json_schema(inspect.Parameter.empty) == {}

    def test_lambda_tool_accepts_the_type_it_actually_wants(self):
        reg = ToolRegistry()
        reg.register(lambda x: x + 1, name="inc", description="Increment")
        assert reg.execute("inc", {"x": 5}) == 6

    def test_unannotated_def_parameter_is_not_typed(self):
        def echo(value):
            """Echo a value back.

            Args:
                value: Anything at all.
            """
            return value

        td = tool_from_function(echo)
        prop = td.parameters["properties"]["value"]
        assert "type" not in prop
        # The description is still carried through for the model's benefit.
        assert prop["description"] == "Anything at all."

    def test_unannotated_parameter_accepts_every_json_type(self):
        reg = ToolRegistry()
        reg.register(lambda value: value, name="echo", description="Echo")
        for payload in (1, "s", 1.5, True, None, [1, 2], {"k": "v"}):
            assert reg.execute("echo", {"value": payload}) == payload

    def test_annotated_parameters_still_validate(self):
        """Loosening unannotated params must not loosen annotated ones."""

        def strict(count: int) -> int:
            """Take an int.

            Args:
                count: A real integer.
            """
            return count

        reg = ToolRegistry()
        reg.register(strict)
        assert reg.execute("strict", {"count": 3}) == 3
        # A coercible string is still accepted...
        assert reg.execute("strict", {"count": "7"}) == 7
        # ...but junk is reported back to the model rather than executed.
        error = reg.execute("strict", {"count": "not-an-int"})
        assert isinstance(error, str)
        assert "expected int" in error


# ---------------------------------------------------------------------------
# Type -> JSON Schema coverage
#
# These annotations all used to collapse to {"type": "string"}, so the schema
# the model was handed disagreed with what the tool actually accepted.
# ---------------------------------------------------------------------------


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Point(BaseModel):
    x: int
    y: int


@dataclass
class Box:
    w: int
    h: int


class Movie(TypedDict):
    title: str
    year: int


class TestPythonTypeToJsonSchema:
    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            (int, {"type": "integer"}),
            (str, {"type": "string"}),
            (float, {"type": "number"}),
            (bool, {"type": "boolean"}),
        ],
    )
    def test_scalars(self, annotation, expected):
        assert _python_type_to_json_schema(annotation) == expected

    def test_any_is_unconstrained(self):
        assert _python_type_to_json_schema(Any) == {}

    def test_unknown_type_falls_back_to_string(self):
        assert _python_type_to_json_schema(object) == {"type": "string"}

    # -- unions ---------------------------------------------------------

    def test_pep604_optional(self):
        assert _python_type_to_json_schema(int | None) == {"anyOf": [{"type": "integer"}, {"type": "null"}]}

    def test_typing_optional_matches_pep604(self):
        # The legacy spelling is the point of this test, so keep it verbatim.
        legacy = Optional[int]  # noqa: UP045
        assert _python_type_to_json_schema(legacy) == _python_type_to_json_schema(int | None)

    def test_union_without_none(self):
        assert _python_type_to_json_schema(Union[int, str]) == {"anyOf": [{"type": "integer"}, {"type": "string"}]}

    def test_multi_member_union_keeps_every_member(self):
        assert _python_type_to_json_schema(str | int | None) == {
            "anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]
        }

    # -- enumerations ---------------------------------------------------

    def test_literal_becomes_enum(self):
        assert _python_type_to_json_schema(Literal["a", "b"]) == {"enum": ["a", "b"], "type": "string"}

    def test_mixed_literal_omits_type(self):
        """A Literal spanning types cannot claim a single JSON type."""
        assert _python_type_to_json_schema(Literal["a", 1]) == {"enum": ["a", 1]}

    def test_enum_class_uses_member_values(self):
        assert _python_type_to_json_schema(Color) == {"enum": ["red", "blue"], "type": "string"}

    # -- containers -----------------------------------------------------

    def test_list_of_scalars(self):
        assert _python_type_to_json_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}

    def test_bare_list_has_no_item_constraint(self):
        assert _python_type_to_json_schema(list) == {"type": "array"}

    def test_homogeneous_tuple(self):
        assert _python_type_to_json_schema(tuple[int, ...]) == {"type": "array", "items": {"type": "integer"}}

    def test_fixed_length_tuple_uses_prefix_items(self):
        assert _python_type_to_json_schema(tuple[int, str]) == {
            "type": "array",
            "prefixItems": [{"type": "integer"}, {"type": "string"}],
            "items": False,
        }

    def test_dict_value_type_becomes_additional_properties(self):
        assert _python_type_to_json_schema(dict[str, int]) == {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        }

    def test_dict_of_any_stays_unconstrained(self):
        assert _python_type_to_json_schema(dict[str, Any]) == {"type": "object"}

    # -- formatted scalars ----------------------------------------------

    @pytest.mark.parametrize(
        ("annotation", "fmt"),
        [(datetime, "date-time"), (date, "date")],
    )
    def test_datetime_scalars_carry_a_format(self, annotation, fmt):
        assert _python_type_to_json_schema(annotation) == {"type": "string", "format": fmt}

    # -- nested structured types ----------------------------------------

    def test_nested_pydantic_model_expands(self):
        schema = _python_type_to_json_schema(Point)
        assert schema["type"] == "object"
        assert set(schema["properties"]) == {"x", "y"}
        assert schema["properties"]["x"]["type"] == "integer"
        assert sorted(schema["required"]) == ["x", "y"]

    def test_nested_dataclass_expands(self):
        schema = _python_type_to_json_schema(Box)
        assert set(schema["properties"]) == {"w", "h"}

    def test_typed_dict_expands(self):
        schema = _python_type_to_json_schema(Movie)
        assert schema["properties"]["year"]["type"] == "integer"

    def test_list_of_models_expands_items(self):
        schema = _python_type_to_json_schema(list[Point])
        assert schema["type"] == "array"
        assert set(schema["items"]["properties"]) == {"x", "y"}


class TestToolFromFunctionRichTypes:
    """End-to-end: a tool advertises a schema matching what it accepts."""

    def test_optional_literal_and_enum_params_are_faithful(self):
        def configure(
            mode: Literal["fast", "slow"],
            color: Color,
            retries: int | None = None,
        ) -> str:
            """Configure something.

            Args:
                mode: How fast to go.
                color: Which colour.
                retries: Optional retry count.
            """
            return f"{mode}/{color.value}/{retries}"

        td = tool_from_function(configure)
        props = td.parameters["properties"]

        assert props["mode"]["enum"] == ["fast", "slow"]
        assert props["color"]["enum"] == ["red", "blue"]
        assert props["retries"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
        # Only the non-defaulted params are required.
        assert td.parameters["required"] == ["mode", "color"]
        # Docstring descriptions survive alongside the richer types.
        assert props["mode"]["description"] == "How fast to go."

    def test_nested_model_param_is_advertised_as_an_object(self):
        def move(target: Point) -> str:
            """Move to a point.

            Args:
                target: Where to go.
            """
            return f"{target.x},{target.y}"

        props = tool_from_function(move).parameters["properties"]
        assert props["target"]["type"] == "object"
        assert set(props["target"]["properties"]) == {"x", "y"}

    def test_dict_param_advertises_its_value_type(self):
        def tally(counts: dict[str, int]) -> int:
            """Sum a mapping.

            Args:
                counts: Name to count.
            """
            return sum(counts.values())

        props = tool_from_function(tally).parameters["properties"]
        assert props["counts"]["additionalProperties"] == {"type": "integer"}

    def test_registry_executes_a_literal_typed_tool(self):
        """The schema is honest, so a valid call runs and an invalid one is
        reported back to the model instead of raising."""

        def pick(mode: Literal["fast", "slow"]) -> str:
            """Pick a mode.

            Args:
                mode: Which mode.
            """
            return f"picked {mode}"

        reg = ToolRegistry()
        reg.register(pick)
        assert reg.execute("pick", {"mode": "fast"}) == "picked fast"
        error = reg.execute("pick", {"mode": "sideways"})
        assert isinstance(error, str)
        assert "sideways" in error

    def test_rich_schema_survives_conversion_to_openai_format(self):
        def configure(mode: Literal["fast", "slow"]) -> str:
            """Configure.

            Args:
                mode: Which mode.
            """
            return mode

        td: ToolDefinition = tool_from_function(configure)
        wire = td.to_openai_format()
        assert wire["function"]["parameters"]["properties"]["mode"]["enum"] == ["fast", "slow"]


# ---------------------------------------------------------------------------
# Structured types when pydantic's TypeAdapter refuses them
#
# pydantic rejects `typing.TypedDict` on Python < 3.12 (it wants the
# typing_extensions variant), so on two of the four supported Python versions
# TypeAdapter raises and the parameter used to be advertised as a bare string.
# The failure is forced here so the fallback is covered on every version.
# ---------------------------------------------------------------------------


class PartialMovie(TypedDict, total=False):
    title: str
    year: int


@dataclass
class BoxWithDefaults:
    w: int
    h: int = 3


class PointWithOptional(BaseModel):
    x: int
    y: int | None = None


@pytest.fixture
def type_adapter_unavailable(monkeypatch):
    """Make TypeAdapter raise, as it does for typing.TypedDict on Py < 3.12."""

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated: unsupported on this Python version")

    monkeypatch.setattr("prompture.agents.tools_schema.TypeAdapter", _boom)


class TestStructuredTypeFallback:
    def test_typed_dict_still_expands(self, type_adapter_unavailable):
        schema = _python_type_to_json_schema(Movie)
        assert schema["type"] == "object"
        assert schema["properties"]["year"] == {"type": "integer"}
        assert schema["properties"]["title"] == {"type": "string"}
        assert sorted(schema["required"]) == ["title", "year"]

    def test_never_degrades_to_a_bare_string(self, type_adapter_unavailable):
        """The regression this guards: a whole object described as a string."""
        assert _python_type_to_json_schema(Movie) != {"type": "string"}

    def test_total_false_typed_dict_has_no_required_keys(self, type_adapter_unavailable):
        schema = _python_type_to_json_schema(PartialMovie)
        assert set(schema["properties"]) == {"title", "year"}
        assert "required" not in schema

    def test_dataclass_defaults_are_not_required(self, type_adapter_unavailable):
        schema = _python_type_to_json_schema(BoxWithDefaults)
        assert schema["required"] == ["w"]
        assert schema["properties"]["h"] == {"type": "integer"}

    def test_model_optional_fields_are_not_required(self, type_adapter_unavailable):
        schema = _python_type_to_json_schema(PointWithOptional)
        assert schema["required"] == ["x"]
        assert schema["properties"]["y"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}

    def test_nested_annotations_are_resolved_recursively(self, type_adapter_unavailable):
        @dataclass
        class Outer:
            tags: list[str]
            where: Movie

        schema = _python_type_to_json_schema(Outer)
        assert schema["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
        assert schema["properties"]["where"]["properties"]["year"] == {"type": "integer"}

    def test_tool_parameter_keeps_its_object_shape(self, type_adapter_unavailable):
        def rate(movie: Movie) -> str:
            """Rate a movie.

            Args:
                movie: The movie to rate.
            """
            return movie["title"]

        prop = tool_from_function(rate).parameters["properties"]["movie"]
        assert prop["type"] == "object"
        assert set(prop["properties"]) == {"title", "year"}
        # The docstring description survives the fallback.
        assert prop["description"] == "The movie to rate."
