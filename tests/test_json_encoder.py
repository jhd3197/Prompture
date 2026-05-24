"""Tests for the shared PromptureJSONEncoder.

Added in Phase 0 — replaces the ``json.dumps(..., default=str)`` pattern that
was duplicated across 15+ call sites. Verifies the encoder handles every
common Python type Prompture emits while preserving the legacy fallback
behaviour for unknown objects.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel

from prompture._internal.json_encoder import PromptureJSONEncoder, dumps


class _Color(Enum):
    RED = "red"
    BLUE = "blue"


class _Person(BaseModel):
    name: str
    age: int


class TestPromptureJSONEncoder:
    def _encode(self, payload):
        return json.loads(dumps(payload))

    def test_datetime_serialises_to_iso8601(self) -> None:
        out = self._encode({"when": datetime(2025, 1, 2, 3, 4, 5)})
        assert out["when"] == "2025-01-02T03:04:05"

    def test_date_serialises_to_iso8601(self) -> None:
        out = self._encode({"day": date(2025, 1, 2)})
        assert out["day"] == "2025-01-02"

    def test_time_serialises_to_iso8601(self) -> None:
        out = self._encode({"t": time(3, 4, 5)})
        assert out["t"] == "03:04:05"

    def test_timedelta_serialises_to_total_seconds(self) -> None:
        out = self._encode({"span": timedelta(seconds=90)})
        assert out["span"] == 90.0

    def test_decimal_preserves_precision(self) -> None:
        out = self._encode({"x": Decimal("3.14159265358979323846")})
        assert out["x"] == "3.14159265358979323846"

    def test_uuid_serialises_to_string(self) -> None:
        out = self._encode({"id": UUID("00000000-0000-4000-8000-000000000000")})
        assert out["id"] == "00000000-0000-4000-8000-000000000000"

    def test_enum_serialises_to_value(self) -> None:
        out = self._encode({"c": _Color.RED})
        assert out["c"] == "red"

    def test_path_serialises_to_string(self) -> None:
        out = self._encode({"p": Path("foo") / "bar"})
        assert "bar" in out["p"]

    def test_pydantic_model_serialises_via_model_dump(self) -> None:
        out = self._encode({"who": _Person(name="A", age=1)})
        assert out["who"] == {"name": "A", "age": 1}

    def test_set_is_serialised_deterministically(self) -> None:
        # str-comparable items -> sorted
        out = self._encode({"s": {"b", "a"}})
        assert out["s"] == ["a", "b"]

    def test_set_of_enums_falls_back_to_str_key_sort(self) -> None:
        # Enum members aren't directly comparable; encoder should still emit
        # them deterministically by stringifying the comparison key.
        out_a = self._encode({"s": {_Color.RED, _Color.BLUE}})
        out_b = self._encode({"s": {_Color.RED, _Color.BLUE}})
        # Each call independently produces the same ordering.
        assert out_a == out_b
        # And the values are the enum .value strings.
        assert set(out_a["s"]) == {"red", "blue"}

    def test_bytes_printable_ascii_round_trips(self) -> None:
        out = self._encode({"b": b"hello"})
        assert out["b"] == "hello"

    def test_bytes_binary_is_base64(self) -> None:
        import base64

        out = self._encode({"b": b"\x00\x01\x02"})
        assert out["b"] == base64.b64encode(b"\x00\x01\x02").decode()

    def test_unknown_object_falls_back_to_str(self) -> None:
        text = dumps({"o": object()})
        assert "object" in text

    def test_explicit_encoder_class_works(self) -> None:
        text = json.dumps({"d": date(2025, 1, 1)}, cls=PromptureJSONEncoder)
        assert json.loads(text)["d"] == "2025-01-01"

    def test_kwargs_are_forwarded(self) -> None:
        text = dumps({"a": 1, "b": 2}, sort_keys=True, indent=2)
        assert text.startswith("{\n  ")
        assert "\"a\": 1" in text and "\"b\": 2" in text

    def test_ensure_ascii_defaults_off(self) -> None:
        # The legacy call sites used ``ensure_ascii=False``; dumps() preserves it.
        text = dumps({"name": "Ańa"})
        assert "Ańa" in text

    @pytest.mark.parametrize(
        "payload",
        [
            {"nested": {"when": datetime(2025, 1, 1), "id": UUID(int=0)}},
            [datetime(2025, 1, 1), Decimal("1.5"), _Color.BLUE],
        ],
    )
    def test_recursive_structures(self, payload) -> None:
        # Should not raise; recursion goes through default() for each leaf.
        text = dumps(payload)
        json.loads(text)
