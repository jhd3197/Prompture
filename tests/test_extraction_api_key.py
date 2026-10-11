"""extract_with_model / extract_and_jsonify pass a per-call api_key to the driver they resolve."""

import pytest
from pydantic import BaseModel

from prompture.extraction import core


class Person(BaseModel):
    name: str


class Resolved(Exception):
    """Stops the extraction right after the driver lookup, carrying its arguments."""


def _capture(monkeypatch):
    def fake_get_driver_for_model(model_str, *, env=None, api_key=None, **overrides):
        raise Resolved(model_str, api_key)

    monkeypatch.setattr(core, "get_driver_for_model", fake_get_driver_for_model)


def test_extract_with_model_passes_the_key(monkeypatch):
    _capture(monkeypatch)
    with pytest.raises(Resolved) as found:
        core.extract_with_model(Person, "Her name is Ana.", "openai/gpt-4o-mini", api_key="sk-user", cache=False)
    assert found.value.args == ("openai/gpt-4o-mini", "sk-user")


def test_without_a_key_the_global_settings_apply(monkeypatch):
    _capture(monkeypatch)
    with pytest.raises(Resolved) as found:
        core.extract_and_jsonify("Her name is Ana.", Person.model_json_schema(), model_name="openai/gpt-4o-mini")
    assert found.value.args == ("openai/gpt-4o-mini", None)
