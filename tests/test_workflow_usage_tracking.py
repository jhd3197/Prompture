"""Workflow LLM/media nodes run through the hook wrappers, so their spend
reaches the usage tracker like every other Prompture surface."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from prompture.drivers.base import Driver
from prompture.infra.tracker import UsageEvent, configure_tracker
from prompture.workflow import Graph, GraphRunner
from prompture.workflow.model import Node


class _HookedDriver(Driver):
    def __init__(self) -> None:
        super().__init__()
        self.model = "mock/hooked"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "ok", "meta": {"total_tokens": 5, "cost": 0.01}}


class _BareDriver:
    """Exotic driver exposing only generate() — the pre-fix test contract."""

    def generate(self, prompt, options):
        return {"text": "bare", "meta": {"total_tokens": 3, "cost": 0.002}}


def _capture_events(tmp_path) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    configure_tracker(db_path=str(tmp_path / "u.db"), persist=False, sinks=[events.append])
    return events


def _llm_graph() -> Graph:
    g = Graph(id="g")
    g.add_node(Node(id="s", type="llm", config={"model": "mock/hooked", "prompt": "hi"}))
    g.outputs = {"text": "{{s.outputs.text}}"}
    return g


def test_llm_node_records_usage_event(tmp_path):
    events = _capture_events(tmp_path)

    with patch("prompture.drivers.get_driver_for_model", return_value=_HookedDriver()):
        run = GraphRunner().run(_llm_graph())

    assert run.ok
    assert run.outputs["text"] == "ok"
    assert len(events) == 1
    assert events[0].cost == 0.01


def test_llm_node_falls_back_to_bare_generate(tmp_path):
    events = _capture_events(tmp_path)

    with patch("prompture.drivers.get_driver_for_model", return_value=_BareDriver()):
        run = GraphRunner().run(_llm_graph())

    assert run.ok
    assert run.outputs["text"] == "bare"
    # No hook wrapper on the driver → nothing recorded, but the run works.
    assert events == []
