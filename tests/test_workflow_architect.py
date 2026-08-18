"""Tests for the workflow Architect (NL -> validated Graph) (G08)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from prompture.workflow import (
    ArchitectResult,
    GraphRunner,
    NodeSpec,
    WorkflowGraphSpec,
    aarchitect,
    architect,
    node_type_catalog,
)

_EXTRACT = "prompture.extraction.core.extract_with_model"
_AEXTRACT = "prompture.extraction.async_core.extract_with_model"


def _spec(**kw) -> WorkflowGraphSpec:
    base = dict(
        name="demo",
        inputs=["topic"],
        nodes=[
            NodeSpec(id="topic", type="input", config={"name": "topic"}),
            NodeSpec(id="s", type="llm", config={"model": "mock/m", "prompt": "Brief on {{topic.outputs.value}}"}),
        ],
        outputs={"brief": "{{s.outputs.text}}"},
        message="Make a brief.",
        suggestions=["add image"],
    )
    base.update(kw)
    return WorkflowGraphSpec(**base)


class TestSpecToGraph:
    def test_builds_valid_graph(self):
        g = _spec().to_graph("wf")
        assert g.id == "wf"
        assert g.topo_order() == ["topic", "s"]
        assert g.outputs == {"brief": "{{s.outputs.text}}"}
        assert [h.name for h in g.inputs] == ["topic"]

    def test_cycle_spec_raises(self):
        spec = WorkflowGraphSpec(
            nodes=[
                NodeSpec(id="a", type="constant", config={"value": "{{b.outputs.value}}"}),
                NodeSpec(id="b", type="constant", config={"value": "{{a.outputs.value}}"}),
            ]
        )
        from prompture.workflow import CycleError

        with pytest.raises(CycleError):
            spec.to_graph()


class TestCatalog:
    def test_lists_builtins(self):
        cat = node_type_catalog()
        for t in ["input", "output", "llm", "media_gen", "tool", "transform", "constant"]:
            assert t in cat


class TestArchitect:
    def test_happy_path(self):
        spec = _spec()
        with patch(_EXTRACT, return_value={"model": spec, "usage": {"cost": 0.01, "total_tokens": 50}}):
            res = architect("brief about a topic", model="mock/m")
        assert isinstance(res, ArchitectResult)
        assert res.ok
        assert [n.id for n in res.graph.nodes] == ["topic", "s"]
        assert res.message == "Make a brief."
        assert res.suggestions == ["add image"]
        assert res.usage["cost"] == 0.01

    def test_unknown_node_type_is_error(self):
        spec = WorkflowGraphSpec(nodes=[NodeSpec(id="x", type="quantum")])
        with patch(_EXTRACT, return_value={"model": spec, "usage": {}}):
            res = architect("weird")
        assert not res.ok
        assert "unknown node types" in res.error
        assert res.graph is None

    def test_cycle_is_error(self):
        spec = WorkflowGraphSpec(
            nodes=[
                NodeSpec(id="a", type="constant", config={"value": "{{b.outputs.value}}"}),
                NodeSpec(id="b", type="constant", config={"value": "{{a.outputs.value}}"}),
            ]
        )
        with patch(_EXTRACT, return_value={"model": spec, "usage": {}}):
            res = architect("cyclic")
        assert not res.ok and "cycle" in res.error.lower()

    def test_passes_model_and_system_prompt(self):
        spec = _spec()
        with patch(_EXTRACT, return_value={"model": spec, "usage": {}}) as mx:
            architect("x", model="openai/gpt-4o-mini", strategy="tool_call", max_retries=3)
        kwargs = mx.call_args.kwargs
        assert kwargs["model_name"] == "openai/gpt-4o-mini"
        assert kwargs["strategy"] == "tool_call"
        assert kwargs["max_retries"] == 3
        assert "Available node types" in kwargs["system_prompt"]

    def test_edit_mode_includes_current_graph(self):
        current = _spec().to_graph("existing")
        spec = _spec()
        with patch(_EXTRACT, return_value={"model": spec, "usage": {}}) as mx:
            architect("add an image", current_graph=current)
        assert "existing" in mx.call_args.kwargs["system_prompt"]

    def test_async_architect(self):
        spec = _spec()
        with patch(_AEXTRACT, new=AsyncMock(return_value={"model": spec, "usage": {}})):
            res = asyncio.run(aarchitect("brief", model="mock/m"))
        assert res.ok
        assert [n.id for n in res.graph.nodes] == ["topic", "s"]


class _FakeLLM:
    def generate(self, prompt, options):
        return {"text": f"brief:{prompt}", "meta": {"cost": 0.0}}


class TestArchitectToRunnable:
    def test_architected_graph_runs(self):
        # Architect produces a graph; it then executes end-to-end (LLM mocked).
        spec = _spec()
        with patch(_EXTRACT, return_value={"model": spec, "usage": {}}):
            res = architect("brief", model="mock/m")
        with patch("prompture.drivers.get_driver_for_model", return_value=_FakeLLM()):
            run = GraphRunner().run(res.graph, inputs={"topic": "robots"})
        assert run.ok
        assert run.outputs["brief"] == "brief:Brief on robots"


@pytest.mark.integration
def test_architect_live(default_model):
    res = architect("Summarize a topic with an LLM and output the summary", model=default_model)
    assert res.spec is not None
    if res.ok:
        assert res.graph.topo_order()
