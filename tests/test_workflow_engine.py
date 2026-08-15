"""Tests for the workflow scheduler, node runners, and engine facade."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from prompture.groups.types import ErrorPolicy
from prompture.workflow import (
    Graph,
    GraphRunner,
    Handle,
    HandleType,
    Node,
    NodeResult,
    NodeStatus,
    RunStatus,
    WorkflowCallbacks,
    WorkflowError,
    compile_graph,
    register_node_type,
    run_node,
    unregister_node_type,
)


@pytest.fixture(autouse=True)
def _fake_node_types():
    register_node_type(
        "boom",
        lambda node, cfg, ctx: (_ for _ in ()).throw(RuntimeError("kaboom")),
        overwrite=True,
    )
    register_node_type(
        "echo",
        lambda node, cfg, ctx: NodeResult(node.id, NodeStatus.COMPLETED, outputs={"value": cfg.get("value")}),
        overwrite=True,
    )
    yield
    unregister_node_type("boom")
    unregister_node_type("echo")


class TestExecution:
    def test_linear_flow(self):
        g = Graph(id="g", inputs=[Handle("x")])
        g.add_node(Node(id="x", type="input"))
        g.add_node(Node(id="a", type="echo", config={"value": "{{x.outputs.value}}"}))
        g.add_node(
            Node(id="b", type="transform", config={"op": "format", "template": "[{}]", "args": ["{{a.outputs.value}}"]})
        )
        g.outputs = {"r": "{{b.outputs.value}}"}
        run = GraphRunner().run(g, inputs={"x": "hi"})
        assert run.ok
        assert run.outputs["r"] == "[hi]"
        assert run.order == ["x", "a", "b"]

    def test_diamond_each_node_once(self):
        calls: list[str] = []
        register_node_type(
            "track",
            lambda node, cfg, ctx: (
                calls.append(node.id),
                NodeResult(node.id, NodeStatus.COMPLETED, outputs={"value": node.id}),
            )[1],
            overwrite=True,
        )
        try:
            g = Graph(id="g")
            g.add_node(Node(id="root", type="track"))
            g.add_node(Node(id="l", type="track", config={"_": "{{root.outputs.value}}"}))
            g.add_node(Node(id="r", type="track", config={"_": "{{root.outputs.value}}"}))
            g.add_node(Node(id="j", type="track", config={"_": "{{l.outputs.value}}{{r.outputs.value}}"}))
            run = GraphRunner().run(g)
            assert run.ok
            assert sorted(calls) == ["j", "l", "r", "root"]
        finally:
            unregister_node_type("track")


class TestErrorHandling:
    def _diamond_with_failure(self):
        g = Graph(id="g")
        g.add_node(Node(id="a", type="echo", config={"value": 1}))
        g.add_node(Node(id="bad", type="boom", config={"_": "{{a.outputs.value}}"}))
        g.add_node(Node(id="dep", type="echo", config={"value": "{{bad.outputs.value}}"}))
        g.add_node(Node(id="indep", type="echo", config={"value": 99}))
        return g

    def test_fail_fast_skips_everything_after_failure(self):
        run = GraphRunner().run(self._diamond_with_failure(), error_policy=ErrorPolicy.fail_fast)
        assert run.status is RunStatus.FAILED
        assert run["a"].status is NodeStatus.COMPLETED
        assert run["bad"].status is NodeStatus.FAILED
        assert run["dep"].status is NodeStatus.SKIPPED
        assert run["indep"].status is NodeStatus.SKIPPED  # fail_fast halts scheduling
        assert run.errors and run.errors[0][0] == "bad"

    def test_continue_on_error_runs_independent_branch(self):
        run = GraphRunner().run(self._diamond_with_failure(), error_policy=ErrorPolicy.continue_on_error)
        assert run.status is RunStatus.FAILED  # a node failed
        assert run["dep"].status is NodeStatus.SKIPPED  # transitive dependent of failure
        assert run["indep"].status is NodeStatus.COMPLETED  # independent branch survives

    def test_raise_on_error_reraises(self):
        from prompture.workflow import NodeExecutionError

        g = Graph(id="g", nodes=[Node(id="bad", type="boom")])
        with pytest.raises(NodeExecutionError):
            GraphRunner().run(g, error_policy=ErrorPolicy.raise_on_error)

    def test_compiled_raises_on_failure(self):
        wf = compile_graph(Graph(id="g", nodes=[Node(id="bad", type="boom")]))
        with pytest.raises(WorkflowError):
            wf()


class TestRunNode:
    def test_run_node_with_upstream(self):
        g = Graph(id="g")
        g.add_node(Node(id="up", type="echo"))
        g.add_node(Node(id="me", type="echo", config={"value": "{{up.outputs.value}}!"}))
        nr = run_node(g, "me", upstream={"up": {"value": "X"}})
        assert nr.status is NodeStatus.COMPLETED
        assert nr.outputs["value"] == "X!"

    def test_run_node_missing_upstream_fails(self):
        from prompture.workflow import NodeExecutionError

        g = Graph(id="g", nodes=[Node(id="me", type="echo", config={"value": "{{ghost.outputs.value}}"})])
        with pytest.raises(NodeExecutionError):
            run_node(g, "me")


class TestCallbacksAndUsage:
    def test_callbacks_fire(self):
        events: list[str] = []
        cbs = WorkflowCallbacks(
            on_node_start=lambda nid: events.append(f"start:{nid}"),
            on_node_complete=lambda nid, r: events.append(f"done:{nid}"),
            on_graph_complete=lambda r: events.append("graph"),
        )
        g = Graph(id="g", nodes=[Node(id="a", type="echo", config={"value": 1})])
        GraphRunner().run(g, callbacks=cbs)
        assert events == ["start:a", "done:a", "graph"]

    def test_usage_rollup(self):
        register_node_type(
            "costly",
            lambda node, cfg, ctx: NodeResult(
                node.id, NodeStatus.COMPLETED, outputs={"v": 1}, usage={"cost": 0.25, "total_tokens": 10}
            ),
            overwrite=True,
        )
        try:
            g = Graph(id="g")
            g.add_node(Node(id="a", type="costly"))
            g.add_node(Node(id="b", type="costly", config={"_": "{{a.outputs.v}}"}))
            run = GraphRunner().run(g)
            assert run.total_usage["cost"] == 0.5
            assert run.total_usage["total_tokens"] == 20
            assert run.total_usage["call_count"] == 2
        finally:
            unregister_node_type("costly")


class _FakeLLMDriver:
    def generate(self, prompt, options):
        return {"text": f"echo:{prompt}", "meta": {"total_tokens": 5, "cost": 0.01, "strategy": "auto"}}


class TestLLMNode:
    def test_llm_node_delegates_to_driver(self):
        g = Graph(id="g", inputs=[Handle("topic")])
        g.add_node(Node(id="topic", type="input"))
        g.add_node(Node(id="s", type="llm", config={"model": "mock/m", "prompt": "brief on {{topic.outputs.value}}"}))
        g.outputs = {"text": "{{s.outputs.text}}"}
        with patch("prompture.drivers.get_driver_for_model", return_value=_FakeLLMDriver()):
            run = GraphRunner().run(g, inputs={"topic": "robots"})
        assert run.ok
        assert run.outputs["text"] == "echo:brief on robots"
        assert run["s"].cost == 0.01
        assert run["s"].strategy == "auto"
        assert run.total_usage["cost"] == 0.01


class _Img:
    def __init__(self, url):
        self.url = url


class _FakeImageDriver:
    def generate_image(self, prompt, options):
        return {"images": [_Img("https://x/a.png")], "meta": {"model_name": "muapi/nano", "cost": 0.039}}


class TestMediaNode:
    def test_media_gen_blocks_and_returns_url(self):
        g = Graph(id="g")
        g.add_node(
            Node(id="img", type="media_gen", config={"modality": "image", "model": "muapi/nano", "prompt": "a cat"})
        )
        g.outputs = {"url": "{{img.outputs.url}}"}
        with patch("prompture.drivers.img_gen_registry.get_img_gen_driver_for_model", return_value=_FakeImageDriver()):
            run = GraphRunner().run(g)
        assert run.ok
        assert run.outputs["url"] == "https://x/a.png"
        assert run["img"].cost == 0.039


class TestToolNode:
    def test_tool_node_runs_registry_tool(self):
        from prompture.agents import ToolRegistry

        reg = ToolRegistry()

        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: first
                b: second
            """
            return a + b

        reg.register(add)
        g = Graph(id="g")
        g.add_node(Node(id="t", type="tool", config={"tool": "add", "args": {"a": 2, "b": 3}}))
        g.outputs = {"sum": "{{t.outputs.result}}"}
        run = GraphRunner().run(g, options={"tool_registry": reg})
        assert run.ok
        assert run.outputs["sum"] == 5


class TestCompiledWorkflow:
    def test_call_returns_outputs(self):
        g = Graph(id="g", inputs=[Handle("topic")])
        g.add_node(Node(id="topic", type="input"))
        g.add_node(
            Node(
                id="up",
                type="transform",
                config={"op": "format", "template": "{}!", "args": ["{{topic.outputs.value}}"]},
            )
        )
        g.outputs = {"r": "{{up.outputs.value}}"}
        wf = compile_graph(g)
        assert wf(topic="hi") == {"r": "hi!"}
        assert wf.signature == {"inputs": ["topic"], "outputs": ["r"]}
        assert wf.last_run is not None and wf.last_run.ok

    def test_as_tool(self):
        g = Graph(id="brief", inputs=[Handle("topic", HandleType.TEXT)])
        g.add_node(Node(id="topic", type="input"))
        g.add_node(Node(id="c", type="transform", config={"value": "{{topic.outputs.value}}"}))
        g.outputs = {"r": "{{c.outputs.value}}"}
        td = compile_graph(g).as_tool()
        assert td.name == "brief"
        assert "topic" in td.parameters["properties"]
        assert td.to_openai_format()["function"]["name"] == "brief"
