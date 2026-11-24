"""Contract tests for reusable local recipes and the matched warehouse benchmark."""

import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
from jsonschema import ValidationError

from prompture.agents.tools_schema import ToolRegistry
from prompture.workflow import Graph, Handle, HandleType, Node, RecipeStore, WorkflowError
from scripts.benchmark_pack_checks import compare, parse_answer
from scripts.pack_checks import fixture, pack_graph, tools_for


@pytest.fixture
def warehouse(tmp_path):
    manifests, scans, expected = fixture()
    reads = Counter()
    registry = tools_for(manifests, scans, reads)
    store = RecipeStore(tmp_path / "catalog.sqlite", scope="A", registry=registry, contract="v1")
    return store, registry, scans, expected, reads


def test_reopen_alias_fresh_inputs_and_project_isolation(warehouse):
    store, registry, scans, expected, reads = warehouse
    store.publish(pack_graph(), aliases=["verify this carton"])
    reopened = RecipeStore(store.path, scope="A", registry=registry, contract="v1")
    tool = reopened.as_tool("verify this carton", verify=lambda out: out["decisions"] == expected)
    assert tool.function(batch="B1")["decisions"]["S11"] == ["pass", "match"]
    scans["S11"]["carton"] = "other"
    expected["S11"] = ["hold", "carton"]
    assert tool.function(batch="B1")["decisions"]["S11"] == ["hold", "carton"]
    assert reads == {"manifests": 2, "scans": 2}
    assert reopened.popular() == [("audit_pack_batch", 2)]
    with pytest.raises(KeyError):
        RecipeStore(store.path, scope="B", registry=registry, contract="v1").load("verify this carton")


def test_learning_runs_builder_on_miss_and_only_promotes_verified_results(warehouse):
    store, _, _, expected, _ = warehouse
    builds = []

    def build():
        builds.append(True)
        return pack_graph()

    def run(verify):
        return store.run_or_learn("audit_pack_batch", {"batch": "B1"}, build=build, verify=verify, event_id="E1")

    with pytest.raises(WorkflowError, match="Verification"):
        run(lambda _: False)
    assert store.popular() == []
    for _ in range(2):
        assert run(lambda out: out["decisions"] == expected)["decisions"] == expected
    assert len(builds) == 2
    assert store.popular() == [("audit_pack_batch", 1)]


def test_version_changes_and_alias_collision(warehouse):
    store, registry, _, _, _ = warehouse
    store.publish(pack_graph(), aliases=["pick"])
    other = pack_graph()
    other.id = "other"
    store.publish(other, aliases=["old"])
    with pytest.raises(sqlite3.IntegrityError):
        store.publish(other, aliases=["pick"])
    assert store.load("old").id == "other"
    with pytest.raises(WorkflowError, match="stale"):
        RecipeStore(store.path, scope="A", registry=registry, contract="v2").load("pick")
    registry.get("audit").parameters["description"] = "changed schema"
    assert store.popular() == []
    with pytest.raises(WorkflowError, match="stale"):
        store.load("pick")


def test_schema_and_tool_failures_are_never_counted(warehouse):
    store, registry, _, _, _ = warehouse
    graph = pack_graph()
    graph.inputs[0] = Handle("batch", HandleType.TEXT, schema={"enum": ["B1"]})
    graph.nodes[-1].config["args"]["invalid_argument"] = True
    store.publish(graph)
    with pytest.raises(ValidationError):
        store.run(graph.id, {"batch": "wrong"}, verify=lambda _: True)
    with pytest.raises(WorkflowError, match="argument"):
        store.run(graph.id, {"batch": "B1"}, verify=lambda _: True)
    assert store.popular() == [(graph.id, 0)]
    registry.get("audit").metadata["is_write"] = True
    with pytest.raises(WorkflowError, match="read-only"):
        store.load(graph.id)


def test_concurrent_first_publication_and_event_deduplication(tmp_path):
    store = RecipeStore(tmp_path / "race.db", scope="demo", registry=ToolRegistry(), contract="1")
    graph = Graph(
        id="echo",
        inputs=[Handle("qty", HandleType.NUMBER)],
        nodes=[Node("echo_input", "transform", {"value": "{{inputs.qty}}"})],
        outputs={"qty": "{{echo_input.value}}"},
    )

    def invoke(i):
        return store.run_or_learn(
            "echo", {"qty": i}, build=lambda: graph, verify=lambda out: out == {"qty": i}, event_id=str(i % 3)
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(invoke, range(8))) == [{"qty": i} for i in range(8)]
    assert store.popular() == [("echo", 3)]
    with pytest.raises(ValidationError):
        store.run("echo", {"qty": "bad"}, verify=lambda _: True)


def test_benchmark_scores_every_id_and_keeps_reads_fresh(tmp_path):
    report = compare(tmp_path / "pack.db", live=False, model="offline", arms=[])
    for result in report["arms"].values():
        assert result["qualified"] and result["correct"] == result["total"] == 12
        assert result["reads"] == {"manifests": 1, "scans": 1}
    _, _, expected = fixture()
    assert parse_answer("The checks are complete.\n" + json.dumps(expected)) == expected
    with pytest.raises(ValueError, match="one JSON"):
        parse_answer("{} then {}")


def test_recipe_input_defaults_and_async_only_tools(tmp_path):
    registry = ToolRegistry()

    @registry.tool
    async def read_count() -> int:
        return 4

    graph = Graph(
        id="scaled",
        inputs=[Handle("scale", HandleType.NUMBER, required=False, default=2)],
        outputs={"value": "{{inputs.scale}}"},
    )
    store = RecipeStore(tmp_path / "recipes.db", scope="demo", registry=registry, contract="1")
    store.publish(graph)
    tool = store.as_tool("scaled", verify=lambda out: out == {"value": 2})
    assert tool.parameters["properties"]["scale"]["default"] == 2
    assert tool.function() == {"value": 2}
    graph.nodes.append(Node("read", "tool", {"tool": "read_count"}))
    store.publish(graph)
    with pytest.raises(WorkflowError, match="Re-register"):
        tool.function()
    with pytest.raises(WorkflowError, match="asynchronous"):
        store.run("scaled", {}, verify=lambda _: True)
    assert store.popular() == [("scaled", 0)]


def test_simulated_tool_loop_can_use_saved_recipe(warehouse):
    from prompture.agents.conversation import Conversation

    store, _, _, expected, reads = warehouse
    store.publish(pack_graph())
    tools = ToolRegistry()
    tools.add(store.as_tool("audit_pack_batch", verify=lambda out: out["decisions"] == expected))

    class TextDriver:
        supports_tool_use = False

        def generate_messages_with_hooks(self, messages, options):
            if "Tool 'audit_pack_batch' returned" in messages[-1]["content"]:
                assert "conflict" in messages[-1]["content"]
                return {"text": '{"type":"final_answer","content":"checked"}'}
            return {"text": '{"type":"tool_call","name":"audit_pack_batch","arguments":{"batch":"B1"}}'}

    assert Conversation(driver=TextDriver(), tools=tools).ask("Verify batch B1") == "checked"
    assert reads == {"manifests": 1, "scans": 1}


def test_hosted_callbacks_preserve_container_and_raw_blocks(monkeypatch, tmp_path):
    from scripts import benchmark_pack_checks as bench

    manifests, _, expected = fixture()
    calls = []
    tool_block = {
        "type": "tool_use",
        "name": "read_manifests",
        "id": "t1",
        "input": {"batch": "B1"},
        "caller": {"type": bench.EXECUTOR, "tool_id": "code1"},
    }
    assistant = [
        {"type": "server_tool_use", "id": "code1", "name": "code_execution", "input": {"code": "..."}},
        tool_block,
    ]
    payloads = [
        {"content": assistant, "container": {"id": "container1"}, "stop_reason": "tool_use"},
        {
            "content": [{**tool_block, "name": "read_scans", "id": "t2"}],
            "container": {"id": "container1"},
            "stop_reason": "tool_use",
        },
        {"content": [{"type": "text", "text": json.dumps(expected)}], "stop_reason": "end_turn"},
    ]

    def post(url, *, json, **kwargs):
        from copy import deepcopy

        calls.append(deepcopy(json))
        data = {**payloads[len(calls) - 1], "usage": {"input_tokens": 20, "output_tokens": 3}}
        return type("Response", (), {"status_code": 200, "json": lambda self: data})()

    monkeypatch.setattr(bench.requests, "post", post)
    monkeypatch.setattr("prompture.drivers.claude_driver.env_api_key", lambda: "dummy")
    result = bench.compare(tmp_path / "ptc.sqlite", live=True, model="mock", arms=["ptc"])["arms"]["ptc"]
    assert result["qualified"] and result["hosted_callbacks"] == 2
    assert (result["input_tokens"], result["output_tokens"]) == (60, 9)
    assert calls[1]["container"] == "container1" and calls[1]["tools"] == calls[0]["tools"]
    assert calls[1]["messages"][-2]["content"] == assistant
    (block,) = calls[1]["messages"][-1]["content"]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "t1"
    assert json.loads(block["content"]) == manifests
