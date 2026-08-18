"""Tests for the workflow data model (prompture/workflow/model.py)."""

from __future__ import annotations

import pytest

from prompture.workflow import (
    CycleError,
    Edge,
    Graph,
    GraphValidationError,
    Handle,
    HandleType,
    Node,
    build_graph,
)


class TestSerialization:
    def test_graph_json_round_trip(self):
        g = Graph(
            id="g",
            nodes=[Node(id="a", type="input", outputs=[Handle("value", HandleType.TEXT)])],
            inputs=[Handle("topic", HandleType.TEXT)],
            outputs={"r": "{{a.outputs.value}}"},
        )
        g2 = Graph.from_json(g.to_json())
        assert g2.to_dict() == g.to_dict()
        assert g2.nodes[0].outputs[0].type is HandleType.TEXT

    def test_from_dict_filters_unknown_keys(self):
        n = Node.from_dict({"id": "x", "type": "constant", "bogus": 1, "config": {"value": 1}})
        assert n.id == "x" and not hasattr(n, "bogus")

    def test_handle_round_trip(self):
        h = Handle("img", HandleType.IMAGE, required=False, default=None, description="d")
        assert Handle.from_dict(h.to_dict()) == h


class TestAdjacency:
    def test_predecessors_union_edges_and_refs(self):
        g = Graph(id="g")
        g.add_node(Node(id="a", type="constant", config={"value": 1}))
        g.add_node(Node(id="b", type="constant", config={"value": 2}))
        g.add_node(Node(id="c", type="transform", config={"op": "value", "value": "{{a.outputs.value}}"}))
        g.add_edge(Edge("b", "value", "c", "x"))  # explicit edge
        assert g.predecessors("c") == ["a", "b"]
        assert "c" in g.successors("a")
        assert "c" in g.successors("b")

    def test_input_and_output_nodes(self):
        g = build_graph(
            [
                Node(id="i", type="input"),
                Node(id="m", type="transform", config={"value": "{{i.outputs.value}}"}),
            ]
        )
        assert [n.id for n in g.input_nodes()] == ["i"]
        assert [n.id for n in g.output_nodes()] == ["m"]


class TestTopoOrder:
    def test_deterministic_insertion_order(self):
        g = Graph(id="g")
        for nid in ["a", "b", "c"]:
            g.add_node(Node(id=nid, type="constant", config={"value": 1}))
        g.add_node(
            Node(
                id="d", type="transform", config={"value": "{{a.outputs.value}}{{b.outputs.value}}{{c.outputs.value}}"}
            )
        )
        order = g.topo_order()
        assert order.index("a") < order.index("d")
        assert order[:3] == ["a", "b", "c"]

    def test_diamond(self):
        g = Graph(id="g")
        g.add_node(Node(id="root", type="constant", config={"value": 1}))
        g.add_node(Node(id="l", type="transform", config={"value": "{{root.outputs.value}}"}))
        g.add_node(Node(id="r", type="transform", config={"value": "{{root.outputs.value}}"}))
        g.add_node(Node(id="join", type="transform", config={"value": "{{l.outputs.value}}{{r.outputs.value}}"}))
        order = g.topo_order()
        assert order[0] == "root" and order[-1] == "join"
        assert len(order) == 4


class TestValidation:
    def test_duplicate_ids(self):
        g = Graph(id="g", nodes=[Node(id="a", type="constant"), Node(id="a", type="constant")])
        with pytest.raises(GraphValidationError):
            g.validate()

    def test_edge_endpoint_missing(self):
        g = Graph(id="g", nodes=[Node(id="a", type="constant")], edges=[Edge("a", "value", "ghost", "x")])
        with pytest.raises(GraphValidationError):
            g.validate()

    def test_template_root_unknown(self):
        g = Graph(id="g", nodes=[Node(id="a", type="transform", config={"value": "{{ghost.outputs.value}}"})])
        with pytest.raises(GraphValidationError):
            g.validate()

    def test_self_reference_is_cycle(self):
        g = Graph(id="g", nodes=[Node(id="a", type="transform", config={"value": "{{a.outputs.value}}"})])
        with pytest.raises(CycleError):
            g.validate()

    def test_cycle_via_edges(self):
        g = Graph(
            id="g",
            nodes=[Node(id="a", type="constant"), Node(id="b", type="constant")],
            edges=[Edge("a", "v", "b", "x"), Edge("b", "v", "a", "x")],
        )
        with pytest.raises(CycleError):
            g.validate()

    def test_cycle_via_templates(self):
        g = Graph(
            id="g",
            nodes=[
                Node(id="a", type="constant", config={"value": "{{b.outputs.value}}"}),
                Node(id="b", type="constant", config={"value": "{{a.outputs.value}}"}),
            ],
        )
        with pytest.raises(CycleError) as ei:
            g.validate()
        assert set(ei.value.nodes) == {"a", "b"}


class TestHandleCompatibility:
    def test_any_matches_all(self):
        assert HandleType.ANY.is_compatible(HandleType.IMAGE)
        assert HandleType.VIDEO.is_compatible(HandleType.ANY)

    def test_asset_matches_media(self):
        assert HandleType.ASSET.is_compatible(HandleType.IMAGE)
        assert HandleType.AUDIO.is_compatible(HandleType.ASSET)

    def test_exact_otherwise(self):
        assert HandleType.TEXT.is_compatible(HandleType.TEXT)
        assert not HandleType.TEXT.is_compatible(HandleType.NUMBER)
