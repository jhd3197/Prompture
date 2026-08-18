"""Tests for the late-bound reference resolver (prompture/workflow/references.py)."""

from __future__ import annotations

import pytest

from prompture.workflow.errors import ReferenceResolutionError
from prompture.workflow.references import (
    extract_dependencies,
    extract_refs,
    parse_reference,
    resolve_template,
)
from prompture.workflow.state import NodeResult, NodeResultStore, NodeStatus


def scope(inputs=None, **nodes):
    store = NodeResultStore()
    for nid, outs in nodes.items():
        store.set(nid, NodeResult(nid, NodeStatus.COMPLETED, outputs=outs))
    inp = inputs or {}

    class _S:
        def __contains__(self, k):
            return k == "inputs" or k in store

        def __getitem__(self, k):
            return inp if k == "inputs" else store.view(k)

    return _S()


class TestWholeStringVsInterpolation:
    def test_whole_string_preserves_dict(self):
        sc = scope(n={"data": {"x": [1, 2]}})
        out = resolve_template("{{n.outputs.data}}", sc)
        assert out == {"x": [1, 2]} and isinstance(out, dict)

    def test_whole_string_preserves_list_and_int(self):
        sc = scope(n={"lst": [1, 2], "num": 7})
        assert resolve_template("{{n.outputs.lst}}", sc) == [1, 2]
        assert resolve_template("{{n.outputs.num}}", sc) == 7
        assert isinstance(resolve_template("{{n.outputs.num}}", sc), int)

    def test_embedded_interpolates_to_str(self):
        sc = scope(n={"text": "hi"})
        assert resolve_template("say {{n.outputs.text}}!", sc) == "say hi!"

    def test_mixed_multiple_refs(self):
        sc = scope(a={"v": "X"}, b={"v": "Y"})
        assert resolve_template("{{a.outputs.v}}-{{b.outputs.v}}", sc) == "X-Y"

    def test_recurses_dict_and_list(self):
        sc = scope(n={"text": "hi"})
        out = resolve_template({"k": ["{{n.outputs.text}}", 5]}, sc)
        assert out == {"k": ["hi", 5]}


class TestAddressingForms:
    def test_first_output_sugar(self):
        sc = scope(n={"text": "hi"})
        assert resolve_template("{{n.outputs[0].value}}", sc) == "hi"

    def test_named_dotted_and_quoted(self):
        sc = scope(n={"caption": "c"})
        assert resolve_template("{{n.outputs.caption}}", sc) == "c"
        assert resolve_template("{{n.outputs['caption']}}", sc) == "c"

    def test_sole_value_sugar(self):
        sc = scope(n={"only": 42})
        assert resolve_template("{{n.value}}", sc) == 42

    def test_value_sugar_errors_when_multiple_outputs(self):
        sc = scope(n={"a": 1, "b": 2})
        with pytest.raises(ReferenceResolutionError):
            resolve_template("{{n.value}}", sc)

    def test_deep_index(self):
        sc = scope(n={"data": {"items": [{"name": "z"}]}})
        assert resolve_template("{{n.outputs.data['items'][0].name}}", sc) == "z"

    def test_status_and_usage(self):
        store = NodeResultStore()
        store.set("n", NodeResult("n", NodeStatus.COMPLETED, outputs={"t": 1}, usage={"cost": 0.5}))

        class _S:
            def __contains__(self, k):
                return k in store

            def __getitem__(self, k):
                return store.view(k)

        assert resolve_template("{{n.status}}", _S()) == "completed"
        assert resolve_template("{{n.usage.cost}}", _S()) == 0.5

    def test_inputs_root(self):
        assert resolve_template("{{inputs.topic}}", scope(inputs={"topic": "robots"})) == "robots"


class TestMalformedAndSafety:
    @pytest.mark.parametrize("tmpl", ["{{ }}", "{{n.}}", "{{n[}}", "{{1abc}}"])
    def test_malformed_raises(self, tmpl):
        with pytest.raises(ReferenceResolutionError):
            resolve_template(tmpl, scope(n={"x": 1}))

    def test_no_eval_on_adversarial(self):
        # Treated as plain attr/index access — never evaluated.
        with pytest.raises(ReferenceResolutionError):
            resolve_template("{{n.__class__}}", scope(n={"x": 1}))

    def test_unknown_root_raises(self):
        with pytest.raises(ReferenceResolutionError):
            resolve_template("{{missing.outputs.x}}", scope(n={"x": 1}))

    def test_non_strict_passes_through(self):
        assert resolve_template("{{missing.x}}", scope(), strict=False) == "{{missing.x}}"


class TestExtractDependencies:
    def test_recurses_and_excludes_reserved(self):
        cfg = {
            "prompt": "a {{n.outputs.text}} and {{inputs.topic}}",
            "nested": ["{{m.value}}", {"deep": "{{k.outputs[0].value}}"}],
        }
        assert extract_dependencies(cfg) == {"n", "m", "k"}

    def test_empty_when_no_refs(self):
        assert extract_dependencies({"x": "plain", "y": 5}) == set()

    def test_extract_refs_count(self):
        refs = extract_refs("{{a.x}} {{b.y}}")
        assert [r.root for r in refs] == ["a", "b"]

    def test_parse_reference_steps(self):
        ref = parse_reference("n.outputs[0].value")
        assert ref.root == "n"
        assert ref.steps == (("attr", "outputs"), ("index", 0), ("attr", "value"))
