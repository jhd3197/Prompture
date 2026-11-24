"""Discovery is read-only. Execution requires live inputs, the same scope/version, and a verifier."""

import pytest

from prompture.agents.tools_schema import ToolRegistry
from prompture.drivers.base import Driver
from prompture.drivers.decision_base import ChoiceAnswer, DecisionDriver, DecisionResponse
from prompture.workflow import Graph, Handle, Node, RecipeCandidate, RecipeExampleIndex, RecipeStore, WorkflowError


class CompletionOnly(Driver):
    def __init__(self, answer="1"):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt, options):
        self.prompts.append(prompt)
        return {
            "text": self.answer,
            "meta": {
                "prompt_tokens": 75,
                "completion_tokens": 1,
                "total_tokens": 76,
                "raw_response": {"private": "ignored"},
            },
        }


class FixedDecision(DecisionDriver):
    def __init__(self, choice):
        super().__init__()
        self.choice = choice
        self.criteria = None

    def decide(self, state, questions, **options):
        self.criteria = questions["recipe"].criteria
        answers = {"recipe": ChoiceAnswer(self.choice, {}, 0.7)} if self.choice is not None else {}
        return DecisionResponse("fixed", answers, {"input_tokens": 15, "output_tokens": 0, "cost": 0})


def graph(name, description):
    return Graph(
        id=name,
        name=description,
        inputs=[Handle("record")],
        nodes=[Node("read", "transform", {"value": "{{inputs.record}}"})],
        outputs={"result": "{{read.value}}"},
    )


@pytest.fixture
def store(tmp_path):
    store = RecipeStore(tmp_path / "recipes.sqlite", scope="site-A", registry=ToolRegistry(), contract="v1")
    store.publish(graph("check_items", "Compare carton contents"), aliases=["packing check"])
    store.publish(graph("check_dock", "Check outbound carrier lane"), aliases=["departure bay"])
    return store


def test_novel_request_legacy_completion_only_and_fresh_execution(store):
    driver = CompletionOnly()
    match = store.discover("Which departure bay was used?", model=driver, limit=1)
    assert match.status == "selected" and match.recipe.name == "check_dock"
    assert match.candidates == ("check_dock",) and match.usage == {
        "prompt_tokens": 75,
        "completion_tokens": 1,
        "total_tokens": 76,
    }
    assert len(driver.prompts) == 1 and "check_items" not in driver.prompts[0]
    assert store.popular()[0][1] == 0
    for value in ("lane-a", "lane-b"):
        assert (
            store.run_match(match, {"record": value}, verify=lambda out, v=value: out["result"] == v)["result"] == value
        )
    exact = store.discover("departure bay", model=driver)
    assert (exact.status, exact.recipe.verified_uses, exact.usage) == ("exact", 2, {})
    assert len(driver.prompts) == 1
    # A model selection never silently promotes the phrasing to a verified alias.
    assert store.discover("Which departure bay was used?", model=driver).status == "selected"
    assert len(driver.prompts) == 2


@pytest.mark.parametrize(
    "answer,status",
    [
        ("0", "abstained"),
        ("**0**", "abstained"),
        ("10", "invalid"),
        ("1 or 2", "invalid"),
        ("**1** or **2**", "invalid"),
        ("check_dock", "invalid"),
        ("", "invalid"),
        (None, "invalid"),
    ],
)
def test_abstains_or_rejects_malformed_choice(store, answer, status):
    result = store.discover("Check the paperwork", model=CompletionOnly(answer))
    assert result.status == status and result.recipe is None
    with pytest.raises(WorkflowError):
        store.run_match(result, {}, verify=lambda _: True)


def test_scope_versions_empty_catalog_and_ties(store):
    driver = CompletionOnly()
    match = store.discover("packing check", model=driver)
    other = RecipeStore(store.path, scope="site-B", registry=ToolRegistry(), contract="v1")
    assert other.discover("packing check", model=driver).status == "empty"
    other.publish(graph("other", "Other project's description"))
    assert "other" not in store.discover("check", model=driver).candidates
    with pytest.raises(WorkflowError):
        other.run_match(match, {}, verify=lambda _: True)
    store.run_match(match, {"record": "a"}, verify=lambda _: True)
    assert store.search("unknown")[0].name == "check_items"
    assert store.search("outbound carrier")[0].name == "check_dock"
    store.publish(graph("check_items", "Revised verification contract"))
    with pytest.raises(WorkflowError, match="changed"):
        store.run_match(match, {"record": "a"}, verify=lambda _: True)
    stale = RecipeStore(store.path, scope="site-A", registry=ToolRegistry(), contract="v2")
    assert stale.search("outbound") == []
    assert stale.discover("new wording", model=driver).status == "empty"


def test_one_bold_choice(store):
    result = store.discover("check the dock assignment", model=CompletionOnly(" **1** "))
    assert result.status == "selected" and result.recipe.name == result.candidates[0]


@pytest.mark.parametrize("choice,status", [("1", "selected"), ("0", "abstained"), ("99", "invalid"), (None, "invalid")])
def test_typed_selection(store, choice, status, monkeypatch):
    driver = FixedDecision(choice)
    monkeypatch.setattr("prompture.drivers.get_decision_driver_for_model", lambda _: driver)
    index = RecipeExampleIndex({"check_dock": ["which exit"], "ghost": ["which exit"]})
    result = store.discover("which exit", model="laya/typed-decisions", index=index, limit=1)
    assert set(driver.criteria) == {"0", "1"}
    assert result.candidates == ("check_dock",)
    assert result.status == status
    assert result.usage == {"prompt_tokens": 15, "completion_tokens": 0, "cost": 0}


def test_category_grouping_and_fallback_order():
    candidates = [RecipeCandidate(n, n, n, 0) for n in ("lane", "stock", "receipt", "handoff")]
    samples = {"stock": ["remaining inventory"], "receipt": ["supplier delivery"], "lane": ["carrier truck"]}
    categories = {"stock": "inventory", "receipt": "inventory", "lane": "shipping"}
    index = RecipeExampleIndex(samples, categories=categories)
    assert [c.name for c in index.shortlist("remaining inventory", candidates, limit=3)] == ["stock", "receipt", "lane"]
    assert index.shortlist("§§", candidates, limit=3) == candidates[:3]
    assert index.shortlist("remaining inventory", candidates, limit=1) == [candidates[1]]
    with pytest.raises(TypeError):
        RecipeExampleIndex({"stock": "inventory"})


@pytest.mark.parametrize("answer,status", [("Yes.", "selected"), ("no", "abstained"), ("yes or no", "invalid")])
def test_confirm_only_the_leading_suggestion(store, answer, status):
    driver = CompletionOnly(answer)
    match = store.discover("outbound carrier", model=driver, confirm_first=True)
    assert match.candidates == ("check_dock",)
    assert match.status == status
    assert "carton" not in driver.prompts[0].lower()
    assert store.discover("departure bay", model=driver, confirm_first=True).status == "exact"
    assert len(driver.prompts) == 1


@pytest.mark.parametrize("driver", [CompletionOnly("**3**"), FixedDecision("3")])
def test_typed_and_text_declines_are_never_recipe_calls(store, driver):
    reasons = ("Outside the catalog", "Requires a change")
    result = store.discover("please move the carton", model=driver, decline_reasons=reasons)
    assert (result.status, result.recipe, result.decline_reason) == ("abstained", None, reasons[1])
    if isinstance(driver, FixedDecision):
        assert driver.criteria["3"] == reasons[1] and driver.criteria["0"] == reasons[0]
    with pytest.raises(WorkflowError):
        store.run_match(result, {}, verify=lambda _: True)
    selected = store.discover("carrier lane?", model=FixedDecision("2"), decline_reasons=reasons)
    assert selected.recipe.name == selected.candidates[1] and selected.decline_reason is None
    invalid = store.discover("carrier lane?", model=FixedDecision("4"), decline_reasons=reasons)
    assert invalid.status == "invalid" and invalid.decline_reason is None


def test_invalid_decline_menu(store):
    for reasons in ("a string", [" "]):
        with pytest.raises(ValueError, match="decline reasons"):
            store.discover("departure bay", model=CompletionOnly(), decline_reasons=reasons)
    with pytest.raises(ValueError, match="confirmation"):
        store.discover("carrier lane?", model=CompletionOnly(), decline_reasons=["unclear"], confirm_first=True)


def test_invalid_request_never_runs_model(store):
    driver = CompletionOnly()
    with pytest.raises(ValueError):
        store.discover(" ", model=driver)
    with pytest.raises(ValueError):
        store.discover("packing check", model=driver, limit=0)
    assert not driver.prompts
