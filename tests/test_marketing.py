"""Brand context, claims checking and marketing output models.

Every test stubs the LLM, so the suite never touches a provider.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from prompture.agents.persona import Persona
from prompture.marketing import (
    BrandContext,
    ClaimsChecker,
    CoreMessaging,
    EmailSequence,
    IdealCustomer,
    LeadMagnetIdeas,
    OfferStack,
    ShotList,
    SocialPost,
    TopTens,
    VideoClip,
    analyze_metric,
    find_on_image_text,
    merge_brand_system_prompt,
    scorecard_brief,
    word_count,
)

FILLED_MARKDOWN = """# Brand: fill this in

Short answers are fine.

## Basics
- Name: Northwind Roofing
- What we sell: Roof repair and replacement for homeowners in Austin
- Tagline: Dry by Friday
- Website: https://northwindroofing.com
- Call to action: Book a free inspection
- Phone: (512) 555-0142
- Email: hello@northwindroofing.com

## Ideal customer
(One short paragraph.)
Homeowners aged 35 to 60 who just found a leak
after a storm and do not trust roofers.

## Pains
- Water is coming through the ceiling and nobody calls back
- Last roofer took a deposit and vanished

## Objections
- It will cost too much

## Core messaging
- Umbrella statement: Your roof, fixed right, by Friday.
- Cold audience: Most leaks start small and get expensive fast.

## Voice
- Tone: plain-spoken, calm, local
- Words we never use: cheap, guaranteed lowest

## Proof points
- 1,200 roofs replaced since 2011
- A+ rating with the Better Business Bureau

## Testimonials
- "They showed up the same day." Maria G., Round Rock

## Colors
- Primary: #1e2761
- Accent: f90
- Secondary: pull from logo

## Service area
Austin, Round Rock and Pflugerville
"""


@pytest.fixture
def brand() -> BrandContext:
    return BrandContext.from_markdown(FILLED_MARKDOWN)


class _ScriptedDriver:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.prompts: list[str] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        return {"text": self._texts.pop(0) if self._texts else "", "meta": {"cost": 0.001, "total_tokens": 10}}


# ----------------------------------------------------------------------
# BrandContext
# ----------------------------------------------------------------------


class TestBrandContextMarkdown:
    def test_parses_every_section(self, brand: BrandContext) -> None:
        assert brand.name == "Northwind Roofing"
        assert brand.offer.startswith("Roof repair")
        assert brand.contact == {"phone": "(512) 555-0142", "email": "hello@northwindroofing.com"}
        assert brand.ideal_customer == (
            "Homeowners aged 35 to 60 who just found a leak after a storm and do not trust roofers."
        )
        assert len(brand.pains) == 2
        assert brand.umbrella_statement == "Your roof, fixed right, by Friday."
        assert brand.cold_message.startswith("Most leaks")
        assert brand.voice == ["plain-spoken", "calm", "local"]
        assert brand.avoid_words == ["cheap", "guaranteed lowest"]
        assert brand.proof_points[0] == "1,200 roofs replaced since 2011"
        assert brand.testimonials == ['"They showed up the same day." Maria G., Round Rock']

    def test_colors_normalized_and_non_hex_kept_as_note(self, brand: BrandContext) -> None:
        assert brand.colors.primary == "#1E2761"
        assert brand.colors.accent == "#FF9900"
        assert brand.colors.secondary is None
        assert brand.notes["Colors: Secondary"] == "pull from logo"

    def test_unknown_section_lands_in_notes(self, brand: BrandContext) -> None:
        assert brand.notes["Service area"] == "Austin, Round Rock and Pflugerville"

    def test_blank_template_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="no name"):
            BrandContext.from_markdown(BrandContext.markdown_template())

    def test_template_round_trip(self) -> None:
        filled = BrandContext.markdown_template().replace("- Name:", "- Name: Acme", 1)
        assert BrandContext.from_markdown(filled).name == "Acme"

    def test_yaml_and_json_round_trip(self, brand: BrandContext, tmp_path) -> None:
        brand.save_yaml(tmp_path / "brand.yaml")
        brand.save_json(tmp_path / "brand.json")
        assert BrandContext.load(tmp_path / "brand.yaml") == brand
        assert BrandContext.load(tmp_path / "brand.json") == brand

    def test_rejects_bad_hex(self) -> None:
        with pytest.raises(ValidationError):
            BrandContext(name="X", colors={"primary": "blue"})


class TestBrandContextRendering:
    def test_render_includes_facts_and_rules(self, brand: BrandContext) -> None:
        text = brand.render()
        assert text.startswith("## Brand context")
        assert "Business: Northwind Roofing" in text
        assert "### Verified proof" in text
        assert "1,200 roofs replaced since 2011" in text
        assert "Never use these words: cheap, guaranteed lowest." in text
        assert "Never write a testimonial" in text

    def test_empty_sections_are_skipped(self) -> None:
        text = BrandContext(name="Solo").render()
        assert "### Ideal customer" not in text
        assert "### Brand rules" in text

    def test_as_persona_and_apply_to(self, brand: BrandContext) -> None:
        persona = brand.as_persona()
        assert persona.render().startswith("## Brand context")
        base = Persona(name="writer", system_prompt="You write for {{brand_name}}.", variables={"x": "1"})
        combined = brand.apply_to(base)
        rendered = combined.render()
        assert rendered.startswith("You write for Northwind Roofing.")
        assert "## Brand context" in rendered
        assert combined.variables["x"] == "1"

    def test_merge_brand_system_prompt(self, brand: BrandContext) -> None:
        assert merge_brand_system_prompt("Be brief.", None) == "Be brief."
        merged = merge_brand_system_prompt("Be brief.", brand)
        assert merged.startswith("## Brand context") and merged.endswith("Be brief.")

    def test_with_output_folds_messaging_back(self, brand: BrandContext) -> None:
        messaging = CoreMessaging(
            umbrella_statement="Dry homes, honest roofers.",
            cold={"message": "Small leaks become big bills."},
            warm={"message": "We show up when we say we will."},
            hot={"message": "Book today, dry by Friday."},
        )
        updated = brand.with_output(messaging)
        assert updated.umbrella_statement == "Dry homes, honest roofers."
        assert updated.hot_message == "Book today, dry by Friday."
        assert updated.name == brand.name

    def test_agent_appends_brand_after_persona(self, brand: BrandContext) -> None:
        from prompture.agents.agent import Agent
        from prompture.agents.async_agent import AsyncAgent

        for cls in (Agent, AsyncAgent):
            agent = cls("openai/gpt-4o", system_prompt="You are a copywriter.", brand=brand)
            prompt = agent._resolve_system_prompt()
            assert prompt.startswith("You are a copywriter.")
            assert "## Brand context" in prompt

    def test_extract_with_model_puts_brand_in_system_prompt(self, brand: BrandContext, monkeypatch) -> None:
        from pydantic import BaseModel

        from prompture.drivers.base import Driver
        from prompture.extraction.core import extract_with_model

        class Out(BaseModel):
            headline: str

        seen: list[str] = []

        class StubDriver(Driver):
            supports_json_mode = True
            supports_messages = True

            def __init__(self) -> None:
                self.model = "openai/test"

            def generate(self, prompt, options):
                seen.append(prompt)
                return {"text": '{"headline": "Dry by Friday"}', "meta": {"cost": 0.0, "total_tokens": 1}}

            def generate_messages(self, messages, options):
                seen.append(json.dumps(messages))
                return self.generate("", options)

        monkeypatch.setattr("prompture.extraction.core.get_driver_for_model", lambda *a, **kw: StubDriver())
        result = extract_with_model(
            Out, "Write a headline", model_name="openai/test", brand=brand, cache=False, ai_cleanup=False
        )
        assert result["model"].headline == "Dry by Friday"
        assert any("Brand context" in text for text in seen)


# ----------------------------------------------------------------------
# ClaimsChecker
# ----------------------------------------------------------------------


class TestClaimsRules:
    def test_clean_copy_passes(self, brand: BrandContext) -> None:
        copy = "Book a free inspection at northwindroofing.com or call (512) 555-0142. hello@northwindroofing.com"
        report = ClaimsChecker(brand).check(copy)
        assert report.passed, report.format()
        assert not report.grounding_ran

    def test_flags_invented_contact_details(self, brand: BrandContext) -> None:
        copy = "Visit www.northwind-roofs.com, email info@northwindroofing.com or call 1-800-ROOF-NOW."
        report = ClaimsChecker(brand).check(copy)
        kinds = {issue.kind for issue in report.issues}
        assert kinds == {"unknown_url", "unknown_email", "unknown_phone"}
        assert not report.passed

    def test_same_phone_in_another_format_is_fine(self, brand: BrandContext) -> None:
        assert ClaimsChecker(brand).check("Call +1 512.555.0142 today.").passed

    def test_dates_and_years_are_not_phones(self, brand: BrandContext) -> None:
        assert ClaimsChecker(brand).check("Open 2011 - 2025, 7am to 7pm.").passed

    def test_allowed_domains(self, brand: BrandContext) -> None:
        checker = ClaimsChecker(brand, allowed_domains=["calendly.com"])
        assert checker.check("Pick a time at https://calendly.com/northwind.").passed

    def test_avoid_words(self, brand: BrandContext) -> None:
        report = ClaimsChecker(brand).check("The cheap option is not us.")
        assert [i.kind for i in report.issues] == ["avoid_word"]


class TestClaimsGrounding:
    def test_unsupported_and_contradicted_claims(self, brand: BrandContext) -> None:
        driver = _ScriptedDriver(
            [
                json.dumps(
                    [
                        "Northwind has replaced 1,200 roofs since 2011.",
                        "Northwind has 5,000 five-star reviews.",
                        "Northwind was founded in 2020.",
                    ]
                ),
                "SUPPORTED",
                "SILENT",
                "CONTRADICTED",
            ]
        )
        report = ClaimsChecker(brand, driver).check("We've replaced 1,200 roofs, have 5,000 five-star reviews...")
        assert report.grounding_ran
        assert report.claims_checked == 3
        assert report.supported_claims == ["Northwind has replaced 1,200 roofs since 2011."]
        assert [i.kind for i in report.issues] == ["unsupported_claim", "contradicted_claim"]
        assert "1,200 roofs replaced since 2011" in driver.prompts[1]
        assert report.meta["calls"] == 4

    def test_copy_without_claims_passes(self, brand: BrandContext) -> None:
        report = ClaimsChecker(brand, _ScriptedDriver(["[]"])).check("Your roof deserves better.")
        assert report.passed
        assert report.claims_checked == 0


# ----------------------------------------------------------------------
# Output models
# ----------------------------------------------------------------------


def _post(**overrides: Any) -> dict[str, Any]:
    data = {
        "platform": "linkedin",
        "post_type": "contrarian",
        "hook": "Stop patching your roof every spring.",
        "copy_text": "Stop patching your roof every spring. Here is why.",
        "image_prompt": "A tired homeowner on a ladder at dusk, rain clouds, no text",
        "image_hook": "Patch once, pay twice",
        "image_subtext": "Why spring repairs cost more than you think",
        "hashtags": "roofing, #Austin homeowners",
    }
    data.update(overrides)
    return data


class TestSocialPost:
    def test_valid_post_and_hashtags(self) -> None:
        post = SocialPost(**_post())
        assert post.hashtags == ["#roofing", "#Austin", "#homeowners"]

    def test_hook_word_limit(self) -> None:
        with pytest.raises(ValidationError, match="at most 5 words"):
            SocialPost(**_post(image_hook="This hook is far too long"))

    def test_subtext_word_limit(self) -> None:
        with pytest.raises(ValidationError, match="at most 9 words"):
            SocialPost(**_post(image_subtext="one two three four five six seven eight nine ten"))

    def test_image_prompt_cannot_ask_for_text(self) -> None:
        with pytest.raises(ValidationError, match="rendered text"):
            SocialPost(**_post(image_prompt='A storefront with a sign that says "OPEN"'))

    def test_find_on_image_text_respects_negation(self) -> None:
        assert find_on_image_text("Clean product shot, no text, without captions") == []
        assert find_on_image_text("Poster with a bold headline") == ["headline"]


class TestVideo:
    def _clip(self, index: int, **overrides: Any) -> dict[str, Any]:
        data = {
            "index": index,
            "duration_s": 3,
            "description": "Drone shot over a neighborhood after rain",
            "camera": "high wide, slow push",
            "voiceover": "Storms don't wait.",
        }
        data.update(overrides)
        return data

    def test_shotlist_prompts(self) -> None:
        shots = ShotList(
            title="Dry by Friday",
            logline="A storm, a leak, a crew that shows up.",
            style_prefix="Photoreal, natural light.",
            clips=[self._clip(1), self._clip(2, voiceover="", dialogue="We'll be there tomorrow.")],
        )
        assert shots.total_duration_s == 6
        assert shots.voiceover_script == "Storms don't wait.\nWe'll be there tomorrow."
        prompt = shots.prompts()[0]
        assert prompt.startswith("Photoreal, natural light.")
        payload = json.loads(prompt.split("\n\n", 1)[1])
        assert payload["text"] == "None"
        assert "index" not in payload

    def test_em_dash_in_voiceover_rejected(self) -> None:
        with pytest.raises(ValidationError, match="em dash"):
            VideoClip(**self._clip(1, voiceover="Storms come — we come faster."))

    def test_on_screen_text_must_be_empty(self) -> None:
        assert VideoClip(**self._clip(1, on_screen_text="None")).on_screen_text == ""
        with pytest.raises(ValidationError):
            VideoClip(**self._clip(1, on_screen_text="SALE 50% OFF"))

    def test_clip_order_and_voiceover_budget(self) -> None:
        with pytest.raises(ValidationError, match="run 1..2"):
            ShotList(title="t", logline="l", clips=[self._clip(2), self._clip(1)])
        too_long = " ".join(["word"] * 20)
        with pytest.raises(ValidationError, match="do not fit"):
            ShotList(title="t", logline="l", clips=[self._clip(1, voiceover=too_long)])


class TestStrategyModels:
    def test_ideal_customer_to_brand_fields(self) -> None:
        icp = IdealCustomer(
            nickname="Leaky-Ceiling Linda",
            name="Linda Park",
            age_range="45-55",
            role="Homeowner",
            location="Austin",
            description="Owns a 1990s home.",
            pains=["a", "b", "c"],
            objections=["x", "y", "z"],
        )
        fields = icp.to_brand_fields()
        assert fields["ideal_customer"].startswith("Leaky-Ceiling Linda (Linda Park, 45-55, Homeowner, Austin)")

    def test_top_tens_needs_ten_each(self) -> None:
        ten = [f"item {i}" for i in range(10)]
        TopTens(**dict.fromkeys(TopTens.model_fields, ten))
        with pytest.raises(ValidationError):
            TopTens(**{**dict.fromkeys(TopTens.model_fields, ten), "fears": ten[:9]})

    def test_lead_magnets_ranked_by_value(self) -> None:
        ideas = LeadMagnetIdeas(
            ideas=[
                {
                    "title": "Guide",
                    "format": "guide",
                    "problem_solved": "p",
                    "dream_outcome": 6,
                    "likelihood": 5,
                    "time_delay": 5,
                    "effort": 5,
                },
                {
                    "title": "Quiz",
                    "format": "quiz",
                    "problem_solved": "p",
                    "dream_outcome": 8,
                    "likelihood": 8,
                    "time_delay": 1,
                    "effort": 2,
                },
                {
                    "title": "Course",
                    "format": "course",
                    "problem_solved": "p",
                    "dream_outcome": 9,
                    "likelihood": 6,
                    "time_delay": 8,
                    "effort": 8,
                },
            ]
        )
        assert [idea.title for idea in ideas.ranked()] == ["Quiz", "Guide", "Course"]
        assert ideas.ranked()[0].value_score == 32.0

    def test_offer_stack_totals(self) -> None:
        stack = OfferStack(
            name="Storm Ready",
            dream_outcome="A dry house",
            items=[
                {"name": "Inspection", "solves": "unknown damage", "delivery": "visit", "value": 300},
                {"name": "Repair", "solves": "leak", "delivery": "crew", "value": 1200},
                {"name": "Warranty", "solves": "worry", "delivery": "document", "value": 500},
            ],
            price=1000,
        )
        assert stack.total_value == 2000
        assert stack.value_ratio == 2.0

    def test_email_sequence_days_increase(self) -> None:
        email = {"subject": "s", "body": "b"}
        EmailSequence(goal="g", emails=[{**email, "send_day": 0}, {**email, "send_day": 2}])
        with pytest.raises(ValidationError, match="send_day must increase"):
            EmailSequence(goal="g", emails=[{**email, "send_day": 2}, {**email, "send_day": 2}])
        with pytest.raises(ValidationError, match="at most 60 characters"):
            EmailSequence(goal="g", emails=[{**email, "subject": "x" * 61, "send_day": 0}, {**email, "send_day": 1}])


class TestScorecard:
    def test_analyze_metric(self) -> None:
        trend = analyze_metric("Leads", [10, 12, None, 15, 18, 21], goal=15, owner="Sam")
        assert trend.weeks_reported == 5
        assert trend.best_week == 6 and trend.worst_week == 1
        assert trend.weeks_on_goal == 3
        assert trend.trend == "rising"

    def test_lower_is_better(self) -> None:
        trend = analyze_metric("Cost per lead", [40, 35, 30, 28], goal=30, higher_is_better=False)
        assert trend.best_week == 4 and trend.worst_week == 1
        assert trend.weeks_on_goal == 2
        assert trend.trend == "falling"

    def test_volatile_and_empty(self) -> None:
        assert analyze_metric("Views", [100, 5000, 50, 4000]).trend == "volatile"
        assert analyze_metric("Views", [None, None]).trend == "no data"

    def test_brief(self) -> None:
        brief = scorecard_brief([analyze_metric("Leads", [10, None, 12], goal=11, owner="Sam")])
        assert "Leads (owner: Sam)" in brief
        assert "weekly: 10, -, 12" in brief


def test_word_count() -> None:
    assert word_count("Don't stop—ever, well-known fact") == 5
