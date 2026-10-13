"""Typed outputs for common marketing work, with their limits enforced.

Each model encodes rules that copywriters and producers follow by hand: a
headline overlay fits in five words, a video prompt asks for no on-screen text,
a voiceover fits the clip's length, an email sequence goes out in order.  The
rules are validators, so ``extract_with_model`` retries with the exact
violation instead of hoping the model counted words.

Strategy:
    :class:`IdealCustomer`, :class:`CoreMessaging`, :class:`TopTens`,
    :class:`LeadMagnetIdeas`, :class:`OfferStack`
Production:
    :class:`SocialPostSet`, :class:`EmailSequence`, :class:`ShotList`
Measurement:
    :class:`ScorecardAnalysis`, plus :func:`analyze_metric` for the arithmetic

The strategy models have ``to_brand_fields()``, so
``brand.with_output(messaging)`` folds a generated result back into the
:class:`~prompture.marketing.BrandContext` for every later call.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

__all__ = [
    "ActionItem",
    "AudienceMessage",
    "CoreMessaging",
    "Email",
    "EmailSequence",
    "IdealCustomer",
    "LeadMagnet",
    "LeadMagnetIdeas",
    "MetricTrend",
    "OfferItem",
    "OfferStack",
    "ScorecardAnalysis",
    "ShotList",
    "SocialPost",
    "SocialPostSet",
    "TopTens",
    "VideoClip",
    "WeekReview",
    "analyze_metric",
    "find_on_image_text",
    "scorecard_brief",
    "word_count",
]

#: Comfortable voiceover pace.  Faster than this sounds rushed in a short spot.
VOICEOVER_WORDS_PER_SECOND = 2.5


# ----------------------------------------------------------------------
# Shared validators
# ----------------------------------------------------------------------


def word_count(text: str) -> int:
    """Count words the way a reader would (hyphenated words count once)."""
    return len(re.findall(r"[\w'’-]+", text or ""))


def _max_words(limit: int):
    def check(value: str) -> str:
        count = word_count(value)
        if count > limit:
            raise ValueError(f"must be at most {limit} words; got {count}: {value!r}")
        return value

    return AfterValidator(check)


def _max_chars(limit: int):
    def check(value: str) -> str:
        if len(value) > limit:
            raise ValueError(f"must be at most {limit} characters; got {len(value)}")
        return value

    return AfterValidator(check)


def _no_em_dash(value: str) -> str:
    # Video and voice models read an em dash unpredictably: a stall, a skipped
    # word, or a garbled pause.  Commas, periods and ellipses read cleanly.
    if "—" in value:
        raise ValueError("spoken lines must not contain an em dash; use a comma, period or ellipsis")
    return value


_TEXT_WORDS = re.compile(
    r"\b(text|caption|captions|headline|headlines|lettering|typography|slogan|that says|that reads|with the words)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(no|without|avoid|zero|free of|not any)\s+(any\s+|on-image\s+|visible\s+)?$", re.IGNORECASE)
_QUOTED = re.compile(r"[\"“]([^\"”]{2,})[\"”]")


def find_on_image_text(prompt: str) -> list[str]:
    """Return the parts of an image or video prompt that ask for rendered text.

    Image and video models misspell words and warp letters, so overlay text
    belongs in a design tool, not the generation prompt.  "No text" and
    "without captions" are fine; "a sign that says OPEN" and quoted words are not.
    """
    found: list[str] = []
    for match in _TEXT_WORDS.finditer(prompt or ""):
        before = prompt[max(0, match.start() - 24) : match.start()]
        if not _NEGATION.search(before):
            found.append(match.group(0))
    found.extend(f'"{m.group(1)}"' for m in _QUOTED.finditer(prompt or ""))
    return found


def _no_on_image_text(value: str) -> str:
    found = find_on_image_text(value)
    if found:
        raise ValueError(
            "the image prompt asks for rendered text (" + ", ".join(found) + "); describe only the visual and put "
            "words in image_hook / image_subtext"
        )
    return value


Score10 = Annotated[int, Field(ge=1, le=10)]


# ----------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------


class IdealCustomer(BaseModel):
    """One specific buyer, detailed enough to recognize in a room."""

    nickname: Annotated[str, _max_words(5)] = Field(
        description="A vivid label naming their situation, e.g. 'Stuck-in-the-Weeds Sandra'. At most 5 words."
    )
    name: str = Field(description="A realistic first and last name.")
    age_range: str
    role: str = Field(description="Job title or life situation.")
    income: str = ""
    location: str = ""
    description: str = Field(description="3-5 sentences on who they are, their ambition and their daily reality.")
    goals: list[str] = Field(default_factory=list)
    pains: list[str] = Field(
        min_length=3, max_length=7, description="Problems in the customer's own words, not marketing language."
    )
    fears: list[str] = Field(default_factory=list)
    objections: list[str] = Field(min_length=3, max_length=7)
    triggers: list[str] = Field(
        default_factory=list, description="What recently happened that made them start looking for a solution."
    )
    search_terms: list[str] = Field(default_factory=list, description="Phrases they would type into a search box.")
    already_tried: list[str] = Field(default_factory=list, description="What they tried before and why it fell short.")
    channels: list[str] = Field(default_factory=list, description="Where they discover and buy.")

    def to_brand_fields(self) -> dict[str, Any]:
        summary = f"{self.nickname} ({self.name}, {self.age_range}, {self.role}"
        summary += f", {self.location})" if self.location else ")"
        return {"ideal_customer": f"{summary}. {self.description}", "pains": self.pains, "objections": self.objections}


class AudienceMessage(BaseModel):
    """One core message for one awareness level, with the proof behind it."""

    message: Annotated[str, _max_words(30)]
    proof: list[str] = Field(default_factory=list, description="Evidence that makes the message believable.")


class CoreMessaging(BaseModel):
    """An umbrella statement plus one message per audience temperature.

    Cold: knows the problem, not the solutions.  Warm: comparing solutions, so
    say what makes this one different.  Hot: ready to buy, so give certainty.
    """

    umbrella_statement: Annotated[str, _max_words(12)]
    alternates: list[Annotated[str, _max_words(12)]] = Field(default_factory=list, max_length=5)
    cold: AudienceMessage
    warm: AudienceMessage
    hot: AudienceMessage

    def to_brand_fields(self) -> dict[str, Any]:
        return {
            "umbrella_statement": self.umbrella_statement,
            "cold_message": self.cold.message,
            "warm_message": self.warm.message,
            "hot_message": self.hot.message,
        }


_Ten = Annotated[list[str], Field(min_length=10, max_length=10)]


class TopTens(BaseModel):
    """Ten of each thing a buyer brings to the decision.

    Content built to answer these lists answers the questions buyers actually have.
    """

    objections: _Ten
    fears: _Ten
    questions: _Ten
    emotional_motivations: _Ten
    financial_motivations: _Ten
    limiting_beliefs: _Ten
    search_questions: _Ten = Field(description="Questions they type into a search engine.")

    def to_brand_fields(self) -> dict[str, Any]:
        return {
            "objections": self.objections,
            "questions": self.questions,
            "motivations": [*self.emotional_motivations[:5], *self.financial_motivations[:5]],
        }


class LeadMagnet(BaseModel):
    """A free offer scored on the value equation.

    Value = (dream outcome x likelihood of achieving it) / (time delay x effort).
    Scores run 1-10; for time delay and effort, lower is better.
    """

    title: str
    format: str = Field(description="quiz, scorecard, checklist, guide, template, tool, webinar, mini course, ...")
    problem_solved: str
    dream_outcome: Score10
    likelihood: Score10
    time_delay: Score10 = Field(description="1 = instant result, 10 = months away.")
    effort: Score10 = Field(description="1 = effortless, 10 = a lot of work.")

    @property
    def value_score(self) -> float:
        return round(self.dream_outcome * self.likelihood / (self.time_delay * self.effort), 3)


class LeadMagnetIdeas(BaseModel):
    ideas: list[LeadMagnet] = Field(min_length=3)

    def ranked(self) -> list[LeadMagnet]:
        """Ideas sorted by value score, best first."""
        return sorted(self.ideas, key=lambda idea: idea.value_score, reverse=True)


class OfferItem(BaseModel):
    name: str
    solves: str = Field(description="The problem or obstacle this item removes.")
    delivery: str = Field(description="How it is delivered: call, course, template, done-for-you, community, ...")
    value: float = Field(ge=0, description="Stand-alone value in the offer's currency.")


class OfferStack(BaseModel):
    """An offer built as a stack of problem-solving items with stated values."""

    name: str
    dream_outcome: str
    items: list[OfferItem] = Field(min_length=3)
    price: float | None = Field(default=None, ge=0)
    guarantee: str = ""

    @property
    def total_value(self) -> float:
        return round(sum(item.value for item in self.items), 2)

    @property
    def value_ratio(self) -> float | None:
        """Total stated value divided by price (``None`` without a price)."""
        if not self.price:
            return None
        return round(self.total_value / self.price, 2)


# ----------------------------------------------------------------------
# Production
# ----------------------------------------------------------------------


class SocialPost(BaseModel):
    """A post plus the creative brief for its image.

    The image prompt describes only the visual.  The words that go on the image
    are kept separate (hook and subtext) and added in a design tool.
    """

    platform: str
    post_type: Literal["insight", "trend", "question", "contrarian", "story", "proof", "offer"] = Field(
        description="insight: teach something surprising; trend: a fresh angle on news; question: answer a top "
        "buyer question; contrarian: flip common advice; story; proof; offer."
    )
    hook: Annotated[str, _max_words(15)] = Field(description="The first line. At most 15 words.")
    copy_text: str = Field(description="The full post copy, starting with the hook.")
    image_prompt: Annotated[str, AfterValidator(_no_on_image_text)] = Field(
        description="Visual description only. No words, captions, signs, screens or UI."
    )
    image_hook: Annotated[str, _max_words(5)] = Field(description="Overlay headline. At most 5 words.")
    image_subtext: Annotated[str, _max_words(9)] = Field(description="Overlay subline. At most 9 words.")
    hashtags: list[str] = Field(default_factory=list, max_length=10)
    call_to_action: str = ""

    @field_validator("hashtags", mode="before")
    @classmethod
    def _normalize_hashtags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = value.replace(",", " ").split()
        tags = []
        for tag in value:
            cleaned = re.sub(r"\s+", "", str(tag)).lstrip("#")
            if cleaned:
                tags.append(f"#{cleaned}")
        return tags


class SocialPostSet(BaseModel):
    posts: list[SocialPost] = Field(min_length=1)


class Email(BaseModel):
    subject: Annotated[str, _max_chars(60)]
    preview_text: Annotated[str, _max_chars(90)] = ""
    body: str
    ps: str = ""
    send_day: int = Field(ge=0, description="Days after the sequence starts (0 = the first day).")


class EmailSequence(BaseModel):
    goal: str
    emails: list[Email] = Field(min_length=2)

    @model_validator(mode="after")
    def _send_days_in_order(self) -> EmailSequence:
        days = [email.send_day for email in self.emails]
        if any(later <= earlier for earlier, later in itertools.pairwise(days)):
            raise ValueError(f"send_day must increase from one email to the next; got {days}")
        return self


class VideoClip(BaseModel):
    """One text-to-video generation: a single shot with its spoken line."""

    index: int = Field(ge=1)
    duration_s: float = Field(gt=0, le=60)
    description: Annotated[str, AfterValidator(_no_on_image_text)]
    style: str = ""
    camera: str = Field(description="Shot type, lens and movement.")
    lighting: str = ""
    setting: str = ""
    elements: list[str] = Field(default_factory=list)
    motion: str = ""
    ending: str = Field(default="", description="The final moment of the shot.")
    voiceover: Annotated[str, AfterValidator(_no_em_dash)] = ""
    dialogue: Annotated[str, AfterValidator(_no_em_dash)] = Field(default="", description="Words spoken on camera.")
    on_screen_text: Literal["", "none"] = Field(
        default="", description="Always empty: text is added in editing, never generated."
    )
    keywords: list[str] = Field(default_factory=list)

    @field_validator("on_screen_text", mode="before")
    @classmethod
    def _normalize_none(cls, value: Any) -> str:
        if value is None or str(value).strip().lower() in ("", "none", "n/a", "no text"):
            return ""
        return str(value)

    @property
    def spoken_words(self) -> int:
        return word_count(self.voiceover) + word_count(self.dialogue)

    def to_prompt(self, style_prefix: str = "") -> str:
        """The clip as a JSON prompt for a text-to-video model."""
        payload = self.model_dump(exclude={"index", "on_screen_text"}, exclude_defaults=True)
        payload["text"] = "None"
        body = json.dumps(payload, indent=2, ensure_ascii=False)
        return f"{style_prefix.strip()}\n\n{body}" if style_prefix.strip() else body


class ShotList(BaseModel):
    """A short video as an ordered list of clips."""

    title: str
    logline: Annotated[str, _max_chars(300)]
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:5"] = "16:9"
    style_prefix: str = Field(default="", description="Look and feel shared by every clip.")
    clips: list[VideoClip] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_clips(self) -> ShotList:
        indexes = [clip.index for clip in self.clips]
        if indexes != list(range(1, len(self.clips) + 1)):
            raise ValueError(f"clip indexes must run 1..{len(self.clips)} in order; got {indexes}")
        budget = math.floor(self.total_duration_s * VOICEOVER_WORDS_PER_SECOND)
        spoken = sum(clip.spoken_words for clip in self.clips)
        if spoken > budget:
            raise ValueError(
                f"{spoken} spoken words do not fit {self.total_duration_s:g}s; keep it to {budget} words or fewer"
            )
        return self

    @property
    def total_duration_s(self) -> float:
        return sum(clip.duration_s for clip in self.clips)

    @property
    def voiceover_script(self) -> str:
        return "\n".join(line for clip in self.clips for line in (clip.voiceover, clip.dialogue) if line)

    def prompts(self) -> list[str]:
        """One ready-to-paste prompt per clip, each with the style prefix."""
        return [clip.to_prompt(self.style_prefix) for clip in self.clips]


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------


class MetricTrend(BaseModel):
    """One scorecard metric over the period.  The numbers come from :func:`analyze_metric`."""

    metric: str
    owner: str = ""
    goal: float | None = None
    values: list[float | None] = Field(default_factory=list, description="One value per week; null if not reported.")
    average: float | None = None
    weeks_reported: int = 0
    weeks_on_goal: int | None = None
    best_week: int | None = Field(default=None, description="1-based week number.")
    worst_week: int | None = Field(default=None, description="1-based week number.")
    trend: Literal["rising", "falling", "flat", "volatile", "no data"] = "no data"
    diagnosis: str = Field(default="", description="Likely cause and where to look next.")


class WeekReview(BaseModel):
    week: int = Field(ge=1)
    wins: list[str] = Field(default_factory=list)
    needs_attention: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    who: str
    what: str
    when: str


class ScorecardAnalysis(BaseModel):
    """A weekly scorecard read: data notes first, then trends, then who does what next."""

    data_notes: list[str] = Field(
        default_factory=list, description="Metrics found, gaps, and anything that could not be read."
    )
    metrics: list[MetricTrend] = Field(min_length=1)
    cross_metric_insights: list[str] = Field(
        default_factory=list, description="Relationships between metrics, e.g. clicks up while leads stay flat."
    )
    weekly_checklist: list[WeekReview] = Field(default_factory=list)
    next_actions: list[ActionItem] = Field(min_length=1)


def analyze_metric(
    metric: str,
    values: list[float | int | None],
    *,
    goal: float | None = None,
    owner: str = "",
    higher_is_better: bool = True,
) -> MetricTrend:
    """Compute a metric's trend without a model.

    Do the arithmetic here and let the model write only the diagnosis: it is
    faster, cheaper, and the numbers are right.

    The trend is the least-squares slope across reported weeks relative to
    their mean: more than +/-2% a week is ``rising``/``falling``.  A
    coefficient of variation above 0.5 is ``volatile`` regardless of slope.
    For metrics where lower is better (cost per lead), pass
    ``higher_is_better=False``: best/worst weeks and the goal check flip.
    """
    cleaned = [None if value is None else float(value) for value in values]
    reported = [(week, value) for week, value in enumerate(cleaned, start=1) if value is not None]
    trend = MetricTrend(metric=metric, owner=owner, goal=goal, values=cleaned, weeks_reported=len(reported))
    if not reported:
        return trend

    numbers = [value for _, value in reported]
    mean = statistics.fmean(numbers)
    trend.average = round(mean, 4)
    pick_best = max if higher_is_better else min
    pick_worst = min if higher_is_better else max
    trend.best_week = pick_best(reported, key=lambda pair: pair[1])[0]
    trend.worst_week = pick_worst(reported, key=lambda pair: pair[1])[0]
    if goal is not None:
        trend.weeks_on_goal = sum(1 for value in numbers if (value >= goal if higher_is_better else value <= goal))

    if len(reported) < 3 or mean == 0:
        trend.trend = "flat"
        return trend
    weeks = [week for week, _ in reported]
    week_mean = statistics.fmean(weeks)
    denominator = sum((week - week_mean) ** 2 for week in weeks)
    slope = sum((week - week_mean) * (value - mean) for week, value in reported) / denominator
    relative_slope = slope / abs(mean)
    variation = statistics.pstdev(numbers) / abs(mean)
    if variation > 0.5:
        trend.trend = "volatile"
    elif relative_slope > 0.02:
        trend.trend = "rising"
    elif relative_slope < -0.02:
        trend.trend = "falling"
    else:
        trend.trend = "flat"
    return trend


def scorecard_brief(trends: list[MetricTrend]) -> str:
    """Render computed trends as a compact block to put in front of the model."""
    lines = []
    for trend in trends:
        values = ", ".join("-" if value is None else f"{value:g}" for value in trend.values)
        parts = [f"{trend.metric}" + (f" (owner: {trend.owner})" if trend.owner else "")]
        if trend.goal is not None:
            parts.append(f"goal {trend.goal:g}/week, met {trend.weeks_on_goal} of {trend.weeks_reported} weeks")
        if trend.average is not None:
            parts.append(f"average {trend.average:g}, trend {trend.trend}")
            parts.append(f"best week {trend.best_week}, worst week {trend.worst_week}")
        lines.append("- " + "; ".join(parts) + f"\n  weekly: {values}")
    return "\n".join(lines)
