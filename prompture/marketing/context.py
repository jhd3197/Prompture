"""Brand context: the facts every marketing call should start from.

Generic prompts produce generic copy.  The fix is to put the same business
facts in front of the model on every call: who the customer is, what hurts,
what the business says about itself, which proof is real, and which words are
off limits.  :class:`BrandContext` holds those facts in one typed object and
renders them as a system-prompt block.

It plugs into the rest of Prompture in four places:

* ``Agent(..., brand=brand)`` / ``AsyncAgent(..., brand=brand)``
* ``Assistant(..., brand=brand)``
* ``extract_with_model(..., brand=brand)`` (sync and async)
* ``brand.as_persona()`` / ``brand.apply_to(persona)`` for anything that takes a
  :class:`~prompture.agents.persona.Persona`

Load one from YAML, JSON, or a fill-in Markdown file::

    from prompture.marketing import BrandContext

    BrandContext.save_markdown_template("brand.md")   # hand this to the client
    brand = BrandContext.load("brand.md")              # after they fill it in

The rendered block ends with rules that forbid stating numbers, testimonials or
contact details that are not listed in the context.  :class:`~prompture.marketing.ClaimsChecker`
checks the output against the same facts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "BrandColors",
    "BrandContext",
    "load_brand",
    "merge_brand_system_prompt",
]

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class BrandColors(BaseModel):
    """Brand colors as hex codes (``#1E2761``).  ``#abc`` is expanded to ``#AABBCC``."""

    model_config = ConfigDict(extra="forbid")

    primary: str | None = None
    secondary: str | None = None
    accent: str | None = None

    @field_validator("primary", "secondary", "accent", mode="before")
    @classmethod
    def _normalize_hex(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        match = _HEX_RE.match(text)
        if not match:
            raise ValueError(f"{text!r} is not a hex color such as #1E2761")
        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(ch * 2 for ch in digits)
        return f"#{digits.upper()}"

    def is_empty(self) -> bool:
        return not (self.primary or self.secondary or self.accent)


class BrandContext(BaseModel):
    """Everything a marketing call needs to know about one business.

    Only ``name`` is required.  Empty fields are left out of the rendered
    block, so a half-filled context still works.

    Attributes:
        name: Business or brand name.
        offer: What the business sells, in one line.
        tagline: Current tagline, if any.
        website: Main website URL.  Treated as a verified fact.
        call_to_action: The main call to action (book a call, shop, DM a keyword).
        contact: Verified contact details (``{"phone": ..., "email": ...}``).
            The claims check flags any phone number or email not listed here.
        ideal_customer: A short paragraph describing the buyer.
        pains: The customer's problems, in their own words.
        objections: Reasons they give for not buying.
        motivations: What pulls them toward buying.
        questions: Questions they ask before buying.
        umbrella_statement: The one line that sums up the brand's message.
        cold_message: Message for people who know the problem but not the solutions.
        warm_message: Message for people comparing solutions.
        hot_message: Message for people ready to buy.
        offers: Products, services, packages or lead magnets.
        voice: Tone words (``["direct", "warm", "plain-spoken"]``).
        use_words: Words and phrases the brand uses often.
        avoid_words: Words the brand never uses.  The claims check flags them.
        proof_points: Verified facts: results, numbers, credentials, client names.
            These are the only claims the model may state as fact.
        testimonials: Verified quotes with attribution.
        colors: Brand colors.
        notes: Anything else, keyed by a short label.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    offer: str = ""
    tagline: str = ""
    website: str = ""
    call_to_action: str = ""
    contact: dict[str, str] = Field(default_factory=dict)
    ideal_customer: str = ""
    pains: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    umbrella_statement: str = ""
    cold_message: str = ""
    warm_message: str = ""
    hot_message: str = ""
    offers: list[str] = Field(default_factory=list)
    voice: list[str] = Field(default_factory=list)
    use_words: list[str] = Field(default_factory=list)
    avoid_words: list[str] = Field(default_factory=list)
    proof_points: list[str] = Field(default_factory=list)
    testimonials: list[str] = Field(default_factory=list)
    colors: BrandColors = Field(default_factory=BrandColors)
    notes: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _require_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("a brand needs a name")
        return value

    @field_validator(
        "pains",
        "objections",
        "motivations",
        "questions",
        "offers",
        "voice",
        "use_words",
        "avoid_words",
        "proof_points",
        "testimonials",
        mode="before",
    )
    @classmethod
    def _clean_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = value.split(",")
        return [str(item).strip() for item in value if str(item).strip()]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Render the context as a system-prompt block, ending with the brand rules."""
        lines: list[str] = ["## Brand context"]

        basics = [
            ("Business", self.name),
            ("What we sell", self.offer),
            ("Tagline", self.tagline),
            ("Website", self.website),
            ("Call to action", self.call_to_action),
        ]
        basics.extend((label.title(), value) for label, value in self.contact.items())
        lines.extend(f"{label}: {value}" for label, value in basics if value)

        if self.ideal_customer:
            lines += ["", "### Ideal customer", self.ideal_customer]

        for heading, items in (
            ("Their pains, in their words", self.pains),
            ("Their objections", self.objections),
            ("What motivates them", self.motivations),
            ("Questions they ask", self.questions),
        ):
            _append_list(lines, heading, items)

        messaging = [
            ("Umbrella statement", self.umbrella_statement),
            ("Cold audience (knows the problem, not the solutions)", self.cold_message),
            ("Warm audience (comparing solutions)", self.warm_message),
            ("Hot audience (ready to buy)", self.hot_message),
        ]
        if any(value for _, value in messaging):
            lines += ["", "### Core messaging"]
            lines.extend(f"{label}: {value}" for label, value in messaging if value)

        _append_list(lines, "Offers", self.offers)

        voice = [
            ("Tone", ", ".join(self.voice)),
            ("Words we use", ", ".join(self.use_words)),
            ("Words we never use", ", ".join(self.avoid_words)),
        ]
        if any(value for _, value in voice):
            lines += ["", "### Voice"]
            lines.extend(f"{label}: {value}" for label, value in voice if value)

        _append_list(lines, "Verified proof (the only results and numbers you may state)", self.proof_points)
        _append_list(lines, "Verified testimonials (quote exactly, never invent others)", self.testimonials)

        if not self.colors.is_empty():
            lines += ["", "### Brand colors"]
            lines.extend(
                f"{label}: {value}"
                for label, value in (
                    ("Primary", self.colors.primary),
                    ("Secondary", self.colors.secondary),
                    ("Accent", self.colors.accent),
                )
                if value
            )

        if self.notes:
            lines += ["", "### Notes"]
            lines.extend(f"- {label}: {value}" for label, value in self.notes.items())

        lines += ["", "### Brand rules", *(f"- {rule}" for rule in self.rules())]
        return "\n".join(lines)

    def rules(self) -> list[str]:
        """The guardrails appended to every rendered block."""
        rules = [
            "Write for the ideal customer above, using their words for their problems.",
            "State numbers, results, awards, credentials, client names, prices and guarantees only when "
            "they appear under Verified proof. If a claim would help but is not listed, leave it out or "
            "write [NEEDS PROOF].",
            "Quote only the verified testimonials. Never write a testimonial, review or customer quote yourself.",
            "Use only the website, phone number, email and links listed here. Never make up contact details.",
        ]
        if self.avoid_words:
            rules.append("Never use these words: " + ", ".join(self.avoid_words) + ".")
        return rules

    def variables(self) -> dict[str, str]:
        """Template variables for ``{{brand_*}}`` placeholders in personas and prompts."""
        return {
            "brand_name": self.name,
            "brand_offer": self.offer,
            "brand_tagline": self.tagline,
            "brand_website": self.website,
            "brand_cta": self.call_to_action,
            "brand_voice": ", ".join(self.voice),
            "brand_umbrella_statement": self.umbrella_statement,
        }

    def evidence(self) -> list[str]:
        """The verified facts, one per line, for grounding checks."""
        facts = [f"The business is called {self.name}."]
        if self.offer:
            facts.append(f"{self.name} sells: {self.offer}")
        if self.tagline:
            facts.append(f"The tagline is: {self.tagline}")
        if self.website:
            facts.append(f"The website is {self.website}")
        facts.extend(f"The {label} is {value}" for label, value in self.contact.items() if value)
        facts.extend(f"Offer: {offer}" for offer in self.offers)
        facts.extend(self.proof_points)
        facts.extend(f"Testimonial: {quote}" for quote in self.testimonials)
        return facts

    def as_persona(self, base: Any = None, *, name: str | None = None) -> Any:
        """Return a :class:`~prompture.agents.persona.Persona` carrying this context.

        With ``base``, the context is appended to that persona and its
        variables gain the ``brand_*`` values.  Without it, a persona whose
        whole prompt is the context block is returned.
        """
        from ..agents.persona import Persona

        if base is None:
            return Persona(
                name=name or _slug(self.name),
                system_prompt=self.render(),
                description=f"Brand context for {self.name}",
                variables=self.variables(),
            )
        return self.apply_to(base)

    def apply_to(self, persona: Any) -> Any:
        """Append this context to an existing persona (persona variables win on conflict)."""
        import dataclasses

        extended = persona.extend(self.render())
        return dataclasses.replace(extended, variables={**self.variables(), **persona.variables})

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def updated(self, **changes: Any) -> BrandContext:
        """Return a validated copy with ``changes`` applied."""
        return BrandContext.model_validate({**self.model_dump(), **changes})

    def with_output(self, output: BaseModel) -> BrandContext:
        """Fold a generated marketing model back into the context.

        Accepts :class:`~prompture.marketing.IdealCustomer`,
        :class:`~prompture.marketing.CoreMessaging` and
        :class:`~prompture.marketing.CustomerInsights`.  This is how a strategy
        step's output becomes context for every later step.
        """
        if hasattr(output, "to_brand_fields"):
            return self.updated(**output.to_brand_fields())
        raise TypeError(f"{type(output).__name__} has no brand fields to merge")

    # ------------------------------------------------------------------
    # Loading and saving
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> BrandContext:
        """Load from ``.yaml``, ``.yml``, ``.json`` or ``.md`` by file extension."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        if suffix == ".json":
            return cls.from_json(path)
        if suffix in (".md", ".markdown", ".txt"):
            return cls.from_markdown(path.read_text(encoding="utf-8"))
        raise ValueError(f"Unsupported brand file type {suffix!r}; use .yaml, .json or .md")

    @classmethod
    def from_yaml(cls, path: str | Path) -> BrandContext:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: str | Path) -> BrandContext:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_yaml(self, path: str | Path) -> None:
        import yaml

        data = self.model_dump(exclude_defaults=True)
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.model_dump(exclude_defaults=True), indent=2), encoding="utf-8")

    @classmethod
    def from_markdown(cls, text: str) -> BrandContext:
        """Parse a filled-in copy of :meth:`markdown_template`.

        The format is forgiving: headings and labels are matched
        case-insensitively, lines in parentheses are treated as hints and
        skipped, and unknown sections land in ``notes``.
        """
        return cls.model_validate(_parse_markdown(text))

    @staticmethod
    def markdown_template() -> str:
        """A blank brand file for a person to fill in."""
        return _MARKDOWN_TEMPLATE

    @staticmethod
    def save_markdown_template(path: str | Path) -> None:
        Path(path).write_text(_MARKDOWN_TEMPLATE, encoding="utf-8")


def load_brand(source: BrandContext | dict[str, Any] | str | Path) -> BrandContext:
    """Coerce a context, a dict, or a file path into a :class:`BrandContext`."""
    if isinstance(source, BrandContext):
        return source
    if isinstance(source, dict):
        return BrandContext.model_validate(source)
    return BrandContext.load(source)


def merge_brand_system_prompt(system_prompt: str | None, brand: Any) -> str | None:
    """Prepend a brand block to ``system_prompt``.  Returns ``system_prompt`` unchanged without a brand."""
    if brand is None:
        return system_prompt
    block = load_brand(brand).render()
    return f"{block}\n\n{system_prompt}" if system_prompt else block


def _append_list(lines: list[str], heading: str, items: list[str]) -> None:
    if items:
        lines += ["", f"### {heading}", *(f"- {item}" for item in items)]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "brand"


# ----------------------------------------------------------------------
# Markdown fill-in format
# ----------------------------------------------------------------------

_MARKDOWN_TEMPLATE = """# Brand: fill this in

Short answers are fine. Leave anything that does not apply blank.
Only list proof and testimonials you can back up: the model is told these are the only claims it may make.

## Basics
- Name:
- What we sell:
- Tagline:
- Website:
- Call to action:
- Phone:
- Email:

## Ideal customer
(One short paragraph: who buys, their role or situation, and what recently made them start looking for help.)

## Pains
(Their problems, written the way they would say them to a friend after a bad day.)
-

## Objections
-

## Motivations
-

## Questions they ask
-

## Core messaging
- Umbrella statement:
- Cold audience:
- Warm audience:
- Hot audience:

## Offers
-

## Voice
- Tone:
- Words we use:
- Words we never use:

## Proof points
(Results, numbers, credentials and client names you can prove.)
-

## Testimonials
(Real quotes with the person's name.)
-

## Colors
- Primary:
- Secondary:
- Accent:
"""

# heading (normalized) -> (kind, target)
_SECTIONS: dict[str, tuple[str, str]] = {
    "basics": ("kv", "basics"),
    "the basics": ("kv", "basics"),
    "business": ("kv", "basics"),
    "ideal customer": ("text", "ideal_customer"),
    "ideal customer persona": ("text", "ideal_customer"),
    "customer": ("text", "ideal_customer"),
    "pains": ("list", "pains"),
    "pain points": ("list", "pains"),
    "problems": ("list", "pains"),
    "objections": ("list", "objections"),
    "motivations": ("list", "motivations"),
    "desires": ("list", "motivations"),
    "questions they ask": ("list", "questions"),
    "questions": ("list", "questions"),
    "core messaging": ("kv", "messaging"),
    "messaging": ("kv", "messaging"),
    "offers": ("list", "offers"),
    "offer": ("list", "offers"),
    "voice": ("kv", "voice"),
    "brand voice": ("kv", "voice"),
    "proof points": ("list", "proof_points"),
    "proof": ("list", "proof_points"),
    "proof i can use": ("list", "proof_points"),
    "testimonials": ("list", "testimonials"),
    "colors": ("kv", "colors"),
    "brand colors": ("kv", "colors"),
    "brand look": ("kv", "colors"),
}

_KV_FIELDS: dict[str, dict[str, str]] = {
    "basics": {
        "name": "name",
        "business name": "name",
        "brand name": "name",
        "what we sell": "offer",
        "what you sell": "offer",
        "offer": "offer",
        "tagline": "tagline",
        "website": "website",
        "url": "website",
        "call to action": "call_to_action",
        "cta": "call_to_action",
        "main cta": "call_to_action",
    },
    "messaging": {
        "umbrella statement": "umbrella_statement",
        "umbrella": "umbrella_statement",
        "cold audience": "cold_message",
        "cold": "cold_message",
        "warm audience": "warm_message",
        "warm": "warm_message",
        "hot audience": "hot_message",
        "hot": "hot_message",
    },
    "voice": {
        "tone": "voice",
        "voice": "voice",
        "words we use": "use_words",
        "words you use": "use_words",
        "phrases we use": "use_words",
        "words we never use": "avoid_words",
        "words you never use": "avoid_words",
        "never use": "avoid_words",
        "avoid": "avoid_words",
    },
    "colors": {"primary": "primary", "secondary": "secondary", "accent": "accent"},
}

_LIST_FIELDS = {"voice", "use_words", "avoid_words"}
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s*(.*)$")
_KV_RE = re.compile(r"^([^:]{1,60}):\s*(.*)$")


def _norm(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _parse_markdown(text: str) -> dict[str, Any]:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    data: dict[str, Any] = {}
    contact: dict[str, str] = {}
    colors: dict[str, str] = {}
    notes: dict[str, str] = {}
    paragraphs: dict[str, list[str]] = {}

    heading = ""
    kind, target = "skip", ""

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            heading = line[level:].strip()
            if level == 1:
                kind, target = "skip", ""
            else:
                kind, target = _SECTIONS.get(_norm(heading), ("note", heading))
            continue
        if not line or (line.startswith("(") and line.endswith(")")) or kind == "skip":
            continue

        bullet = _BULLET_RE.match(line)
        content = bullet.group(1).strip() if bullet else line
        if not content:
            continue

        if kind == "list":
            data.setdefault(target, []).append(content)
        elif kind == "text":
            paragraphs.setdefault(target, []).append(content)
        elif kind == "kv":
            match = _KV_RE.match(content)
            if not match:
                notes[heading] = f"{notes[heading]} {content}".strip() if heading in notes else content
                continue
            label, value = _norm(match.group(1)), match.group(2).strip()
            if not value:
                continue
            field_name = _KV_FIELDS[target].get(label)
            if target == "colors" and field_name and _HEX_RE.match(value):
                colors[field_name] = value
            elif target == "colors":
                notes[f"Colors: {match.group(1).strip()}"] = value
            elif field_name in _LIST_FIELDS:
                data[field_name] = [part.strip() for part in value.split(",") if part.strip()]
            elif field_name:
                data[field_name] = value
            elif target == "basics":
                contact[label] = value
            else:
                notes[f"{heading}: {match.group(1).strip()}"] = value
        else:  # unknown section
            notes[heading] = f"{notes[heading]} {content}".strip() if heading in notes else content

    for field_name, parts in paragraphs.items():
        data[field_name] = " ".join(parts)
    if contact:
        data["contact"] = contact
    if colors:
        data["colors"] = colors
    if notes:
        data["notes"] = notes
    if not data.get("name"):
        raise ValueError("The brand file has no name. Fill in '- Name:' under '## Basics'.")
    return data
