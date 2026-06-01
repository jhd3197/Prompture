"""Art-director image-set planner — decompose one brief into a cohesive set of
standalone image prompts.

Model-agnostic: rides any text model through ``ask_for_json``. The planner
emits abstract aspect tokens (``square`` / ``landscape`` / ``portrait`` /
``auto``) only — never pixel sizes or provider aspect strings — because image
providers disagree on the size vocabulary. Resolving a token to a concrete size
is left to the caller (or a provider-aware resolver), so the plan stays
portable across every backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

_ASPECTS = ("square", "landscape", "portrait", "auto")

_PLAN_SYSTEM = (
    "You are an art director planning a coherent set of images from a single "
    "brief. Decompose the brief into individual images, each with a vivid, "
    "self-contained generation prompt (include style, lighting, and mood so "
    "each stands alone) and an aspect ratio. Keep the set cohesive — shared "
    "palette, style, and subject treatment. Prefer the smallest set that "
    "satisfies the brief."
)

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short label, e.g. 'hero' or 'feature-icon-1'",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Full standalone image generation prompt",
                    },
                    "aspect": {"type": "string", "enum": list(_ASPECTS)},
                },
                "required": ["prompt"],
            },
        }
    },
    "required": ["images"],
}


@dataclass(frozen=True)
class ImageSpec:
    """One planned image: a standalone prompt plus an abstract aspect token."""

    prompt: str
    name: str | None = None
    aspect: Literal["square", "landscape", "portrait", "auto"] = "square"


@dataclass(frozen=True)
class ImageSetPlan:
    """A planned set of images derived from a brief."""

    images: list[ImageSpec]
    brief: str
    model: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        return iter(self.images)


def _coerce_specs(raw_images: Any, max_images: int) -> list[ImageSpec]:
    specs: list[ImageSpec] = []
    for item in raw_images or []:
        if not isinstance(item, dict):
            continue
        prompt = (item.get("prompt") or "").strip()
        if not prompt:
            continue
        aspect = str(item.get("aspect") or "square").lower()
        if aspect not in _ASPECTS:
            aspect = "square"
        specs.append(ImageSpec(prompt=prompt, name=(item.get("name") or None), aspect=aspect))
        if len(specs) >= max_images:
            break
    return specs


def _build_prompt(brief: str, max_images: int, style_guidance: str | None) -> str:
    extra = f"\nStyle guidance: {style_guidance}" if style_guidance else ""
    return f"Brief: {brief}{extra}\n\nPlan at most {max_images} images for this brief."


def plan_image_set(
    brief: str,
    *,
    model: str | None = None,
    driver: Any | None = None,
    max_images: int = 6,
    system_prompt: str | None = None,
    style_guidance: str | None = None,
    env: Any | None = None,
    options: dict[str, Any] | None = None,
) -> ImageSetPlan:
    """Plan a cohesive image set from ``brief`` using a text model.

    Provide either ``model`` (e.g. ``"openai/gpt-4o-mini"``) or a ready
    ``driver``. Prompt-less items are dropped, the aspect defaults to ``square``,
    and the result is clamped to ``max_images``.
    """
    from ..agents.conversation import Conversation

    conv = Conversation(model_name=model, driver=driver, system_prompt=system_prompt or _PLAN_SYSTEM, env=env)
    result = conv.ask_for_json(_build_prompt(brief, max_images, style_guidance), _PLAN_SCHEMA, options=options)
    raw = result.get("json_object") or {}
    return ImageSetPlan(images=_coerce_specs(raw.get("images"), max_images), brief=brief, model=model, raw=raw)


async def aplan_image_set(
    brief: str,
    *,
    model: str | None = None,
    driver: Any | None = None,
    max_images: int = 6,
    system_prompt: str | None = None,
    style_guidance: str | None = None,
    env: Any | None = None,
    options: dict[str, Any] | None = None,
) -> ImageSetPlan:
    """Async twin of :func:`plan_image_set`."""
    from ..agents.async_conversation import AsyncConversation

    conv = AsyncConversation(model_name=model, driver=driver, system_prompt=system_prompt or _PLAN_SYSTEM, env=env)
    result = await conv.ask_for_json(_build_prompt(brief, max_images, style_guidance), _PLAN_SCHEMA, options=options)
    raw = result.get("json_object") or {}
    return ImageSetPlan(images=_coerce_specs(raw.get("images"), max_images), brief=brief, model=model, raw=raw)
