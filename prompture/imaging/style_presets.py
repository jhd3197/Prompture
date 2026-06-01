"""Reusable, provider-agnostic image style presets.

A :class:`StylePreset` composes the final generation prompt (prefix + user
prompt + suffix) and a normalized options overlay *before* any image driver is
called, so it behaves identically across every provider. Image drivers read a
fixed whitelist of option keys and silently ignore the rest, so a preset's
``params`` overlay that a given driver doesn't understand is harmlessly dropped
— the descriptive prompt prefix/suffix always carries the visual intent.

Preset ids and labels are intentionally descriptive and vendor-neutral; they do
not reference any product, studio, or company.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


@dataclass(frozen=True)
class StylePreset:
    """A named, reusable bundle of prompt text and option overrides.

    Attributes:
        id: Stable, vendor-neutral identifier (e.g. ``"cinematic"``).
        label: Human-readable name for UI.
        description: One-line summary of the look (also used by the prompt
            enhancer when the preset is passed as a style hint).
        prompt_prefix: Text prepended to the user prompt.
        prompt_suffix: Text appended to the user prompt.
        negative: Suggested negative-prompt guidance (route it with
            :func:`prompture.imaging.compose_negative_prompt`).
        params: A normalized option overlay merged into the generation options
            (e.g. ``{"quality": "hd"}``). Keys a driver doesn't read are ignored.
    """

    id: str
    label: str
    description: str = ""
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    negative: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def apply(
        self, prompt: str, options: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Return ``(final_prompt, options)`` with this preset applied.

        The prefix/suffix wrap the user prompt (whitespace-collapsed). The
        preset's ``params`` overlay is shallow-merged *over* the supplied
        options, so the preset's explicit style choices win on key collisions.
        """
        parts = [self.prompt_prefix, prompt or "", self.prompt_suffix]
        final_prompt = _collapse_ws(" ".join(p for p in parts if p and p.strip()))
        merged: dict[str, Any] = {**(options or {}), **dict(self.params)}
        return final_prompt, merged


_REGISTRY: dict[str, StylePreset] = {}


def register_style_preset(preset: StylePreset, *, overwrite: bool = False) -> None:
    """Register a preset so it can be looked up by id.

    Raises ``ValueError`` if the id is already taken and ``overwrite`` is False.
    """
    if not overwrite and preset.id in _REGISTRY:
        raise ValueError(
            f"Style preset '{preset.id}' is already registered (pass overwrite=True to replace)."
        )
    _REGISTRY[preset.id] = preset


def get_style_preset(preset_id: str) -> StylePreset | None:
    """Return the registered preset for ``preset_id``, or ``None``."""
    return _REGISTRY.get(preset_id)


def list_style_presets() -> list[StylePreset]:
    """All registered presets, in registration order."""
    return list(_REGISTRY.values())


# Curated default library — descriptive, vendor-neutral looks that work on any
# text-to-image provider (positive prompt) with optional native params.
DEFAULT_STYLE_PRESETS: tuple[StylePreset, ...] = (
    StylePreset(
        id="photoreal",
        label="Photorealistic",
        description="Clean, true-to-life photography.",
        prompt_suffix="photorealistic, natural lighting, high detail, sharp focus, 35mm photograph",
        negative="illustration, cartoon, painting, low detail, watermark",
    ),
    StylePreset(
        id="studio_product",
        label="Studio Product Shot",
        description="A product on a clean seamless background with soft studio light.",
        prompt_prefix="Studio product photograph of",
        prompt_suffix="on a seamless white background, soft diffused studio lighting, high detail, commercial photography",
        negative="clutter, busy background, harsh shadows, text",
    ),
    StylePreset(
        id="cinematic",
        label="Cinematic",
        description="Moody, film-like frame with shallow depth of field.",
        prompt_suffix="cinematic lighting, shallow depth of field, dramatic mood, color graded, anamorphic, film still",
        params={"quality": "hd"},
    ),
    StylePreset(
        id="flat_vector",
        label="Flat Vector",
        description="Simple flat illustration with bold solid colors.",
        prompt_suffix="flat vector illustration, clean geometric shapes, bold solid colors, minimal, no gradients",
        negative="photo, realistic, 3d, texture, noise",
    ),
    StylePreset(
        id="line_art",
        label="Line Art",
        description="Minimal single-weight line drawing on white.",
        prompt_suffix="minimal line art, single-weight black lines, white background, no shading",
        negative="color, shading, photo, gradient",
    ),
    StylePreset(
        id="watercolor",
        label="Watercolor",
        description="Soft hand-painted watercolor with visible paper texture.",
        prompt_suffix="soft watercolor painting, gentle washes, visible paper texture, hand-painted, delicate",
    ),
    StylePreset(
        id="isometric_3d",
        label="Isometric 3D",
        description="Clean isometric 3D render with soft lighting.",
        prompt_suffix="isometric 3d render, soft global illumination, clean materials, subtle ambient occlusion",
        negative="flat, 2d, photo",
    ),
    StylePreset(
        id="soft_portrait",
        label="Soft Portrait",
        description="Flattering portrait in soft natural light.",
        prompt_suffix="portrait, soft window light, shallow depth of field, natural skin tones, gentle bokeh",
        negative="harsh light, oversaturated, distorted features",
    ),
    StylePreset(
        id="minimal_poster",
        label="Minimal Poster",
        description="Bold minimal graphic poster with generous negative space.",
        prompt_suffix="minimal graphic poster, bold composition, generous negative space, limited color palette, modern",
        negative="clutter, busy, photorealistic",
    ),
    StylePreset(
        id="storybook",
        label="Storybook Illustration",
        description="Warm hand-drawn storybook art with gentle colors.",
        prompt_suffix="warm storybook illustration, soft hand-drawn textures, gentle colors, whimsical",
    ),
)

for _preset in DEFAULT_STYLE_PRESETS:
    register_style_preset(_preset, overwrite=True)
