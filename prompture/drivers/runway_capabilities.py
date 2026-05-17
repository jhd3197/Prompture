"""Per-model capability registry for Runway models.

Runway's endpoints are operation-shaped: each path (``/v1/text_to_image``,
``/v1/image_to_video``, ``/v1/voice_dubbing`` …) is what a model can *do*.
The same model often supports multiple operations (``gen4.5`` does both
``text_to_video`` and ``image_to_video``) or is locked to one
(``gen4_aleph`` is video_to_video only).

This module exposes that mapping as data so callers can answer:

- *What operations does model X support?* → :func:`get_runway_model_info`
- *Which models can do text_to_video?*    → :func:`get_runway_models_by_op`
- *Give me everything I know about Runway* → :data:`RUNWAY_MODEL_INFO`

Driver classes still own the request shape; this is metadata only.
"""

from __future__ import annotations

from typing import TypedDict


class RunwayModelInfo(TypedDict):
    """Capability metadata for a single Runway model."""

    modality: str  # "image" | "video" | "audio"
    operations: list[str]  # e.g. ["text_to_video", "image_to_video"]
    endpoints: list[str]  # POST paths, one per operation
    cost: str  # human-readable cost note


# Operation names mirror Runway's endpoint slugs.
RUNWAY_MODEL_INFO: dict[str, RunwayModelInfo] = {
    # ── Image (POST /v1/text_to_image) ───────────────────────────────
    "gen4_image": {
        "modality": "image",
        "operations": ["text_to_image"],
        "endpoints": ["/v1/text_to_image"],
        "cost": "$0.05 (720p) / $0.08 (1080p) per image",
    },
    "gen4_image_turbo": {
        "modality": "image",
        "operations": ["text_to_image"],
        "endpoints": ["/v1/text_to_image"],
        "cost": "$0.02 per image (reference image required)",
    },
    "gpt_image_2": {
        "modality": "image",
        "operations": ["text_to_image"],
        "endpoints": ["/v1/text_to_image"],
        "cost": "$0.04 (low) / $0.06 (medium) / $0.08 (high) per image",
    },
    "gemini_image3_pro": {
        "modality": "image",
        "operations": ["text_to_image"],
        "endpoints": ["/v1/text_to_image"],
        "cost": "$0.08 per image",
    },
    "gemini_2.5_flash": {
        "modality": "image",
        "operations": ["text_to_image"],
        "endpoints": ["/v1/text_to_image"],
        "cost": "$0.05 per image",
    },
    # ── Video ───────────────────────────────────────────────────────
    "gen4.5": {
        "modality": "video",
        "operations": ["text_to_video", "image_to_video"],
        "endpoints": ["/v1/text_to_video", "/v1/image_to_video"],
        "cost": "$0.12 per second",
    },
    "gen4_turbo": {
        "modality": "video",
        "operations": ["image_to_video"],
        "endpoints": ["/v1/image_to_video"],
        "cost": "$0.05 per second",
    },
    "gen3a_turbo": {
        "modality": "video",
        "operations": ["image_to_video"],
        "endpoints": ["/v1/image_to_video"],
        "cost": "$0.05 per second",
    },
    "gen4_aleph": {
        "modality": "video",
        "operations": ["video_to_video"],
        "endpoints": ["/v1/video_to_video"],
        "cost": "$0.15 per second of input",
    },
    "veo3": {
        "modality": "video",
        "operations": ["text_to_video", "image_to_video"],
        "endpoints": ["/v1/text_to_video", "/v1/image_to_video"],
        "cost": "$0.40 per second",
    },
    "veo3.1": {
        "modality": "video",
        "operations": ["text_to_video", "image_to_video"],
        "endpoints": ["/v1/text_to_video", "/v1/image_to_video"],
        "cost": "$0.30 per second",
    },
    "veo3.1_fast": {
        "modality": "video",
        "operations": ["text_to_video", "image_to_video"],
        "endpoints": ["/v1/text_to_video", "/v1/image_to_video"],
        "cost": "$0.12 per second",
    },
    # ── Audio ───────────────────────────────────────────────────────
    "eleven_multilingual_v2": {
        "modality": "audio",
        "operations": ["text_to_speech"],
        "endpoints": ["/v1/text_to_speech"],
        "cost": "~$0.000030 per character",
    },
    "eleven_text_to_sound_v2": {
        "modality": "audio",
        "operations": ["sound_effect"],
        "endpoints": ["/v1/sound_effect"],
        "cost": "~$0.04 per second of generated sound",
    },
    "eleven_voice_dubbing": {
        "modality": "audio",
        "operations": ["voice_dubbing"],
        "endpoints": ["/v1/voice_dubbing"],
        "cost": "per-second of input audio (Runway billing)",
    },
    "eleven_voice_isolation": {
        "modality": "audio",
        "operations": ["voice_isolation"],
        "endpoints": ["/v1/voice_isolation"],
        "cost": "per-second of input audio (Runway billing)",
    },
    "eleven_multilingual_sts_v2": {
        "modality": "audio",
        "operations": ["speech_to_speech"],
        "endpoints": ["/v1/speech_to_speech"],
        "cost": "per-second of input audio (Runway billing)",
    },
}


ALL_OPERATIONS: list[str] = sorted({op for info in RUNWAY_MODEL_INFO.values() for op in info["operations"]})
ALL_MODALITIES: list[str] = sorted({info["modality"] for info in RUNWAY_MODEL_INFO.values()})


def get_runway_model_info(model_id: str) -> RunwayModelInfo | None:
    """Return capability info for a Runway model, or ``None`` if unknown.

    Accepts a bare model id (``"gen4.5"``) or a ``provider/model`` string
    (``"runway/gen4.5"``, ``"runwayml/gen4.5"``).
    """
    if "/" in model_id:
        model_id = model_id.split("/", 1)[1]
    return RUNWAY_MODEL_INFO.get(model_id)


def get_runway_models_by_op(operation: str) -> list[str]:
    """Return Runway model ids that support the given operation.

    Example::

        >>> get_runway_models_by_op("text_to_video")
        ['gen4.5', 'veo3', 'veo3.1', 'veo3.1_fast']
    """
    if operation not in ALL_OPERATIONS:
        raise ValueError(f"Unknown operation {operation!r}. Known: {ALL_OPERATIONS}")
    return sorted(model_id for model_id, info in RUNWAY_MODEL_INFO.items() if operation in info["operations"])


def get_runway_models_by_modality(modality: str) -> list[str]:
    """Return Runway model ids in the given modality (``image``/``video``/``audio``)."""
    if modality not in ALL_MODALITIES:
        raise ValueError(f"Unknown modality {modality!r}. Known: {ALL_MODALITIES}")
    return sorted(model_id for model_id, info in RUNWAY_MODEL_INFO.items() if info["modality"] == modality)
