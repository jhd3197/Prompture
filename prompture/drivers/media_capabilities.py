"""Cross-provider media model schema/capability registry (the media KB).

Generation UIs and workflow engines need to know — *without instantiating a
driver or holding an API key* — what inputs a given media model accepts: its
parameter types/enums/defaults/ranges, which field a conditioning image goes
in, how many images it takes, what aspect ratios / resolutions / durations /
effects / modes it supports, and which operation it performs.

:data:`runway_capabilities.RUNWAY_MODEL_INFO` did this for Runway only. This
module generalizes it to a provider-neutral registry keyed by
``"provider/model"`` so callers get one lookup across every media provider:

- *What can model X do / take?*       → :func:`get_model_schema`
- *Which models do image_to_video?*   → :func:`get_models_by_op`
- *Render dynamic controls for X*      → :func:`get_video_model_controls` etc.

Driver classes still own the request shape; this is metadata only. Entries are
extensible at runtime via :func:`register_media_model`; Runway models are
bridged in automatically from the existing Runway KB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .runway_capabilities import RUNWAY_MODEL_INFO

__all__ = [
    "MEDIA_MODEL_INFO",
    "MediaModelInfo",
    "ParamSpec",
    "get_aspect_ratios",
    "get_audio_model_controls",
    "get_durations",
    "get_effects",
    "get_max_images",
    "get_model_schema",
    "get_models_by_modality",
    "get_models_by_op",
    "get_modes",
    "get_resolutions",
    "get_video_model_controls",
    "register_media_model",
]

# Canonical operation names (provider-neutral).
OPS = {
    "text_to_image",
    "image_to_image",
    "text_to_video",
    "image_to_video",
    "video_to_video",
    "lipsync",
    "text_to_speech",
    "speech_to_text",
    "music",
    "sound_effect",
    "voice_clone",
}


@dataclass(frozen=True)
class ParamSpec:
    """JSON-Schema-flavored spec for one model input parameter."""

    name: str
    type: str = "string"  # string | int | number | boolean | array | object
    description: str = ""
    enum: list[Any] | None = None
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    required: bool = False

    def to_json_schema(self) -> dict[str, Any]:
        """Render as a JSON-Schema property fragment."""
        json_type = {
            "int": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }.get(self.type, "string")
        out: dict[str, Any] = {"type": json_type}
        if self.description:
            out["description"] = self.description
        if self.enum is not None:
            out["enum"] = list(self.enum)
        if self.default is not None:
            out["default"] = self.default
        if self.min_value is not None:
            out["minimum"] = self.min_value
        if self.max_value is not None:
            out["maximum"] = self.max_value
        return out


@dataclass(frozen=True)
class MediaModelInfo:
    """Capability metadata + input schema for a single media model."""

    model: str  # provider-local id / endpoint slug
    provider: str  # "muapi" | "runway" | "fal" | ...
    modality: str  # "image" | "video" | "audio" | "lipsync"
    op: str  # one of OPS
    endpoint: str | None = None
    family: str | None = None  # "flux" | "kling" | "seedream" | ...
    inputs: dict[str, ParamSpec] = field(default_factory=dict)
    # Routing flags (how a driver shapes the request).
    image_field: str | None = None
    last_image_field: str | None = None
    video_field: str | None = None
    audio_field: str | None = None
    max_images: int = 1
    requires_request_id: bool = False
    category: str | None = None  # lipsync: "image" | "video"
    cost: str | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    def _enum(self, param: str) -> list[Any]:
        spec = self.inputs.get(param)
        return list(spec.enum) if spec and spec.enum else []

    def aspect_ratios(self) -> list[str]:
        return self._enum("aspect_ratio")

    def resolutions(self) -> list[str]:
        return self._enum("resolution")

    def durations(self) -> list[Any]:
        return self._enum("duration")

    def effects(self) -> list[Any]:
        return self._enum("effect") or self._enum("effects")

    def modes(self) -> list[Any]:
        return self._enum("mode")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["inputs"] = {k: v.to_json_schema() for k, v in self.inputs.items()}
        return d


MEDIA_MODEL_INFO: dict[str, MediaModelInfo] = {}


def register_media_model(info: MediaModelInfo, *, overwrite: bool = True) -> None:
    """Register (or replace) a media model entry, keyed by ``provider/model``."""
    if not overwrite and info.key in MEDIA_MODEL_INFO:
        raise ValueError(f"Media model already registered: {info.key}")
    MEDIA_MODEL_INFO[info.key] = info


def _ar(*ratios: str) -> ParamSpec:
    return ParamSpec("aspect_ratio", "string", "Output aspect ratio.", enum=list(ratios), default=ratios[0])


def _seed_builtins() -> None:
    common_ar = ("1:1", "16:9", "9:16", "4:3", "3:4")
    entries = [
        MediaModelInfo(
            "nano-banana",
            "muapi",
            "image",
            "text_to_image",
            endpoint="nano-banana",
            family="gemini-image",
            inputs={"aspect_ratio": _ar(*common_ar, "3:2", "2:3", "21:9")},
        ),
        MediaModelInfo(
            "flux-dev",
            "muapi",
            "image",
            "text_to_image",
            endpoint="flux-dev-image",
            family="flux",
            inputs={
                "width": ParamSpec(
                    "width", "int", "Width (divisible by 64).", default=1024, min_value=128, max_value=2048, step=64
                ),
                "height": ParamSpec(
                    "height", "int", "Height (divisible by 64).", default=1024, min_value=128, max_value=2048, step=64
                ),
                "num_images": ParamSpec(
                    "num_images", "int", "Images per request.", default=1, min_value=1, max_value=4
                ),
            },
        ),
        MediaModelInfo(
            "seedream-edit",
            "muapi",
            "image",
            "image_to_image",
            endpoint="seedream-edit",
            family="seedream",
            image_field="images_list",
            max_images=14,
            inputs={
                "aspect_ratio": _ar(*common_ar),
                "strength": ParamSpec(
                    "strength", "number", "Edit strength.", default=0.6, min_value=0.0, max_value=1.0
                ),
            },
        ),
        MediaModelInfo(
            "kling-video-v2-1",
            "muapi",
            "video",
            "text_to_video",
            endpoint="kling-video-v2-1",
            family="kling",
            inputs={
                "aspect_ratio": _ar(*common_ar),
                "duration": ParamSpec("duration", "int", "Clip seconds.", enum=[5, 10], default=5),
            },
        ),
        MediaModelInfo(
            "kling-video-v2-1-image-to-video",
            "muapi",
            "video",
            "image_to_video",
            endpoint="kling-video-v2-1-image-to-video",
            family="kling",
            image_field="image_url",
            last_image_field="last_image",
            inputs={"duration": ParamSpec("duration", "int", "Clip seconds.", enum=[5, 10], default=5)},
        ),
        MediaModelInfo(
            "infinite-talk",
            "muapi",
            "lipsync",
            "lipsync",
            endpoint="infinite-talk",
            family="infinite-talk",
            image_field="image_url",
            audio_field="audio_url",
            category="image",
            inputs={
                "resolution": ParamSpec(
                    "resolution", "string", "Output resolution.", enum=["480p", "720p"], default="480p"
                )
            },
        ),
        MediaModelInfo(
            "suno-create-music",
            "muapi",
            "audio",
            "music",
            endpoint="suno-create-music",
            family="suno",
            inputs={
                "instrumental": ParamSpec("instrumental", "boolean", "Instrumental only.", default=False),
                "model": ParamSpec(
                    "model", "string", "Suno model version.", enum=["V3_5", "V4", "V4_5", "V5"], default="V4_5"
                ),
            },
        ),
        MediaModelInfo(
            "flux/dev",
            "fal",
            "image",
            "text_to_image",
            endpoint="fal-ai/flux/dev",
            family="flux",
            inputs={
                "num_images": ParamSpec("num_images", "int", "Images per request.", default=1, min_value=1, max_value=4)
            },
        ),
    ]
    for e in entries:
        MEDIA_MODEL_INFO.setdefault(e.key, e)


def _split(model_str: str) -> tuple[str | None, str]:
    if "/" in model_str:
        provider, _, model = model_str.partition("/")
        return provider.lower(), model
    return None, model_str


def _runway_bridge(model: str) -> MediaModelInfo | None:
    info = RUNWAY_MODEL_INFO.get(model)
    if info is None:
        return None
    op = info["operations"][0] if info["operations"] else info["modality"]
    return MediaModelInfo(
        model=model,
        provider="runway",
        modality=info["modality"],
        op=op,
        endpoint=info["endpoints"][0] if info["endpoints"] else None,
        cost=info.get("cost"),
    )


def get_model_schema(model_str: str) -> MediaModelInfo | None:
    """Return the :class:`MediaModelInfo` for ``provider/model`` (or a bare id).

    Falls back to bridging Runway's own KB when the provider is ``runway``.
    """
    provider, model = _split(model_str)
    if provider:
        hit = MEDIA_MODEL_INFO.get(f"{provider}/{model}")
        if hit:
            return hit
        if provider in {"runway", "runwayml"}:
            return _runway_bridge(model)
        return None
    # Bare id: search by model field, then Runway.
    for info in MEDIA_MODEL_INFO.values():
        if info.model == model:
            return info
    return _runway_bridge(model)


def get_models_by_modality(modality: str) -> list[str]:
    """Return ``provider/model`` keys for every registered model in *modality*."""
    return sorted(k for k, v in MEDIA_MODEL_INFO.items() if v.modality == modality)


def get_models_by_op(op: str) -> list[str]:
    """Return ``provider/model`` keys for every registered model supporting *op*."""
    if op not in OPS:
        raise ValueError(f"Unknown op {op!r}. Known: {sorted(OPS)}")
    return sorted(k for k, v in MEDIA_MODEL_INFO.items() if v.op == op)


# ── Per-param convenience getters ───────────────────────────────────────────


def _info_or_empty(model_str: str) -> MediaModelInfo | None:
    return get_model_schema(model_str)


def get_aspect_ratios(model_str: str) -> list[str]:
    info = _info_or_empty(model_str)
    return info.aspect_ratios() if info else []


def get_resolutions(model_str: str) -> list[str]:
    info = _info_or_empty(model_str)
    return info.resolutions() if info else []


def get_durations(model_str: str) -> list[Any]:
    info = _info_or_empty(model_str)
    return info.durations() if info else []


def get_effects(model_str: str) -> list[Any]:
    info = _info_or_empty(model_str)
    return info.effects() if info else []


def get_modes(model_str: str) -> list[Any]:
    info = _info_or_empty(model_str)
    return info.modes() if info else []


def get_max_images(model_str: str) -> int:
    info = _info_or_empty(model_str)
    return info.max_images if info else 1


# ── UI control siblings (mirror infra.discovery.get_image_model_controls) ───


def get_video_model_controls(model_str: str) -> dict[str, Any]:
    """UI-ready controls for a video model, derived from the schema KB.

    Returns ``{"aspect_ratios": [...], "resolutions": [...], "durations": [...],
    "modes": [...], "supports_image_input": bool, "op": str}``.
    """
    info = _info_or_empty(model_str)
    if info is None:
        return {
            "aspect_ratios": [],
            "resolutions": [],
            "durations": [],
            "modes": [],
            "supports_image_input": False,
            "op": None,
        }
    return {
        "aspect_ratios": info.aspect_ratios(),
        "resolutions": info.resolutions(),
        "durations": info.durations(),
        "modes": info.modes(),
        "supports_image_input": info.op in {"image_to_video", "video_to_video"} or info.image_field is not None,
        "op": info.op,
    }


def get_audio_model_controls(model_str: str) -> dict[str, Any]:
    """UI-ready controls for an audio (TTS / music / SFX) model from the KB."""
    info = _info_or_empty(model_str)
    if info is None:
        return {"op": None, "params": {}}
    return {"op": info.op, "params": {k: v.to_json_schema() for k, v in info.inputs.items()}}


_seed_builtins()
