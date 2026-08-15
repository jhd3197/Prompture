"""Expose Prompture's media generation as agent-callable tools.

Wraps the image/video drivers and the job-resume API as
:class:`~prompture.agents.tools_schema.ToolDefinition` objects so an agent (the
canvas design loop, a workflow Architect, a chat agent) can *call generation*
the same way it calls any other tool. Functions are URL/asset oriented — they
return hosted URLs rather than inline base64 — matching how a studio addresses
assets across turns.

Usage::

    from prompture.agents import ToolRegistry
    from prompture.media.agent_tools import register_media_tools

    registry = register_media_tools(ToolRegistry())
    # registry now exposes: generate_image, edit_image, generate_video,
    #                       resume_media_job, list_media_models
"""

from __future__ import annotations

from typing import Any

from ..agents.tools_schema import ToolDefinition, ToolRegistry, tool_from_function

__all__ = [
    "edit_image",
    "generate_image",
    "generate_music",
    "generate_video",
    "list_media_models",
    "media_tool_definitions",
    "register_media_tools",
    "resume_media_job",
]


def _payload(response: dict[str, Any], key: str) -> dict[str, Any]:
    items = response.get(key, []) or []
    urls = [getattr(c, "url", None) for c in items]
    meta = response.get("meta", {})
    return {
        "urls": [u for u in urls if u],
        "model": meta.get("model_name"),
        "cost": meta.get("cost"),
        "request_id": meta.get("request_id"),
        "status": meta.get("status", "completed"),
        "job_handle": meta.get("job_handle"),
    }


def generate_image(model: str, prompt: str, aspect_ratio: str = "1:1", n: int = 1) -> dict[str, Any]:
    """Generate image(s) from a text prompt and return their hosted URLs.

    Args:
        model: A ``provider/model`` string, e.g. ``"muapi/nano-banana-pro"`` or ``"openai/dall-e-3"``.
        prompt: Text description of the desired image.
        aspect_ratio: Output aspect ratio such as ``"1:1"``, ``"16:9"``, or ``"9:16"``.
        n: Number of images to generate.

    Returns:
        A dict with ``urls`` (list of image URLs), ``model``, and ``cost``.
    """
    from ..drivers.img_gen_registry import get_img_gen_driver_for_model

    drv = get_img_gen_driver_for_model(model)
    res = drv.generate_image(prompt, {"aspect_ratio": aspect_ratio, "num_images": n})
    return _payload(res, "images")


def edit_image(model: str, image_url: str, prompt: str) -> dict[str, Any]:
    """Edit an existing image (image-to-image) given its URL and an instruction.

    Args:
        model: A ``provider/model`` string for an edit-capable model, e.g. ``"muapi/seedream-edit"``.
        image_url: A publicly fetchable URL of the source image.
        prompt: Instruction describing the change to make.

    Returns:
        A dict with ``urls`` (list of result image URLs), ``model``, and ``cost``.
    """
    from ..drivers.img_gen_registry import get_img_gen_driver_for_model

    drv = get_img_gen_driver_for_model(model)
    res = drv.edit_image(b"", prompt, {"image_url": image_url})
    return _payload(res, "images")


def generate_video(
    model: str,
    prompt: str,
    image_url: str | None = None,
    duration: int | None = None,
    aspect_ratio: str | None = None,
) -> dict[str, Any]:
    """Generate a video from a prompt (text-to-video or image-to-video).

    Args:
        model: A ``provider/model`` string, e.g. ``"muapi/kling-video-v2-1-image-to-video"``.
        prompt: Text description / motion prompt.
        image_url: Optional start-frame image URL (makes it image-to-video).
        duration: Optional clip length in seconds.
        aspect_ratio: Optional aspect ratio such as ``"16:9"``.

    Returns:
        A dict with ``urls`` (list of video URLs), ``model``, and ``cost``.
    """
    from ..drivers.video_gen_registry import get_video_gen_driver_for_model

    drv = get_video_gen_driver_for_model(model)
    options: dict[str, Any] = {}
    if image_url:
        options["image_url"] = image_url
    if duration is not None:
        options["duration"] = duration
    if aspect_ratio:
        options["aspect_ratio"] = aspect_ratio
    res = drv.generate_video(prompt, options)
    return _payload(res, "videos")


def generate_music(model: str, prompt: str, instrumental: bool = False, duration: int | None = None) -> dict[str, Any]:
    """Generate a music track from a text prompt.

    Args:
        model: A ``provider/model`` string, e.g. ``"muapi/suno-create-music"``.
        prompt: Description of the desired music (style, mood, lyrics theme).
        instrumental: If true, generate without vocals.
        duration: Optional target length in seconds.

    Returns:
        A dict with ``urls`` (list of audio URLs), ``model``, and ``cost``.
    """
    from ..drivers.music_registry import get_music_driver_for_model

    drv = get_music_driver_for_model(model)
    options: dict[str, Any] = {"instrumental": instrumental}
    if duration is not None:
        options["duration"] = duration
    res = drv.generate_music(prompt, options)
    return _payload(res, "audio")


def resume_media_job(handle_json: str) -> dict[str, Any]:
    """Reconnect to a previously-submitted media job and return its current state.

    Args:
        handle_json: The JSON string of a ``JobHandle`` (from a prior ``poll=False`` submission).

    Returns:
        A dict with ``status``, ``done`` (bool), and ``urls`` (any produced assets).
    """
    from ..jobs import resume

    result = resume(handle_json)
    return {
        "status": result.status,
        "done": result.done,
        "urls": [a.url for a in result.assets if a.url],
    }


def list_media_models(modality: str) -> dict[str, Any]:
    """List known media models for a modality (``image`` / ``video`` / ``audio`` / ``lipsync``).

    Args:
        modality: The modality to list.

    Returns:
        A dict with ``models`` (list of ``provider/model`` ids).
    """
    from ..drivers.media_capabilities import get_models_by_modality

    return {"models": get_models_by_modality(modality)}


_TOOL_FNS = (generate_image, edit_image, generate_video, generate_music, resume_media_job, list_media_models)


def media_tool_definitions() -> list[ToolDefinition]:
    """Return one :class:`ToolDefinition` per media tool."""
    return [tool_from_function(fn) for fn in _TOOL_FNS]


def register_media_tools(registry: ToolRegistry) -> ToolRegistry:
    """Register all media tools into *registry* and return it (for chaining)."""
    for td in media_tool_definitions():
        registry.add(td)
    return registry
