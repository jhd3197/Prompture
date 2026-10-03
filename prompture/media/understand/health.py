"""Health rows for media understanding, registered with the capability registry.

* ``transcription`` — which STT provider ``model="auto"`` would use, plus
  ffmpeg / ffprobe health (needed for long audio and video). Offline unless
  ``live=True``, which transcribes half a second of silence with the active
  provider (only when a key is configured).
* ``media_download`` — ``yt-dlp`` health for video/podcast pages.
"""

from __future__ import annotations

from typing import Any

from ...capabilities.health import HealthStatus, register_capability
from . import ffmpeg as _ff
from .transcribe import STT_PROVIDERS, active_stt_provider, provider_configured, stt_setup_hint


def _probe_detail(name: str) -> dict[str, Any]:
    probe = _ff.probe_tool(name)
    return {
        "status": probe.status,
        "path": probe.path,
        "version": probe.version if probe.ok else None,
        "hint": probe.hint,
    }


def transcription_status(live: bool = False) -> HealthStatus:
    """Health row for the transcription pipeline."""
    providers = [
        {"provider": s.name, "configured": provider_configured(s.name), "env_var": s.env_var, "model": s.default_model}
        for s in STT_PROVIDERS
    ]
    binaries = {name: _probe_detail(name) for name in ("ffmpeg", "ffprobe")}
    details: dict[str, Any] = {"providers": providers, "binaries": binaries}
    backend = active_stt_provider()
    if backend is None:
        return HealthStatus(
            "transcription",
            "unconfigured",
            category="media",
            message="no speech-to-text provider configured",
            fix_hint=stt_setup_hint(),
            details=details,
        )

    ffmpeg_ok = all(b["status"] == "ok" for b in binaries.values())
    if live:
        try:
            backend.live_check()
        except Exception as exc:
            from ...resilience.errors import classify_error

            info = classify_error(exc)
            return HealthStatus(
                "transcription",
                "error",
                category="media",
                active_backend=backend.model_name,
                message=f"live STT check failed ({info.category}): {type(exc).__name__}: {exc}",
                fix_hint=f"Check {backend.requires[0]}." if backend.requires else None,
                details=details,
            )

    if not ffmpeg_ok:
        bad = next(n for n, b in binaries.items() if b["status"] != "ok")
        return HealthStatus(
            "transcription",
            "degraded",
            category="media",
            active_backend=backend.model_name,
            message=f"{backend.model_name} ready for small audio files only; {bad} is {binaries[bad]['status']}",
            fix_hint=binaries[bad]["hint"],
            details=details,
        )
    message = f"{backend.model_name} via ffmpeg pipeline"
    if live:
        message += " (live check passed)"
    others = [p["env_var"] for p in providers if not p["configured"]]
    return HealthStatus(
        "transcription",
        "ok",
        category="media",
        active_backend=backend.model_name,
        message=message,
        fix_hint=None if not others else f"Optional: set {others[0]} for a fallback provider.",
        details=details,
    )


def media_download_status(live: bool = False) -> HealthStatus:
    """Health row for video/podcast page downloads (``yt-dlp``)."""
    probe = _ff.probe_tool("yt-dlp")
    detail = {"path": probe.path, "exit_code": probe.exit_code, "direct_urls": "ok (built in)"}
    if probe.ok:
        return HealthStatus(
            "media_download",
            "ok",
            category="media",
            active_backend="yt-dlp",
            message=f"yt-dlp {probe.version or ''}".strip(),
            details=detail,
        )
    first = probe.output.splitlines()[0] if probe.output else ""
    return HealthStatus(
        "media_download",
        probe.status,  # type: ignore[arg-type]
        category="media",
        message=(
            f"yt-dlp is {probe.status}; direct audio/video URLs and local files still work"
            + (f" ({first[:120]})" if first else "")
        ),
        fix_hint=probe.hint,
        details=detail,
    )


register_capability(
    "transcription",
    "media",
    transcription_status,
    description="Speech-to-text for audio/video files and URLs",
)
register_capability(
    "media_download",
    "media",
    media_download_status,
    description="Audio extraction from video/podcast pages (yt-dlp)",
)
