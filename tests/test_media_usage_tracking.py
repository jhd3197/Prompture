"""Media spend reaches the usage tracker, and $0 pricing holes are closed.

Before these fixes, media cost existed only on the immediate response
``meta`` — no media driver ever recorded a UsageEvent, video priced at $0
whenever ``duration`` wasn't passed explicitly, and async jobs reported no
cost at any lifecycle stage.
"""

from __future__ import annotations

import asyncio
from typing import Any

from prompture.drivers.img_gen_base import ImageGenDriver
from prompture.drivers.lipsync_base import AsyncLipsyncDriver
from prompture.drivers.muapi_aggregator_driver import MuapiLipsyncDriver, MuapiVideoGenDriver
from prompture.drivers.music_base import AsyncMusicGenDriver
from prompture.infra.tracker import UsageEvent, configure_tracker
from prompture.jobs import JobHandle, JobStatus


def _capture_events(tmp_path) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    configure_tracker(db_path=str(tmp_path / "u.db"), persist=False, sinks=[events.append])
    return events


class StubImageDriver(ImageGenDriver):
    model = "stub-image-1"

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "images": [],
            "meta": {"image_count": 2, "cost": 0.08, "model_name": "stub/stub-image-1"},
        }


class StubAsyncMusicDriver(AsyncMusicGenDriver):
    model = "stub-music-1"

    async def generate_music(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "audio": [],
            "meta": {"audio_count": 1, "cost": 0.5, "model_name": "stub/stub-music-1"},
        }


class StubAsyncLipsyncDriver(AsyncLipsyncDriver):
    model = "stub-lipsync-1"

    async def generate_lipsync(self, audio, options, *, image=None, video=None):
        return {
            "videos": [],
            "meta": {"video_count": 1, "cost": 0.2, "model_name": "stub/stub-lipsync-1"},
        }


def test_image_hooks_record_usage_event(tmp_path):
    events = _capture_events(tmp_path)

    StubImageDriver().generate_image_with_hooks("a cat", {})

    assert len(events) == 1
    assert events[0].cost == 0.08
    assert events[0].model_name == "stub/stub-image-1"
    assert events[0].metadata == {"modality": "image", "count": 2}


def test_image_hooks_record_error_event(tmp_path):
    events = _capture_events(tmp_path)

    class Boom(StubImageDriver):
        def generate_image(self, prompt, options):
            raise RuntimeError("nope")

    try:
        Boom().generate_image_with_hooks("a cat", {})
    except RuntimeError:
        pass

    assert len(events) == 1
    assert events[0].status == "error"
    assert events[0].error_type == "RuntimeError"


def test_async_music_hooks_await_and_record(tmp_path):
    # Regression: the async twin used to inherit the sync wrapper, which
    # cannot await the coroutine generate_music returns.
    events = _capture_events(tmp_path)

    resp = asyncio.run(StubAsyncMusicDriver().generate_music_with_hooks("lofi", {}))

    assert resp["meta"]["cost"] == 0.5
    assert len(events) == 1
    assert events[0].metadata == {"modality": "music", "count": 1}


def test_async_lipsync_hooks_await_and_record(tmp_path):
    events = _capture_events(tmp_path)

    resp = asyncio.run(StubAsyncLipsyncDriver().generate_lipsync_with_hooks(b"audio", {}))

    assert resp["meta"]["cost"] == 0.2
    assert len(events) == 1
    assert events[0].metadata == {"modality": "lipsync", "count": 1}


def test_video_duration_defaults_from_capabilities():
    # kling-video-v2-1 declares duration default=5 and is priced per second;
    # pricing with an implicit duration must quote the default, not $0.
    d = MuapiVideoGenDriver(api_key="k", model="kling-video-v2-1")
    assert d._effective_duration("kling-video-v2-1", {}) == 5.0
    assert d._effective_duration("kling-video-v2-1", {"duration": 10}) == 10.0
    cost = d._media_cost(
        "kling-video-v2-1", n=1, duration_seconds=d._effective_duration("kling-video-v2-1", {})
    )
    assert cost > 0.0


def test_resume_prices_completed_job_from_handle(monkeypatch):
    d = MuapiVideoGenDriver(api_key="k", model="kling-video-v2-1")
    handle = d._make_handle(
        "kling-video-v2-1", "req-1", "video",
        pricing={"n": 1, "duration_seconds": 5.0, "resolution": None},
    )
    # Pricing inputs survive serialization — that's the point of the handle.
    handle = JobHandle.from_json(handle.to_json())

    monkeypatch.setattr(d, "_fetch_once", lambda task_id: {"status": "completed"})
    monkeypatch.setattr(d, "status_of", lambda raw: raw.get("status", ""))
    monkeypatch.setattr(d, "extract_urls", lambda raw: ["https://x/video.mp4"])

    result = d.resume_job(handle, poll=False)

    assert result.status == JobStatus.COMPLETED.value
    assert result.meta.get("cost", 0.0) > 0.0


def test_resume_pending_job_has_no_cost(monkeypatch):
    d = MuapiVideoGenDriver(api_key="k", model="kling-video-v2-1")
    handle = d._make_handle("kling-video-v2-1", "req-1", "video", pricing={"n": 1, "duration_seconds": 5.0})

    monkeypatch.setattr(d, "_fetch_once", lambda task_id: {"status": "processing"})
    monkeypatch.setattr(d, "status_of", lambda raw: raw.get("status", ""))
    monkeypatch.setattr(d, "extract_urls", lambda raw: [])

    result = d.resume_job(handle, poll=False)

    assert "cost" not in result.meta


def test_lipsync_resume_resolves_via_lipsync_registry():
    from prompture.jobs import _resolve_driver

    handle = JobHandle(task_id="t", provider="muapi", model="infinite-talk", modality="lipsync")
    driver = _resolve_driver(handle)
    assert isinstance(driver, MuapiLipsyncDriver)
