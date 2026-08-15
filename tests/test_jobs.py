"""Tests for the media job abstraction (jobs.py) and the shared poller (job_runner.py)."""

from __future__ import annotations

import pytest

from prompture.drivers.job_runner import (
    JobFailedError,
    normalize_status,
    poll_job,
)
from prompture.drivers.registry import (
    register_video_gen_driver,
    unregister_video_gen_driver,
)
from prompture.jobs import (
    JobHandle,
    JobNotResumableError,
    JobResult,
    JobStatus,
    MediaAsset,
    fetch_result,
    get_status,
    resume,
)


class TestMediaAsset:
    def test_dict_roundtrip(self):
        a = MediaAsset(kind="image", url="https://x/y.png", model="muapi/nano", provenance={"seed": 7})
        assert MediaAsset.from_dict(a.to_dict()) == a

    def test_from_dict_ignores_unknown_keys(self):
        a = MediaAsset.from_dict({"kind": "video", "url": "u", "bogus": 1})
        assert a.kind == "video" and a.url == "u"

    def test_to_content_image_url(self):
        content = MediaAsset(kind="image", url="https://x/y.png").to_content()
        assert content.source_type == "url"
        assert content.url == "https://x/y.png"

    def test_to_content_unknown_kind(self):
        with pytest.raises(ValueError):
            MediaAsset(kind="hologram").to_content()


class TestJobHandle:
    def test_json_roundtrip(self):
        h = JobHandle(task_id="req1", provider="muapi", model="nano-banana", modality="image")
        assert JobHandle.from_json(h.to_json()) == h

    def test_from_meta(self):
        h = JobHandle(task_id="r", provider="muapi", model="m", modality="video")
        meta = {"job_handle": h.to_dict(), "request_id": "r"}
        assert JobHandle.from_meta(meta) == h

    def test_with_status(self):
        h = JobHandle(task_id="r", provider="muapi", model="m", modality="image")
        assert h.with_status("completed").status == "completed"
        assert h.status == JobStatus.PENDING.value  # original unchanged (frozen)


class TestJobResult:
    def test_first_url_and_done(self):
        r = JobResult(
            handle=JobHandle(task_id="r", provider="p", model="m", modality="image"),
            status="completed",
            assets=[MediaAsset(kind="image"), MediaAsset(kind="image", url="u2")],
        )
        assert r.done is True
        assert r.first_url == "u2"

    def test_dict_roundtrip(self):
        r = JobResult(
            handle=JobHandle(task_id="r", provider="p", model="m", modality="video"),
            status="running",
            assets=[MediaAsset(kind="video", url="v")],
            meta={"x": 1},
        )
        assert JobResult.from_dict(r.to_dict()) == r


class TestNormalizeStatus:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("COMPLETED", "completed"),
            ("succeeded", "completed"),
            ("success", "completed"),
            ("FAILED", "failed"),
            ("error", "failed"),
            ("canceled", "failed"),
            ("queued", "pending"),
            ("in_queue", "pending"),
            ("processing", "running"),
            ("IN_PROGRESS", "running"),
            (None, "running"),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_status(raw) == expected


class TestPollJob:
    def test_polls_until_complete(self):
        states = iter([{"status": "queued"}, {"status": "processing"}, {"status": "completed", "outputs": ["u"]}])
        slept: list[float] = []
        raw = poll_job(lambda: next(states), interval=1.0, timeout=100.0, sleep=slept.append, monotonic=lambda: 0.0)
        assert raw["outputs"] == ["u"]
        assert slept == [1.0, 1.0]  # slept twice before completing

    def test_failure_raises(self):
        with pytest.raises(JobFailedError):
            poll_job(lambda: {"status": "failed", "error": "boom"}, sleep=lambda _: None, monotonic=lambda: 0.0)

    def test_timeout_raises(self):
        clock = iter([0.0, 0.0, 999.0])  # deadline check trips on 3rd read
        with pytest.raises(TimeoutError):
            poll_job(
                lambda: {"status": "processing"},
                interval=1.0,
                timeout=10.0,
                sleep=lambda _: None,
                monotonic=lambda: next(clock),
            )


class _FakeVideoDriver:
    """Minimal driver implementing the resume_job contract."""

    def __init__(self, model=None):
        self.model = model

    def resume_job(self, handle, *, poll=True, poll_interval=None, timeout=None):
        return JobResult(
            handle=handle.with_status("completed"),
            status="completed",
            assets=[MediaAsset(kind="video", url="https://x/out.mp4", model=handle.model)],
            meta={"polled": poll},
        )

    def get_job_status(self, handle):
        return "running"


class _NoResumeDriver:
    def __init__(self, model=None):
        self.model = model


@pytest.fixture
def _fake_provider():
    register_video_gen_driver("fakeprov", lambda model=None: _FakeVideoDriver(model), overwrite=True)
    try:
        yield "fakeprov"
    finally:
        unregister_video_gen_driver("fakeprov")


class TestResumeDispatch:
    def test_resume_dispatches_to_driver(self, _fake_provider):
        h = JobHandle(task_id="req9", provider="fakeprov", model="m1", modality="video")
        result = resume(h)
        assert result.done is True
        assert result.first_url == "https://x/out.mp4"
        assert result.meta["polled"] is True

    def test_resume_accepts_json_handle(self, _fake_provider):
        h = JobHandle(task_id="req9", provider="fakeprov", model="m1", modality="video")
        result = resume(h.to_json())
        assert result.status == "completed"

    def test_fetch_result_uses_no_poll(self, _fake_provider):
        h = JobHandle(task_id="req9", provider="fakeprov", model="m1", modality="video")
        assert fetch_result(h).meta["polled"] is False

    def test_get_status(self, _fake_provider):
        h = JobHandle(task_id="req9", provider="fakeprov", model="m1", modality="video")
        assert get_status(h) == "running"

    def test_unsupported_modality_raises(self):
        h = JobHandle(task_id="r", provider="x", model="m", modality="hologram")
        with pytest.raises(JobNotResumableError):
            resume(h)

    def test_driver_without_resume_raises(self):
        register_video_gen_driver("noresume", lambda model=None: _NoResumeDriver(model), overwrite=True)
        try:
            h = JobHandle(task_id="r", provider="noresume", model="m", modality="video")
            with pytest.raises(JobNotResumableError):
                resume(h)
        finally:
            unregister_video_gen_driver("noresume")
