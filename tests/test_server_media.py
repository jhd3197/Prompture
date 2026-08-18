"""Tests for the gateway media endpoints: /v1/videos, /v1/lipsync, /v1/jobs (G24)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from prompture.cli.server import create_app
from prompture.jobs import JobHandle, JobResult, MediaAsset


class _MockAsyncVideoDriver:
    async def generate_video(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        from prompture.media.video import video_from_url

        if options.get("poll") is False:
            handle = JobHandle(task_id="vt1", provider="muapi", model="kling", modality="video")
            return {
                "videos": [],
                "meta": {
                    "video_count": 0,
                    "status": "pending",
                    "job_handle": handle.to_dict(),
                    "model_name": "muapi/kling",
                    "cost": 0.0,
                },
            }
        return {
            "videos": [video_from_url("https://x/v.mp4")],
            "meta": {"video_count": 1, "model_name": "muapi/kling", "cost": 0.5},
        }


class _MockAsyncLipsyncDriver:
    async def generate_lipsync(self, audio, options, *, image=None, video=None):
        from prompture.media.video import video_from_url

        if options.get("poll") is False:
            handle = JobHandle(task_id="lt1", provider="muapi", model="infinite-talk", modality="lipsync")
            return {
                "videos": [],
                "meta": {"status": "pending", "job_handle": handle.to_dict(), "model_name": "muapi/infinite-talk"},
            }
        return {
            "videos": [video_from_url("https://x/talk.mp4")],
            "meta": {"video_count": 1, "category": "image", "model_name": "muapi/infinite-talk", "cost": 0.2},
        }


class _MockAsyncMusicDriver:
    async def generate_music(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        from prompture.media.audio import audio_from_url

        if options.get("poll") is False:
            handle = JobHandle(task_id="mt1", provider="muapi", model="suno-create-music", modality="music")
            return {
                "audio": [],
                "meta": {"status": "pending", "job_handle": handle.to_dict(), "model_name": "muapi/suno-create-music"},
            }
        return {
            "audio": [audio_from_url("https://x/song.mp3")],
            "meta": {"audio_count": 1, "operation": "create", "model_name": "muapi/suno-create-music", "cost": 0.1},
        }


_VID = "prompture.drivers.video_gen_registry.get_async_video_gen_driver_for_model"
_LIP = "prompture.drivers.lipsync_registry.get_async_lipsync_driver_for_model"
_MUS = "prompture.drivers.music_registry.get_async_music_driver_for_model"


@pytest.fixture
def client():
    app = create_app(model_name="muapi/kling")
    with TestClient(app) as c:
        yield c


class TestVideos:
    def test_sync_returns_url_and_cost(self, client):
        with patch(_VID, return_value=_MockAsyncVideoDriver()):
            resp = client.post(
                "/v1/videos/generations", json={"model": "muapi/kling", "prompt": "a dog", "duration": 5}
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["data"][0]["url"] == "https://x/v.mp4"
        assert data["cost"] == 0.5
        assert data["model"] == "muapi/kling"

    def test_poll_false_returns_202_job(self, client):
        with patch(_VID, return_value=_MockAsyncVideoDriver()):
            resp = client.post("/v1/videos/generations", json={"model": "muapi/kling", "prompt": "x", "poll": False})
        assert resp.status_code == 202, resp.text
        data = resp.json()
        assert data["status"] == "pending"
        assert data["id"].startswith("job_")
        assert data["object"] == "media.job"

    def test_unknown_model_400(self, client):
        with patch(_VID, side_effect=ValueError("nope")):
            resp = client.post("/v1/videos/generations", json={"model": "bad/x", "prompt": "x"})
        assert resp.status_code == 400


class TestLipsync:
    def test_image_audio_sync(self, client):
        with patch(_LIP, return_value=_MockAsyncLipsyncDriver()):
            resp = client.post(
                "/v1/lipsync",
                json={"model": "muapi/infinite-talk", "audio_url": "https://in/a.mp3", "image_url": "https://in/p.png"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"][0]["url"] == "https://x/talk.mp4"

    def test_poll_false_202(self, client):
        with patch(_LIP, return_value=_MockAsyncLipsyncDriver()):
            resp = client.post(
                "/v1/lipsync",
                json={"audio_url": "https://in/a.mp3", "image_url": "https://in/p.png", "poll": False},
            )
        assert resp.status_code == 202
        assert resp.json()["id"].startswith("job_")

    def test_missing_audio_url_422(self, client):
        with patch(_LIP, return_value=_MockAsyncLipsyncDriver()):
            resp = client.post("/v1/lipsync", json={"image_url": "https://in/p.png"})
        assert resp.status_code == 422  # audio_url is required


class TestMusic:
    def test_create_sync(self, client):
        with patch(_MUS, return_value=_MockAsyncMusicDriver()):
            resp = client.post(
                "/v1/music", json={"model": "muapi/suno-create-music", "prompt": "lofi beats", "instrumental": True}
            )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["data"][0]["url"] == "https://x/song.mp3"
        assert data["cost"] == 0.1

    def test_poll_false_202(self, client):
        with patch(_MUS, return_value=_MockAsyncMusicDriver()):
            resp = client.post("/v1/music", json={"prompt": "x", "poll": False})
        assert resp.status_code == 202
        assert resp.json()["id"].startswith("job_")


class TestJobs:
    def test_submit_then_poll_job(self, client):
        with patch(_VID, return_value=_MockAsyncVideoDriver()):
            submit = client.post("/v1/videos/generations", json={"model": "muapi/kling", "prompt": "x", "poll": False})
        job_id = submit.json()["id"]

        fake = JobResult(
            handle=JobHandle(task_id="vt1", provider="muapi", model="kling", modality="video").with_status("completed"),
            status="completed",
            assets=[MediaAsset(kind="video", url="https://x/done.mp4")],
        )
        with patch("prompture.jobs.resume", return_value=fake) as mr:
            resp = client.get(f"/v1/jobs/{job_id}")
        mr.assert_called_once()
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "completed"
        assert data["done"] is True
        assert data["data"][0]["url"] == "https://x/done.mp4"
        assert data["model"] == "muapi/kling"

    def test_unknown_job_404(self, client):
        resp = client.get("/v1/jobs/job_doesnotexist")
        assert resp.status_code == 404
