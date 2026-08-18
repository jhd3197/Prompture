"""Tests for media generation exposed as agent tools (media/agent_tools.py)."""

from __future__ import annotations

from unittest.mock import patch

from prompture.agents import ToolRegistry
from prompture.jobs import JobHandle, JobResult, MediaAsset
from prompture.media.agent_tools import (
    edit_image,
    generate_image,
    generate_video,
    list_media_models,
    media_tool_definitions,
    register_media_tools,
    resume_media_job,
)
from prompture.media.image import image_from_url
from prompture.media.video import video_from_url


class _FakeImageDriver:
    def __init__(self):
        self.last_options = None

    def generate_image(self, prompt, options):
        self.last_options = options
        return {
            "images": [image_from_url("https://x/a.png")],
            "meta": {"model_name": "muapi/nano", "cost": 0.0, "request_id": "r1"},
        }

    def edit_image(self, image, prompt, options, *, mask=None):
        self.last_options = options
        return {
            "images": [image_from_url("https://x/edited.png")],
            "meta": {"model_name": "muapi/seedream", "cost": 0.0},
        }


class _FakeVideoDriver:
    def __init__(self):
        self.last_options = None

    def generate_video(self, prompt, options):
        self.last_options = options
        return {
            "videos": [video_from_url("https://x/v.mp4")],
            "meta": {"model_name": "muapi/kling", "cost": 0.0, "request_id": "rv"},
        }


_IMG = "prompture.drivers.img_gen_registry.get_img_gen_driver_for_model"
_VID = "prompture.drivers.video_gen_registry.get_video_gen_driver_for_model"


class TestGenerateImageTool:
    def test_returns_urls(self):
        fake = _FakeImageDriver()
        with patch(_IMG, return_value=fake):
            out = generate_image("muapi/nano", "a cat", aspect_ratio="16:9", n=2)
        assert out["urls"] == ["https://x/a.png"]
        assert out["model"] == "muapi/nano"
        assert fake.last_options == {"aspect_ratio": "16:9", "num_images": 2}


class TestEditImageTool:
    def test_passes_image_url(self):
        fake = _FakeImageDriver()
        with patch(_IMG, return_value=fake):
            out = edit_image("muapi/seedream", "https://in/img.png", "make it blue")
        assert out["urls"] == ["https://x/edited.png"]
        assert fake.last_options["image_url"] == "https://in/img.png"


class TestGenerateVideoTool:
    def test_t2v(self):
        fake = _FakeVideoDriver()
        with patch(_VID, return_value=fake):
            out = generate_video("muapi/kling", "a dog", duration=5)
        assert out["urls"] == ["https://x/v.mp4"]
        assert fake.last_options == {"duration": 5}

    def test_i2v_with_image_url(self):
        fake = _FakeVideoDriver()
        with patch(_VID, return_value=fake):
            generate_video("muapi/kling", "move", image_url="https://in/f.png", aspect_ratio="16:9")
        assert fake.last_options["image_url"] == "https://in/f.png"
        assert fake.last_options["aspect_ratio"] == "16:9"


class TestResumeTool:
    def test_resume(self):
        handle = JobHandle(task_id="r", provider="muapi", model="m", modality="video")
        fake_result = JobResult(
            handle=handle.with_status("completed"),
            status="completed",
            assets=[MediaAsset(kind="video", url="https://x/done.mp4")],
        )
        with patch("prompture.jobs.resume", return_value=fake_result) as mr:
            out = resume_media_job(handle.to_json())
        mr.assert_called_once()
        assert out["done"] is True
        assert out["urls"] == ["https://x/done.mp4"]


class TestListModelsTool:
    def test_list(self):
        out = list_media_models("video")
        assert "muapi/kling-video-v2-1" in out["models"]


class TestRegistration:
    def test_definitions(self):
        defs = media_tool_definitions()
        names = {d.name for d in defs}
        assert names == {
            "generate_image",
            "edit_image",
            "generate_video",
            "generate_music",
            "resume_media_job",
            "list_media_models",
        }

    def test_register_into_registry(self):
        reg = register_media_tools(ToolRegistry())
        assert len(reg) == 6
        td = reg.get("generate_image")
        assert td.parameters["required"] == ["model", "prompt"]
        assert "aspect_ratio" in td.parameters["properties"]

    def test_tool_openai_format(self):
        td = media_tool_definitions()[0]
        fmt = td.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "generate_image"
