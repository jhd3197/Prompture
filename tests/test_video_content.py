"""Tests for prompture.media.video."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from prompture.media.video import (
    VideoContent,
    _guess_media_type,
    _guess_media_type_from_bytes,
    make_video,
    video_from_base64,
    video_from_bytes,
    video_from_file,
    video_from_url,
)


class TestVideoContent:
    """Tests for VideoContent."""

    def test_frozen(self):
        video = VideoContent(data="abc", media_type="video/mp4")
        with pytest.raises(AttributeError):
            video.data = "xyz"  # type: ignore[misc]

    def test_defaults(self):
        video = VideoContent(data="abc", media_type="video/mp4")
        assert video.source_type == "base64"
        assert video.url is None
        assert video.duration_seconds is None
        assert video.width is None
        assert video.height is None

    def test_url_source(self):
        video = VideoContent(data="", media_type="video/mp4", source_type="url", url="https://example.com/v.mp4")
        assert video.source_type == "url"
        assert video.url == "https://example.com/v.mp4"


class TestVideoMediaDetection:
    """Tests for MIME detection."""

    def test_mp4_magic(self):
        data = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50
        assert _guess_media_type_from_bytes(data) == "video/mp4"

    def test_mov_magic(self):
        data = b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 50
        assert _guess_media_type_from_bytes(data) == "video/quicktime"

    def test_webm_magic(self):
        data = b"\x1aE\xdf\xa3" + b"\x00" * 50
        assert _guess_media_type_from_bytes(data) == "video/webm"

    def test_extension(self):
        assert _guess_media_type("movie.webm") == "video/webm"
        assert _guess_media_type("clip.mov") == "video/quicktime"

    def test_url_with_query(self):
        assert _guess_media_type("https://cdn.example.com/clip.mp4?token=abc") == "video/mp4"


class TestVideoConstructors:
    """Tests for video constructor helpers."""

    def test_video_from_bytes(self):
        raw = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50
        video = video_from_bytes(raw, duration_seconds=2.5, width=1280, height=720)
        assert video.media_type == "video/mp4"
        assert base64.b64decode(video.data) == raw
        assert video.duration_seconds == 2.5
        assert video.width == 1280
        assert video.height == 720

    def test_empty_bytes_raise(self):
        with pytest.raises(ValueError, match="empty"):
            video_from_bytes(b"")

    def test_video_from_base64_data_uri(self):
        raw_b64 = base64.b64encode(b"video bytes").decode()
        video = video_from_base64(f"data:video/webm;base64,{raw_b64}")
        assert video.data == raw_b64
        assert video.media_type == "video/webm"

    def test_video_from_file(self, tmp_path: Path):
        path = tmp_path / "clip.mp4"
        raw = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50
        path.write_bytes(raw)
        video = video_from_file(path)
        assert video.media_type == "video/mp4"
        assert base64.b64decode(video.data) == raw

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            video_from_file("/nonexistent/video.mp4")

    def test_video_from_url(self):
        video = video_from_url("https://example.com/clip.mp4", duration_seconds=8)
        assert video.source_type == "url"
        assert video.url == "https://example.com/clip.mp4"
        assert video.media_type == "video/mp4"
        assert video.duration_seconds == 8


class TestMakeVideo:
    """Tests for make_video."""

    def test_passthrough(self):
        video = VideoContent(data="abc", media_type="video/mp4")
        assert make_video(video) is video

    def test_url_string(self):
        video = make_video("https://example.com/clip.webm")
        assert video.source_type == "url"
        assert video.media_type == "video/webm"

    def test_file_path_string(self, tmp_path: Path):
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50)
        video = make_video(str(path))
        assert video.media_type == "video/mp4"

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported"):
            make_video(42)  # type: ignore[arg-type]
