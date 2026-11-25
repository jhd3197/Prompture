"""Tests for prompture.media.understand.frames (contact sheets, shots, clips).

The real-ffmpeg tests build a 6-second video of three solid-colour shots
(red, lime, blue; two seconds each) and skip when ffmpeg is not installed.
"""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
from pathlib import Path

import pytest

from prompture.drivers.vision_helpers import _prepare_google_vision_messages
from prompture.media.understand import frames as fr
from prompture.media.understand.tools import contact_sheet_media, detect_shots_media, video_understanding_tools
from prompture.media.video import VideoContent

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")


@pytest.fixture(scope="module")
def three_shots(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("frames") / "three.mp4"
    argv = ["ffmpeg", "-v", "error", "-y"]
    for colour in ("red", "lime", "blue"):
        argv += ["-f", "lavfi", "-i", f"color=c={colour}:s=320x180:r=25:d=2"]
    argv += [
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=6",
        "-filter_complex",
        "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map",
        "[v]",
        "-map",
        "3:a",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        str(out),
    ]
    subprocess.run(argv, check=True)
    return out


def test_sample_times_spreads_frames_in_the_middle_of_each_slot():
    assert fr.sample_times(0, 60, 3) == [10.0, 30.0, 50.0]
    assert fr.sample_times(10, 10, 3) == []
    assert fr.sample_times(0, 10, 0) == []


def test_format_clock():
    assert fr.format_clock(5733.4) == "1:35:33"
    assert fr.format_clock(65) == "1:05"
    assert fr.format_clock(-3) == "0:00"


def test_snap_to_shots_moves_only_within_reach():
    shots = [fr.Shot(0, 4.2), fr.Shot(4.2, 9.8), fr.Shot(9.8, 20)]
    assert fr.snap_to_shots(5.0, shots, max_move=1.0) == 4.2
    assert fr.snap_to_shots(7.0, shots, max_move=1.0) == 7.0


def test_contact_sheet_refuses_huge_grids():
    with pytest.raises(ValueError):
        fr.contact_sheet("x.mp4", times=[1.0], cols=10, rows=10)


def test_google_messages_carry_video_blocks_as_inline_data():
    clip = VideoContent(data="AAAA", media_type="video/mp4")
    out = _prepare_google_vision_messages(
        [{"role": "user", "content": [{"type": "text", "text": "what happens?"}, {"type": "video", "source": clip}]}]
    )
    assert out[0]["content"][1] == {"inline_data": {"mime_type": "video/mp4", "data": "AAAA"}}


def test_video_tools_are_named():
    assert [t.name for t in video_understanding_tools()] == ["contact_sheet", "detect_shots"]


def test_tools_return_error_text_instead_of_raising(tmp_path):
    assert contact_sheet_media(str(tmp_path / "missing.mp4"), 0, 10).startswith("Error:")
    assert detect_shots_media(str(tmp_path / "missing.mp4")).startswith("Error:")


@needs_ffmpeg
def test_probe_video(three_shots):
    info = fr.probe_video(three_shots)
    assert info.width == 320 and info.height == 180
    assert info.duration == pytest.approx(6.0, abs=0.2)
    assert info.fps == pytest.approx(25.0)


@needs_ffmpeg
def test_extract_frame_is_a_jpeg_of_the_right_colour(three_shots):
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.open(io.BytesIO(fr.extract_frame(three_shots, 3.0, height=90))).convert("RGB")
    assert img.height == 90
    r, g, b = img.getpixel((img.width // 2, img.height // 2))
    assert g > 150 and r < 80 and b < 80


@needs_ffmpeg
def test_contact_sheet_tiles_and_labels(three_shots):
    pytest.importorskip("PIL")
    from PIL import Image

    sheet = fr.contact_sheet(three_shots, start=0, end=6, cols=3, rows=1, tile_height=90)
    assert sheet.times == [1.0, 3.0, 5.0]
    assert (sheet.cols, sheet.rows) == (3, 1)
    img = Image.open(io.BytesIO(sheet.data)).convert("RGB")
    assert img.size == (sheet.width, sheet.height) == (160 * 3 + 8, 90)
    # The middle of each tile shows that shot's colour.
    assert [max(range(3), key=lambda c: img.getpixel((80 + i * 164, 30))[c]) for i in range(3)] == [0, 1, 2]
    assert sheet.legend() == "#1 0:01 | #2 0:03 | #3 0:05"
    assert sheet.to_image().media_type == "image/jpeg"


@needs_ffmpeg
def test_contact_sheet_tool_saves_the_jpeg(three_shots, tmp_path):
    pytest.importorskip("PIL")
    text = contact_sheet_media(str(three_shots), 0, 6, cols=2, rows=1, out_dir=str(tmp_path))
    assert text.startswith("Contact sheet saved:")
    assert list(tmp_path.glob("contact-*.jpg"))


@needs_ffmpeg
def test_detect_shots_finds_the_two_cuts(three_shots):
    shots = fr.detect_shots(three_shots, method="ffmpeg")
    assert len(shots) == 3
    assert shots[1].start == pytest.approx(2.0, abs=0.1)
    assert shots[2].start == pytest.approx(4.0, abs=0.1)
    assert shots[-1].end == pytest.approx(6.0, abs=0.2)


@needs_ffmpeg
def test_detect_shots_in_a_range_reports_absolute_times(three_shots):
    shots = fr.detect_shots(three_shots, start=3.0, end=6.0, method="ffmpeg")
    assert [round(s.start, 1) for s in shots] == [3.0, 4.0]
    assert shots[-1].end == 6.0


@needs_ffmpeg
def test_video_clip_is_a_small_mp4(three_shots):
    clip = fr.video_clip(three_shots, 1.0, 3.0, height=90)
    assert clip.media_type == "video/mp4"
    assert clip.duration_seconds == 2.0
    data = base64.b64decode(clip.data)
    assert data[4:8] == b"ftyp"


def test_detect_shots_can_be_cancelled(monkeypatch):
    import threading

    killed = []

    class Hung:
        returncode = None
        stderr = None

        def __init__(self, *a, **k):
            pass

        def poll(self):
            return None if not killed else -9

        def wait(self, timeout=None):
            if killed:
                return -9
            raise subprocess.TimeoutExpired("ffmpeg", timeout)

        def kill(self):
            killed.append(True)

    monkeypatch.setattr(fr, "tool_path", lambda name: name)
    monkeypatch.setattr(fr.subprocess, "Popen", Hung)
    stop = threading.Event()
    stop.set()
    with pytest.raises(fr.ShotDetectionCancelled):
        fr.detect_shots("film.mkv", end=10.0, method="ffmpeg", cancel=stop)
    assert killed
