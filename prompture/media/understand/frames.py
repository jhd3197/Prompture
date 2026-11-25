"""Let a model "watch" a video: frames, contact sheets, shot cuts and clips.

Most chat models take images, not video. The usual way to show them a video is
to sample frames, tile them into a grid with the timestamp burned into each
tile (a *contact sheet*), and send that grid as one image. Gemini also takes
short clips natively, so :func:`video_clip` cuts one to a small MP4.

* :func:`probe_video` — duration and frame size (ffprobe).
* :func:`sample_times` — evenly spaced timestamps in a range.
* :func:`extract_frame` / :func:`extract_frames` — JPEG stills at given times.
* :func:`contact_sheet` — a labelled grid of frames as one JPEG (needs Pillow).
* :func:`detect_shots` — camera cuts with the ffmpeg scene score (or
  PySceneDetect when installed) as a list of :class:`Shot`.
* :func:`video_clip` — a small H.264 MP4 of a range, as :class:`VideoContent`.

Every ffmpeg call is an argv list with a timeout (see :mod:`.ffmpeg`).
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...capabilities.probe import probe_env
from ..image import ImageContent
from ..video import VideoContent
from .errors import MediaProcessingError, MediaToolMissingError
from .ffmpeg import run_media_command, tool_path

#: ffmpeg scene score above which a frame starts a new shot.
SCENE_THRESHOLD = 0.32
#: Cuts closer together than this (seconds) are flashes, merged into one shot.
MIN_SHOT = 0.4
#: Seconds allowed per extracted frame (4K HEVC seeks can be slow).
FRAME_TIMEOUT = 60.0
#: Largest grid accepted by :func:`contact_sheet`.
MAX_TILES = 48


@dataclass(frozen=True)
class VideoInfo:
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None = None


@dataclass(frozen=True)
class Shot:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ContactSheet:
    """A grid of frames as one JPEG, with the time of each tile."""

    data: bytes
    times: list[float]
    cols: int
    rows: int
    width: int
    height: int
    media_type: str = "image/jpeg"
    notes: list[str] = field(default_factory=list)

    def to_image(self) -> ImageContent:
        """The sheet as an :class:`ImageContent` for a vision message."""
        return ImageContent(data=base64.b64encode(self.data).decode("ascii"), media_type=self.media_type)

    def save(self, path: str | os.PathLike[str]) -> Path:
        out = Path(path)
        out.write_bytes(self.data)
        return out

    def legend(self) -> str:
        """Tile order and times as text, to send next to the image."""
        return " | ".join(f"#{i + 1} {format_clock(t)}" for i, t in enumerate(self.times))


def format_clock(seconds: float) -> str:
    """``5733.4`` -> ``"1:35:33"``; under an hour -> ``"35:33"``."""
    s = max(0, round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def probe_video(path: str | os.PathLike[str], *, timeout: float = 30.0) -> VideoInfo:
    """Duration, width, height and frame rate of the first video stream."""
    ffprobe = tool_path("ffprobe")
    proc = run_media_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            os.fspath(path),
        ],
        timeout=timeout,
        what="ffprobe",
    )
    try:
        doc = json.loads((proc.stdout or b"{}").decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise MediaProcessingError(f"ffprobe returned unreadable output: {exc}") from exc
    stream = (doc.get("streams") or [{}])[0]
    fps = None
    rate = str(stream.get("avg_frame_rate") or "")
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            fps = float(num) / float(den) if float(den) else None
        except ValueError:
            fps = None
    try:
        duration = float((doc.get("format") or {}).get("duration") or 0) or None
    except ValueError:
        duration = None
    return VideoInfo(duration=duration, width=stream.get("width"), height=stream.get("height"), fps=fps)


def sample_times(start: float, end: float, n: int) -> list[float]:
    """*n* timestamps spread over ``[start, end)``, each in the middle of its slot."""
    if n <= 0 or end <= start:
        return []
    step = (end - start) / n
    return [round(start + step * (i + 0.5), 3) for i in range(n)]


def extract_frame(
    path: str | os.PathLike[str],
    t: float,
    *,
    height: int = 360,
    quality: int = 4,
    timeout: float = FRAME_TIMEOUT,
) -> bytes:
    """One JPEG frame at *t* seconds, scaled to *height* (aspect kept).

    HDR sources are tone-mapped to SDR when ffmpeg has the ``zscale`` filter;
    otherwise they come out flat but readable.
    """
    ffmpeg = tool_path("ffmpeg")
    scale = f"scale=-2:{int(height)}:flags=bilinear"
    argv = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{max(0.0, t):.3f}",
        "-i",
        os.fspath(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-vf",
        f"{scale},format=yuvj420p",
        "-q:v",
        str(int(quality)),
        "-f",
        "image2pipe",
        "-c:v",
        "mjpeg",
        "-",
    ]
    proc = run_media_command(argv, timeout=timeout, what="ffmpeg frame")
    if not proc.stdout:
        raise MediaProcessingError(f"ffmpeg returned no frame at {t:.2f}s (past the end?)")
    return proc.stdout


def extract_frames(path: str | os.PathLike[str], times: Sequence[float], *, height: int = 360) -> list[bytes]:
    """JPEG frames at each of *times* (one seek per frame: fast for sparse samples)."""
    return [extract_frame(path, t, height=height) for t in times]


def _pil():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MediaToolMissingError("Pillow", "missing", "pip install pillow (or prompture[media])") from exc
    return Image, ImageDraw, ImageFont


def _font(ImageFont, size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def contact_sheet(
    path: str | os.PathLike[str],
    *,
    start: float | None = None,
    end: float | None = None,
    times: Sequence[float] | None = None,
    cols: int = 3,
    rows: int = 2,
    tile_height: int = 240,
    label: bool = True,
    quality: int = 85,
) -> ContactSheet:
    """Tile frames into one labelled JPEG grid.

    Give either explicit *times* or a ``start``/``end`` range (frames spread
    evenly over it; the whole video when both are omitted). Each tile shows
    its timestamp in the bottom-left corner, so a model can answer with times.
    """
    cols, rows = max(1, int(cols)), max(1, int(rows))
    if cols * rows > MAX_TILES:
        raise ValueError(f"at most {MAX_TILES} tiles per sheet (asked for {cols}x{rows})")
    if times is None:
        if start is None or end is None:
            info = probe_video(path)
            start = 0.0 if start is None else start
            end = info.duration if end is None else end
            if not end:
                raise MediaProcessingError("could not read the video duration; pass start and end")
        times = sample_times(float(start), float(end), cols * rows)
    times = list(times)[: cols * rows]
    if not times:
        raise ValueError("no frame times (empty range)")
    Image, ImageDraw, ImageFont = _pil()
    frames = []
    notes: list[str] = []
    for t in times:
        try:
            frames.append(Image.open(io.BytesIO(extract_frame(path, t, height=tile_height))).convert("RGB"))
        except MediaProcessingError as exc:
            notes.append(f"{format_clock(t)}: {exc}")
            frames.append(None)
    real = [f for f in frames if f is not None]
    if not real:
        raise MediaProcessingError("no frames could be read: " + "; ".join(notes)[:500])
    tw = max(f.width for f in real)
    th = max(f.height for f in real)
    gap = 4
    used_rows = (len(times) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * gap, used_rows * th + (used_rows - 1) * gap), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    font = _font(ImageFont, max(12, th // 11))
    for i, (img, t) in enumerate(zip(frames, times, strict=True)):
        x, y = (i % cols) * (tw + gap), (i // cols) * (th + gap)
        if img is not None:
            sheet.paste(img, (x + (tw - img.width) // 2, y + (th - img.height) // 2))
        if label:
            text = f"#{i + 1} {format_clock(t)}"
            box = draw.textbbox((0, 0), text, font=font)
            w, h = box[2] - box[0], box[3] - box[1]
            pad = max(3, th // 60)
            draw.rectangle([x, y + th - h - 3 * pad, x + w + 2 * pad, y + th], fill=(0, 0, 0))
            draw.text((x + pad, y + th - h - 2 * pad - box[1]), text, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=int(quality))
    return ContactSheet(
        data=buf.getvalue(),
        times=[float(t) for t in times],
        cols=cols,
        rows=used_rows,
        width=sheet.width,
        height=sheet.height,
        notes=notes,
    )


_PTS = re.compile(r"pts_time:([\d.]+)")
_SCORE = re.compile(r"lavfi\.scene_score=([\d.]+)")


class ShotDetectionCancelled(MediaProcessingError):
    """``cancel`` was set while shot detection ran."""


def _ffmpeg_cuts(
    path: str,
    start: float | None,
    end: float | None,
    threshold: float,
    height: int,
    timeout: float,
    cancel: Any = None,
    progress: Callable[[float], None] | None = None,
) -> list[float]:
    ffmpeg = tool_path("ffmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        argv = [ffmpeg, "-v", "error", "-nostdin", "-nostats", "-progress", "pipe:1"]
        if start:
            argv += ["-ss", f"{start:.3f}"]
        argv += ["-i", path]
        if end is not None:
            argv += ["-t", f"{max(0.0, end - (start or 0.0)):.3f}"]
        # The metadata filter writes a file relative to the cwd: running from
        # the temp folder keeps a Windows drive letter out of the filter string.
        argv += [
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"scale=-2:{int(height)},select='gt(scene\\,{threshold})',metadata=print:file=cuts.txt",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.Popen(  # nosec B603 - argv list, no shell
            argv,
            cwd=tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=probe_env(),
            stdin=subprocess.DEVNULL,
        )
        # Drain both pipes on threads: ffmpeg blocks if a pipe fills up, and
        # "-progress pipe:1" reports how far it has read (out_time_us=...).
        seen = [0.0]
        errs: list[bytes] = []

        def read_progress() -> None:
            for raw in iter(proc.stdout.readline, b"") if proc.stdout else []:
                if raw.startswith(b"out_time_us="):
                    with contextlib.suppress(ValueError):
                        seen[0] = int(raw.split(b"=", 1)[1]) / 1_000_000

        def read_errors() -> None:
            if proc.stderr:
                errs.append(proc.stderr.read())

        readers = [
            threading.Thread(target=read_progress, daemon=True),
            threading.Thread(target=read_errors, daemon=True),
        ]
        for r in readers:
            r.start()
        waited = 0.0
        while proc.poll() is None:
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                waited += 0.5
                if progress is not None:
                    progress(seen[0] + (start or 0.0))
                if cancel is not None and cancel.is_set():
                    proc.kill()
                    proc.wait()
                    raise ShotDetectionCancelled("shot detection was cancelled") from None
                if waited >= timeout:
                    proc.kill()
                    proc.wait()
                    raise MediaProcessingError(f"shot detection timed out after {timeout:g}s") from None
        for r in readers:
            r.join(timeout=5)
        err = b"".join(errs)
        if proc.returncode:
            raise MediaProcessingError("ffmpeg shot detection failed: " + err.decode("utf-8", errors="replace")[-400:])
        marks = Path(tmp) / "cuts.txt"
        text = marks.read_text(encoding="utf-8", errors="replace") if marks.is_file() else ""
    cuts: list[float] = []
    pending = None
    for line in text.splitlines():
        if m := _PTS.search(line):
            pending = float(m.group(1))
        elif _SCORE.search(line) and pending is not None:
            cuts.append(pending + (start or 0.0))
            pending = None
    return cuts


def _pyscenedetect_cuts(path: str, start: float | None, end: float | None) -> list[float]:
    from scenedetect import ContentDetector, detect  # type: ignore[import-not-found]

    kwargs = {}
    if start:
        kwargs["start_time"] = float(start)
    if end is not None:
        kwargs["end_time"] = float(end)
    scenes = detect(path, ContentDetector(), **kwargs)
    return [s[0].get_seconds() for s in scenes[1:]]


def detect_shots(
    path: str | os.PathLike[str],
    *,
    start: float | None = None,
    end: float | None = None,
    threshold: float = SCENE_THRESHOLD,
    min_shot: float = MIN_SHOT,
    method: str = "auto",
    height: int = 144,
    timeout: float = 3600.0,
    cancel: Any = None,
    progress: Callable[[float], None] | None = None,
) -> list[Shot]:
    """Split a video (or ``[start, end]`` of it) into shots at camera cuts.

    Args:
        method: ``"ffmpeg"`` (scene score), ``"pyscenedetect"`` (needs the
            ``scenedetect`` package) or ``"auto"`` (PySceneDetect when installed).
        threshold: ffmpeg scene score that counts as a cut (0..1).
        min_shot: Cuts closer than this to the previous one are dropped.
        height: Frames are scaled to this height before scoring (speed).
        cancel: Anything with ``is_set()`` (a ``threading.Event``); when it is
            set the ffmpeg pass stops and :class:`ShotDetectionCancelled` is raised.
        progress: Called about twice a second with the position (seconds of
            the video) the ffmpeg pass has reached.
    """
    src = os.fspath(path)
    use_psd = method == "pyscenedetect"
    if method == "auto":
        try:
            import scenedetect  # type: ignore[import-not-found]  # noqa: F401

            use_psd = True
        except ImportError:
            use_psd = False
    cuts = (
        _pyscenedetect_cuts(src, start, end)
        if use_psd
        else _ffmpeg_cuts(src, start, end, threshold, height, timeout, cancel, progress)
    )
    lo = float(start or 0.0)
    hi = end
    if hi is None:
        hi = probe_video(src).duration
    if hi is None:
        hi = (max(cuts) + 1.0) if cuts else lo + 1.0
    bounds = [lo]
    for t in sorted(cuts):
        if lo < t < hi and t - bounds[-1] >= min_shot:
            bounds.append(round(t, 3))
    ends = [*bounds[1:], round(float(hi), 3)]
    return [Shot(a, b) for a, b in zip(bounds, ends, strict=True) if b > a]


def snap_to_shots(t: float, shots: Sequence[Shot], *, max_move: float = 3.0) -> float:
    """Move *t* to the nearest shot boundary within *max_move* seconds."""
    best, dist = t, max_move
    for s in shots:
        for b in (s.start, s.end):
            if abs(b - t) <= dist:
                best, dist = b, abs(b - t)
    return best


def video_clip(
    path: str | os.PathLike[str],
    start: float,
    end: float,
    *,
    height: int = 360,
    fps: float = 2.0,
    audio: bool = True,
    max_bytes: int = 18 * 1024 * 1024,
    timeout: float = 600.0,
) -> VideoContent:
    """A small MP4 of ``[start, end]`` for models that take video (Gemini).

    The clip is re-encoded small (H.264, low frame rate, mono AAC) so it fits
    inline in a request. Use it as a ``{"type": "video", "source": clip}``
    content block.
    """
    if end <= start:
        raise ValueError("end must be after start")
    ffmpeg = tool_path("ffmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.mp4"
        argv = [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{start:.3f}",
            "-i",
            os.fspath(path),
            "-t",
            f"{end - start:.3f}",
            "-map",
            "0:v:0",
            "-vf",
            f"scale=-2:{int(height)},fps={fps}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-pix_fmt",
            "yuv420p",
        ]
        argv += ["-map", "0:a:0?", "-c:a", "aac", "-ac", "1", "-b:a", "48k"] if audio else ["-an"]
        argv += ["-sn", "-movflags", "+faststart", str(out)]
        run_media_command(argv, timeout=timeout, what="ffmpeg clip")
        data = out.read_bytes()
    if len(data) > max_bytes:
        raise MediaProcessingError(
            f"clip is {len(data) // 1024} KiB, over the {max_bytes // 1024} KiB limit: use a shorter range"
        )
    return VideoContent(
        data=base64.b64encode(data).decode("ascii"), media_type="video/mp4", duration_seconds=end - start
    )


__all__ = [
    "MIN_SHOT",
    "SCENE_THRESHOLD",
    "ContactSheet",
    "Shot",
    "ShotDetectionCancelled",
    "VideoInfo",
    "contact_sheet",
    "detect_shots",
    "extract_frame",
    "extract_frames",
    "format_clock",
    "probe_video",
    "sample_times",
    "snap_to_shots",
    "video_clip",
]
