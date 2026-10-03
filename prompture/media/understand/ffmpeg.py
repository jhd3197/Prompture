"""ffmpeg / ffprobe helpers for the transcription pipeline.

Every command runs as an argv list (never through a shell) with a timeout and
a scrubbed, UTF-8 child environment. Binaries are located with a real health
probe (:func:`prompture.capabilities.cached_probe`), so a stale shim on
``PATH`` produces a clear "broken" error instead of a cryptic crash halfway
through a job.

The pipeline is:

1. :func:`probe_duration` — ffprobe the source duration (bounded timeout).
2. :func:`transcode_for_speech` — drop video, downmix to mono, resample to
   16 kHz and encode a low-bitrate MP3 (32 kbps ≈ 14 MB per hour).
3. :func:`split_audio` — cut into ``chunk_seconds`` segments, each far below
   the 25 MB upload limit of Whisper-style APIs.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from ...capabilities.probe import ProbeResult, cached_probe, probe_env
from ...security.redaction import scrub_secrets
from .errors import MediaProcessingError, MediaToolMissingError

#: Probe arguments per binary (ffmpeg tools use ``-version``).
PROBE_ARGS: dict[str, tuple[str, ...]] = {
    "ffmpeg": ("-version",),
    "ffprobe": ("-version",),
    "yt-dlp": ("--version",),
}

DEFAULT_FFPROBE_TIMEOUT = 30.0
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BITRATE = "32k"


def probe_tool(name: str) -> ProbeResult:
    """Health-probe *name* (cached for a minute so a batch of jobs shares one probe)."""
    return cached_probe(name, PROBE_ARGS.get(name, ("--version",)))


def tool_path(name: str) -> str:
    """Return the resolved path of a healthy *name* binary.

    Raises:
        MediaToolMissingError: The binary is missing, broken, or does not answer.
    """
    probe = probe_tool(name)
    if not probe.ok or not probe.path:
        raise MediaToolMissingError(name, probe.status, probe.hint)
    return probe.path


def tool_available(name: str) -> bool:
    """``True`` when *name* passes its health probe."""
    return probe_tool(name).ok


def run_media_command(argv: Sequence[str], *, timeout: float, what: str) -> subprocess.CompletedProcess[bytes]:
    """Run *argv* without a shell; raise :class:`MediaProcessingError` on failure.

    Args:
        argv: Full argument vector; ``argv[0]`` must be a resolved binary path.
        timeout: Seconds before the process is killed.
        what: Short label for error messages (``"ffmpeg transcode"``).
    """
    try:
        proc = subprocess.run(  # nosec B603 - argv list, no shell
            list(argv),
            capture_output=True,
            timeout=timeout,
            env=probe_env(),
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaProcessingError(f"{what} timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise MediaProcessingError(f"{what} could not start: {exc}") from exc
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        tail = scrub_secrets(err[-600:]) if err else f"exit code {proc.returncode}"
        raise MediaProcessingError(f"{what} failed: {tail}")
    return proc


def probe_duration(path: str | os.PathLike[str], *, timeout: float = DEFAULT_FFPROBE_TIMEOUT) -> float | None:
    """Return the media duration in seconds, or ``None`` when ffprobe cannot tell.

    Raises:
        MediaToolMissingError: ffprobe is missing or broken.
        MediaProcessingError: ffprobe failed or exceeded *timeout*.
    """
    ffprobe = tool_path("ffprobe")
    proc = run_media_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            os.fspath(path),
        ],
        timeout=timeout,
        what="ffprobe",
    )
    out = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    for line in out.splitlines():
        try:
            value = float(line.strip())
        except ValueError:
            continue
        if value > 0:
            return value
    return None


def transcode_for_speech(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bitrate: str = DEFAULT_BITRATE,
    timeout: float = 900.0,
) -> Path:
    """Convert *src* (any audio/video) to mono, low-bitrate MP3 at *dst*."""
    ffmpeg = tool_path("ffmpeg")
    run_media_command(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            os.fspath(src),
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(int(sample_rate)),
            "-c:a",
            "libmp3lame",
            "-b:a",
            str(bitrate),
            os.fspath(dst),
        ],
        timeout=timeout,
        what="ffmpeg transcode",
    )
    out = Path(dst)
    if not out.is_file() or out.stat().st_size == 0:
        raise MediaProcessingError("ffmpeg transcode produced no audio (does the source have an audio track?)")
    return out


def split_audio(
    src: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    chunk_seconds: float,
    timeout: float = 900.0,
    prefix: str = "chunk",
) -> list[Path]:
    """Split an MP3 into ``chunk_seconds`` pieces (stream copy, no re-encode).

    Returns the chunk paths in playback order.
    """
    ffmpeg = tool_path("ffmpeg")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = out / f"{prefix}_%04d.mp3"
    run_media_command(
        [
            ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            os.fspath(src),
            "-f",
            "segment",
            "-segment_time",
            f"{float(chunk_seconds):g}",
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            os.fspath(pattern),
        ],
        timeout=timeout,
        what="ffmpeg split",
    )
    chunks = sorted(p for p in out.glob(f"{prefix}_*.mp3") if p.is_file() and p.stat().st_size > 0)
    if not chunks:
        raise MediaProcessingError("ffmpeg split produced no chunks")
    return chunks


def chunk_offsets(
    chunks: Sequence[str | os.PathLike[str]],
    *,
    chunk_seconds: float,
    timeout: float = DEFAULT_FFPROBE_TIMEOUT,
) -> list[tuple[float, float | None]]:
    """Return ``(start_offset, duration)`` per chunk.

    Offsets accumulate the probed chunk durations (segment boundaries land on
    frame edges, not exactly on ``chunk_seconds``); a chunk ffprobe cannot
    measure is assumed to be ``chunk_seconds`` long.
    """
    result: list[tuple[float, float | None]] = []
    offset = 0.0
    for chunk in chunks:
        try:
            duration = probe_duration(chunk, timeout=timeout)
        except MediaProcessingError:
            duration = None
        result.append((offset, duration))
        offset += duration if duration else float(chunk_seconds)
    return result
