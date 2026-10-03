"""Tests for prompture.media.understand (transcription, summarization, tools, CLI, health).

No network and no real STT: a fake STT driver replaces ``make_stt_driver``,
ffmpeg helpers are monkeypatched (one test exercises the real ffmpeg binary
when it is installed), and DNS resolution is stubbed for URL tests.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from prompture.capabilities.errors import UnsafeURLError
from prompture.capabilities.http import SafeResponse
from prompture.capabilities.probe import ProbeResult, probe_command
from prompture.drivers.base import Driver
from prompture.media.understand import (
    MediaSourceError,
    MediaTooLargeError,
    MediaToolMissingError,
    Transcript,
    TranscriptionError,
    TranscriptionUnavailableError,
    TranscriptSegment,
    plan_stt_backends,
    summarize_media,
    summarize_media_tool,
    transcribe,
    transcribe_media_tool,
    transcription_available,
)
from prompture.media.understand import ffmpeg as ff_mod
from prompture.media.understand import health as health_mod
from prompture.media.understand import sources as sources_mod
from prompture.media.understand.summarize import transcript_windows
from prompture.media.understand.tools import summarize_media as summarize_media_fn
from prompture.media.understand.tools import transcribe_media as transcribe_media_fn
from prompture.media.understand.transcribe import format_timestamp, parse_timestamp

# The package re-exports the ``transcribe`` function, which shadows the submodule name.
tr_mod = importlib.import_module("prompture.media.understand.transcribe")

KEY_VARS = ("GROQ_API_KEY", "OPENAI_API_KEY", "ELEVENLABS_API_KEY")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeAuthError(Exception):
    """Looks like a 401 to classify_error."""

    status_code = 401


class FakeSTTDriver:
    """Records calls; behaviour comes from ``FakeSTT.behaviour[provider]``."""

    def __init__(self, hub: FakeSTT, provider: str, model: str) -> None:
        self.hub = hub
        self.provider = provider
        self.model = model

    def transcribe(self, audio: bytes, options: dict[str, Any]) -> dict[str, Any]:
        self.hub.calls.append((self.provider, self.model, len(audio), dict(options)))
        behaviour = self.hub.behaviour.get(self.provider)
        if isinstance(behaviour, BaseException):
            raise behaviour
        if callable(behaviour):
            return behaviour(audio, options)
        n = sum(1 for c in self.hub.calls if c[0] == self.provider)
        return {
            "text": f"{self.provider} chunk {n}",
            "segments": [
                {"start": 0.0, "end": 4.0, "text": f"{self.provider} hello {n}"},
                {"start": 4.0, "end": 9.5, "text": f"{self.provider} world {n}"},
            ],
            "language": "en",
            "meta": {"duration_seconds": 10.0, "cost": 0.01, "model_name": f"{self.provider}/{self.model}"},
        }


class FakeSTT:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, dict[str, Any]]] = []
        self.behaviour: dict[str, Any] = {}
        self.usage: list[dict[str, Any]] = []

    def make(self, provider: str, model: str) -> FakeSTTDriver:
        return FakeSTTDriver(self, provider, model)

    @property
    def providers_called(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from prompture.infra.settings import settings

    for var in KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    for attr in ("groq_api_key", "openai_api_key", "elevenlabs_api_key"):
        monkeypatch.setattr(settings, attr, None, raising=False)
    monkeypatch.delenv("PROMPTURE_STT_PROVIDERS", raising=False)
    monkeypatch.delenv("PROMPTURE_WEB_ALLOW_PRIVATE", raising=False)


@pytest.fixture
def stt(monkeypatch: pytest.MonkeyPatch, no_keys: None) -> FakeSTT:
    hub = FakeSTT()
    monkeypatch.setattr(tr_mod, "make_stt_driver", hub.make)

    def fake_record(driver: Any, meta: dict[str, Any], elapsed_ms: float, error: Exception | None = None) -> None:
        hub.usage.append({"meta": dict(meta), "error": error})

    monkeypatch.setattr(tr_mod, "_record_stt_usage", fake_record)
    return hub


@pytest.fixture
def no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> ProbeResult:
        return ProbeResult("missing", name, hint=f"Install {name}")

    monkeypatch.setattr(ff_mod, "probe_tool", missing)


def _audio_file(tmp_path: Path, name: str = "clip.mp3", size: int = 2048) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\xff\xfb" + b"\x00" * (size - 2))
    return p


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the ffmpeg helpers: duration and chunk layout come from ``state``."""
    state: dict[str, Any] = {
        "duration": 1500.0,
        "chunks": [(0.0, 600.0), (600.0, 600.0), (1200.0, 300.0)],
        "chunk_size": 1024,
        "calls": [],
    }

    monkeypatch.setattr(ff_mod, "tool_available", lambda name: True)

    def probe_duration(path: Any, *, timeout: float = 30.0) -> float | None:
        state["calls"].append(("probe", Path(path).name))
        return state["duration"]

    def transcode(src: Any, dst: Any, **kw: Any) -> Path:
        state["calls"].append(("transcode", Path(src).name))
        Path(dst).write_bytes(b"\x00" * 4096)
        return Path(dst)

    def split(src: Any, out_dir: Any, *, chunk_seconds: float, timeout: float = 900.0, prefix: str = "chunk") -> list:
        state["calls"].append(("split", chunk_seconds))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(len(state["chunks"])):
            p = out / f"{prefix}_{i:04d}.mp3"
            p.write_bytes(b"\x00" * state["chunk_size"])
            paths.append(p)
        state["workdir"] = out.parent
        return paths

    def offsets(chunks: Any, *, chunk_seconds: float, timeout: float = 30.0) -> list:
        return list(state["chunks"])

    monkeypatch.setattr(ff_mod, "probe_duration", probe_duration)
    monkeypatch.setattr(ff_mod, "transcode_for_speech", transcode)
    monkeypatch.setattr(ff_mod, "split_audio", split)
    monkeypatch.setattr(ff_mod, "chunk_offsets", offsets)
    return state


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    from prompture.capabilities import url_safety

    monkeypatch.setattr(url_safety, "resolve_host", lambda host, port=None: ["93.184.216.34"])


# ---------------------------------------------------------------------------
# Timestamps and result types
# ---------------------------------------------------------------------------


def test_format_and_parse_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(3725.9) == "01:02:05"
    assert parse_timestamp("01:02:05") == 3725
    assert parse_timestamp("02:05") == 125
    assert parse_timestamp("[00:00:07]") == 7
    assert parse_timestamp(12.5) == 12.5
    assert parse_timestamp("soon") is None
    assert parse_timestamp(None) is None


def test_transcript_markdown_and_dict() -> None:
    t = Transcript(
        text="hi there\nbye",
        segments=[TranscriptSegment(0, 2, "hi there"), TranscriptSegment(65, 70, "bye")],
        source="https://example.com/a.mp3",
        model="groq/whisper-large-v3-turbo",
        provider="groq",
        duration_s=70,
        language="en",
        cost=0.001,
    )
    md = t.to_markdown()
    assert "[00:00:00] hi there" in md
    assert "[00:01:05] bye" in md
    assert "Model: groq/whisper-large-v3-turbo" in md
    plain = t.to_markdown(timestamps=False, header=False)
    assert plain.strip() == "hi there\nbye"
    d = t.to_dict()
    assert d["segments"][1] == {"start": 65, "end": 70, "text": "bye"}
    json.dumps(d)


# ---------------------------------------------------------------------------
# Provider selection and privacy
# ---------------------------------------------------------------------------


def test_auto_pick_order(stt: FakeSTT, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not transcription_available()
    with pytest.raises(TranscriptionUnavailableError, match="GROQ_API_KEY"):
        plan_stt_backends("auto")

    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    assert [b.name for b in plan_stt_backends()] == ["elevenlabs"]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert [b.name for b in plan_stt_backends()] == ["openai"]
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert [b.name for b in plan_stt_backends()] == ["groq"]
    assert transcription_available()

    with_fallback = plan_stt_backends(allow_provider_fallback=True)
    assert [b.name for b in with_fallback] == ["groq", "openai", "elevenlabs"]

    monkeypatch.setenv("PROMPTURE_STT_PROVIDERS", "openai")
    assert [b.name for b in plan_stt_backends()] == ["openai"]


def test_explicit_model_uses_that_provider(stt: FakeSTT) -> None:
    plan = plan_stt_backends("openai/whisper-1")
    assert [(b.name, b.model) for b in plan] == [("openai", "whisper-1")]
    plan = plan_stt_backends("groq")
    assert plan[0].model == "whisper-large-v3-turbo"


def test_no_second_provider_without_opt_in(
    stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    stt.behaviour["groq"] = FakeAuthError("invalid api key")
    with pytest.raises(TranscriptionError) as info:
        transcribe(_audio_file(tmp_path))
    assert stt.providers_called == ["groq"]
    assert "allow_provider_fallback" in str(info.value)
    assert "openai" in str(info.value)
    assert info.value.attempts and info.value.attempts[0]["backend"] == "groq"


def test_fallback_with_opt_in_is_sticky(
    stt: FakeSTT, fake_pipeline: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    stt.behaviour["groq"] = FakeAuthError("invalid api key")
    t = transcribe(_audio_file(tmp_path, "long.wav"), allow_provider_fallback=True)
    # Chunk 1: groq fails then openai; chunks 2-3 go straight to openai.
    assert stt.providers_called == ["groq", "openai", "openai", "openai"]
    assert t.provider == "openai"
    assert t.meta["fallback"] is True
    assert t.meta["route"][0]["fallback"] is True
    assert t.meta["route"][1]["fallback"] is False
    assert t.usage["by_provider"]["openai"]["requests"] == 3


def test_language_option_per_provider(stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el")
    transcribe(_audio_file(tmp_path), language="es")
    assert stt.calls[0][3]["language_code"] == "es"
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    transcribe(_audio_file(tmp_path), language="es", model="openai")
    assert stt.calls[1][3]["language"] == "es"


# ---------------------------------------------------------------------------
# Pipeline: stitching, caps, cleanup
# ---------------------------------------------------------------------------


def test_chunk_stitching_offsets(
    stt: FakeSTT, fake_pipeline: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    t = transcribe(_audio_file(tmp_path, "talk.mp4", size=50_000))
    assert [c[0] for c in fake_pipeline["calls"]][:3] == ["probe", "transcode", "split"]
    assert len(stt.calls) == 3
    starts = [s.start for s in t.segments]
    assert starts == [0.0, 4.0, 600.0, 604.0, 1200.0, 1204.0]
    assert t.segments[-1].end == pytest.approx(1209.5)
    assert t.text == "groq chunk 1\ngroq chunk 2\ngroq chunk 3"
    assert t.cost == pytest.approx(0.03)
    assert t.duration_s == 1500.0
    assert t.language == "en"
    assert t.model == "groq/whisper-large-v3-turbo"
    assert t.meta["chunked"] is True and t.meta["ffmpeg"] is True
    assert t.usage["chunks"] == 3
    # Usage is recorded once per STT call.
    assert len(stt.usage) == 3 and all(u["error"] is None for u in stt.usage)


def test_small_file_without_ffmpeg_skips_pipeline(
    stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    t = transcribe(_audio_file(tmp_path))
    assert len(stt.calls) == 1
    assert stt.calls[0][3]["filename"] == "clip.mp3"
    assert t.meta["ffmpeg"] is False
    assert t.segments[0].start == 0.0


def test_large_file_without_ffmpeg_is_clear_error(
    stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    src = _audio_file(tmp_path, "video.mkv")
    with pytest.raises(MediaToolMissingError) as info:
        transcribe(src)
    assert info.value.tool == "ffprobe"
    assert "Install" in str(info.value)
    assert stt.calls == []


def test_source_size_cap(stt: FakeSTT, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(MediaTooLargeError, match="source cap"):
        transcribe(_audio_file(tmp_path, size=5000), max_source_bytes=1000)
    assert stt.calls == []


def test_duration_cap(stt: FakeSTT, fake_pipeline: dict[str, Any], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_pipeline["duration"] = 5 * 3600.0
    with pytest.raises(MediaTooLargeError, match="05:00:00"):
        transcribe(_audio_file(tmp_path, "long.mp4"))
    assert [c[0] for c in fake_pipeline["calls"]] == ["probe"]
    assert stt.calls == []


def test_chunk_count_cap(stt: FakeSTT, fake_pipeline: dict[str, Any], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_pipeline["duration"] = None  # unknown duration: the post-split count still applies
    with pytest.raises(MediaTooLargeError, match="3 chunks exceeds the 2-chunk cap"):
        transcribe(_audio_file(tmp_path, "long.mkv"), max_chunks=2)
    assert stt.calls == []


def test_chunk_bytes_caps(stt: FakeSTT, fake_pipeline: dict[str, Any], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_pipeline["chunk_size"] = 4000
    with pytest.raises(MediaTooLargeError, match="upload cap"):
        transcribe(_audio_file(tmp_path, "long.mp4"), max_chunk_bytes=3000)
    with pytest.raises(MediaTooLargeError, match="chunks total"):
        transcribe(_audio_file(tmp_path, "long.mp4"), max_total_chunk_bytes=10_000)
    assert stt.calls == []


def test_temp_files_cleaned_on_error_and_success(
    stt: FakeSTT, fake_pipeline: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    made: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args: Any, **kwargs: Any) -> str:
        path = real_mkdtemp(*args, **kwargs)
        made.append(path)
        return path

    monkeypatch.setattr(tr_mod.tempfile, "mkdtemp", tracking_mkdtemp)
    stt.behaviour["openai"] = FakeAuthError("bad key")
    with pytest.raises(TranscriptionError):
        transcribe(_audio_file(tmp_path, "a.mp4"))
    assert made and not os.path.exists(made[-1])

    stt.behaviour.clear()
    transcribe(_audio_file(tmp_path, "b.mp4"))
    assert len(made) == 2 and not os.path.exists(made[-1])
    # The caller's file is never deleted.
    assert (tmp_path / "b.mp4").exists()


def test_failed_stt_call_records_error_usage(stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    stt.behaviour["openai"] = FakeAuthError("bad key")
    with pytest.raises(TranscriptionError):
        transcribe(_audio_file(tmp_path))
    assert stt.usage and isinstance(stt.usage[0]["error"], FakeAuthError)
    assert stt.usage[0]["meta"]["model_name"] == "openai/whisper-1"


def test_usage_goes_through_media_usage_recorder(
    no_keys: None, no_ffmpeg: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the fixture's stub, STT spend reaches record_media_usage as an ``stt`` event."""
    from prompture.drivers import _media_usage

    hub = FakeSTT()
    monkeypatch.setattr(tr_mod, "make_stt_driver", hub.make)
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(_media_usage, "record_media_usage", lambda *a, **kw: recorded.append({"args": a, **kw}))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    transcribe(_audio_file(tmp_path))
    assert len(recorded) == 1
    assert recorded[0]["modality"] == "stt"
    assert recorded[0]["args"][1]["cost"] == 0.01


def test_elevenlabs_words_become_segments(stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el")
    words = [
        {"text": "Hello", "start": 0.1, "end": 0.5, "type": "word"},
        {"text": " ", "start": 0.5, "end": 0.6, "type": "spacing"},
        {"text": "world.", "start": 0.6, "end": 1.0, "type": "word"},
        {"text": " ", "start": 1.0, "end": 1.1, "type": "spacing"},
        {"text": "Again", "start": 3.0, "end": 3.4, "type": "word"},
    ]
    stt.behaviour["elevenlabs"] = lambda a, o: {
        "text": "Hello world. Again",
        "segments": [],
        "language": "en",
        "meta": {
            "duration_seconds": 0,
            "cost": 0.0,
            "model_name": "elevenlabs/scribe_v1",
            "raw_response": {"words": words},
        },
    }
    t = transcribe(_audio_file(tmp_path))
    assert [(s.start, s.text) for s in t.segments] == [(0.1, "Hello world."), (3.0, "Again")]


# ---------------------------------------------------------------------------
# Sources: URL safety, downloads, yt-dlp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a.mp3",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost/x.mp3",
        "file:///etc/passwd",
    ],
)
def test_private_or_bad_urls_refused(stt: FakeSTT, url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    called: list[str] = []
    monkeypatch.setattr(sources_mod, "ytdlp_download", lambda *a, **k: called.append("yt") or None)
    with pytest.raises(UnsafeURLError):
        transcribe(url)
    assert called == [] and stt.calls == []


def test_direct_url_download(stt: FakeSTT, no_ffmpeg: None, public_dns: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: dict[str, Any] = {}

    def fake_safe_get(url: str, **kw: Any) -> SafeResponse:
        seen.update(kw, url=url)
        return SafeResponse(url=url, status_code=200, headers={"content-type": "audio/mpeg"}, content=b"\xff\xfb" * 100)

    monkeypatch.setattr(sources_mod, "safe_get", fake_safe_get)
    t = transcribe("https://user:secret@cdn.example.com/ep1.mp3?token=abc", max_source_bytes=12345)
    assert seen["max_bytes"] == 12345
    assert t.meta["source_kind"] == "download"
    assert "secret" not in t.source
    assert stt.calls[0][3]["filename"] == "source.mp3"


def test_platform_url_with_broken_ytdlp(stt: FakeSTT, public_dns: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    broken = ProbeResult("broken", "yt-dlp", path="/shims/yt-dlp", hint="reinstall: pip install -U yt-dlp")
    monkeypatch.setattr(ff_mod, "probe_tool", lambda name: broken)
    with pytest.raises(MediaToolMissingError) as info:
        transcribe("https://www.youtube.com/watch?v=abc123")
    assert info.value.tool == "yt-dlp" and info.value.status == "broken"
    assert "pip install -U yt-dlp" in str(info.value)
    assert stt.calls == []


def test_html_page_without_ytdlp_explains(stt: FakeSTT, public_dns: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(ff_mod, "probe_tool", lambda name: ProbeResult("missing", name, hint="Install yt-dlp"))
    monkeypatch.setattr(
        sources_mod,
        "safe_get",
        lambda url, **kw: SafeResponse(url, 200, {"content-type": "text/html"}, b"<html>player</html>"),
    )
    with pytest.raises(MediaSourceError, match="yt-dlp is missing"):
        transcribe("https://podcasts.example.org/show/episode-4")


def test_platform_url_uses_ytdlp_argv(
    stt: FakeSTT, no_ffmpeg: None, public_dns: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(ff_mod, "tool_path", lambda name: f"/bin/{name}")
    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], *, timeout: float, what: str) -> Any:
        captured["argv"] = argv
        out = Path(argv[argv.index("--output") + 1]).parent
        (out / "source.m4a").write_bytes(b"\x00" * 500)

        class P:
            stdout = b"Episode Title\n"

        return P()

    monkeypatch.setattr(ff_mod, "run_media_command", fake_run)
    t = transcribe("https://youtu.be/abc123")
    argv = captured["argv"]
    assert argv[0] == "/bin/yt-dlp"
    assert argv[-2] == "--" and argv[-1].startswith("https://youtu.be/")
    assert "--ignore-config" in argv and "--no-playlist" in argv
    assert t.meta["source_kind"] == "yt-dlp"
    assert t.title == "Episode Title"


def test_local_stale_ytdlp_shim_classified_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A shim pointing at a missing interpreter must probe as ``broken``, not ``ok``."""
    if os.name == "nt":
        shim = tmp_path / "yt-dlp.cmd"
        shim.write_text('@echo off\r\necho No Python at "C:\\Python99\\python.exe" 1>&2\r\nexit /b 103\r\n')
    else:
        shim = tmp_path / "yt-dlp"
        shim.write_text("#!/nonexistent/python99\nprint('x')\n")
        shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    if os.name == "nt":
        monkeypatch.setenv("PATHEXT", ".CMD;.EXE")
    result = probe_command("yt-dlp", ("--version",))
    assert result.status == "broken"


# ---------------------------------------------------------------------------
# Real ffmpeg (local binary, no network)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not installed")
def test_real_ffmpeg_chunking(stt: FakeSTT, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not (ff_mod.tool_available("ffmpeg") and ff_mod.tool_available("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not healthy")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    src = tmp_path / "tone.wav"
    ff_mod.run_media_command(
        [
            ff_mod.tool_path("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=25",
            "-ac",
            "2",
            os.fspath(src),
        ],
        timeout=60,
        what="make tone",
    )
    t = transcribe(src, chunk_seconds=10)
    assert len(stt.calls) == 3
    offsets = [r["offset"] for r in t.meta["route"]]
    assert offsets[0] == 0.0
    assert offsets[1] == pytest.approx(10.0, abs=0.5)
    assert offsets[2] == pytest.approx(20.0, abs=0.5)
    assert t.duration_s == pytest.approx(25.0, abs=0.5)
    assert t.segments[2].start == pytest.approx(offsets[1], abs=0.01)


# ---------------------------------------------------------------------------
# Summarization (fake LLM driver)
# ---------------------------------------------------------------------------


class FakeLLM(Driver):
    """Answers map prompts with the first timestamp it sees; reduce prompts merge."""

    def __init__(self) -> None:
        self.model = "fake-llm"
        self.prompts: list[str] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        import re

        if "Below are summaries" in prompt:
            stamps = re.findall(r"- \[(\d\d:\d\d:\d\d)\] (.+)", prompt)
            data = {
                "summary": f"merged {len(stamps)} points",
                "key_points": [{"timestamp": ts, "point": p} for ts, p in stamps],
            }
        else:
            first = re.search(r"\[(\d\d:\d\d:\d\d)\] (.+)", prompt)
            data = {
                "summary": "part summary",
                "key_points": [{"timestamp": first.group(1), "point": f"point at {first.group(1)}"}] if first else [],
            }
        return {
            "text": json.dumps(data),
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.002,
                "raw_response": {},
            },
        }


def _long_transcript() -> Transcript:
    segs = [TranscriptSegment(i * 60.0, i * 60.0 + 50, f"Sentence number {i} " + "word " * 30) for i in range(12)]
    return Transcript(text="\n".join(s.text for s in segs), segments=segs, source="talk.mp3", duration_s=720, cost=0.05)


def test_transcript_windows_respect_limit() -> None:
    windows = transcript_windows(_long_transcript(), max_chars=500)
    assert len(windows) > 1
    assert all(len(w) <= 500 for w in windows)
    assert windows[0].startswith("[00:00:00]")


def test_summarize_map_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLM()
    result = summarize_media(_long_transcript(), model="fake/llm", driver=llm, max_chunk_chars=600, max_key_points=20)
    map_calls = [p for p in llm.prompts if "Below are summaries" not in p]
    reduce_calls = [p for p in llm.prompts if "Below are summaries" in p]
    assert len(map_calls) > 1 and len(reduce_calls) >= 1
    assert result.summary.startswith("merged")
    assert result.key_points[0].timestamp == 0.0
    assert all(kp.timestamp is not None for kp in result.key_points)
    assert [kp.timestamp for kp in result.key_points] == sorted(kp.timestamp for kp in result.key_points)
    assert result.usage["llm_calls"] == len(llm.prompts)
    assert result.cost == pytest.approx(0.05 + 0.002 * len(llm.prompts))
    md = result.to_markdown()
    assert "## Key points" in md and "[00:00:00]" in md
    json.dumps(result.to_dict())


def test_summarize_short_transcript_single_call() -> None:
    llm = FakeLLM()
    t = Transcript(text="hello", segments=[TranscriptSegment(5, 8, "hello world")], duration_s=8)
    result = summarize_media(t, model="fake/llm", driver=llm)
    assert len(llm.prompts) == 1
    assert result.key_points[0].timestamp == 5.0


def test_summarize_empty_transcript_no_llm_call() -> None:
    llm = FakeLLM()
    result = summarize_media(Transcript(text=""), model="fake/llm", driver=llm)
    assert llm.prompts == []
    assert "No speech" in result.summary


def test_summarize_map_chunk_cap() -> None:
    with pytest.raises(MediaTooLargeError):
        summarize_media(_long_transcript(), model="fake/llm", driver=FakeLLM(), max_chunk_chars=300, max_map_chunks=2)


def test_summarize_from_source_transcribes_first(
    stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk")
    llm = FakeLLM()
    result = summarize_media(_audio_file(tmp_path), model="fake/llm", driver=llm)
    assert stt.providers_called == ["groq"]
    assert result.transcript is not None and result.transcript.provider == "groq"
    assert result.cost == pytest.approx(0.01 + 0.002)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_tool_definitions() -> None:
    td = transcribe_media_tool()
    assert td.name == "transcribe_media"
    assert td.parameters["required"] == ["source"]
    sd = summarize_media_tool(model="openai/gpt-4o-mini")
    assert sd.name == "summarize_media"
    assert "source" in sd.parameters["properties"] and "focus" in sd.parameters["properties"]
    assert "timestamped key points" in sd.description


def test_transcribe_tool_success_and_truncation(
    stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk")
    out = transcribe_media_fn(str(_audio_file(tmp_path)))
    assert "[00:00:00] groq hello 1" in out
    short = transcribe_media_tool().function(str(_audio_file(tmp_path)), max_chars=20)
    assert "truncated" in short


def test_tools_never_raise_and_scrub(stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch) -> None:
    out = transcribe_media_fn(str(_audio_file(tmp_path)))
    assert out.startswith("Error:") and "GROQ_API_KEY" in out

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    secret = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd"
    stt.behaviour["openai"] = FakeAuthError(f"Incorrect API key provided: {secret}")
    out = transcribe_media_fn(str(_audio_file(tmp_path)))
    assert out.startswith("Error:")
    assert secret not in out

    out = summarize_media_fn("http://127.0.0.1/a.mp3")
    assert out.startswith("Error:") and "unsafe_url" in out
    out = summarize_media_tool("fake/llm").function("missing-file.mp3")
    assert out.startswith("Error:")


def test_summarize_tool_uses_pinned_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from prompture.media.understand import summarize as sum_mod

    seen: dict[str, Any] = {}

    def fake_summarize(source: Any, *, model: str, focus: str | None = None, **kw: Any) -> Any:
        seen.update(model=model, focus=focus)
        return sum_mod.MediaSummary(summary="ok", model=model)

    monkeypatch.setattr(sum_mod, "summarize_media", fake_summarize)
    out = summarize_media_tool("groq/llama-3.3-70b-versatile").function("x.mp3", focus="pricing")
    assert "ok" in out
    assert seen == {"model": "groq/llama-3.3-70b-versatile", "focus": "pricing"}


def test_media_agent_tools_include_understanding() -> None:
    from prompture.media.agent_tools import media_tool_definitions

    names = [t.name for t in media_tool_definitions()]
    assert "transcribe_media" in names and "summarize_media" in names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_transcribe_stdout_and_json(
    stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prompture.cli.transcribe_cmd import COMMANDS
    from prompture.cli.transcribe_cmd import transcribe as transcribe_cmd

    assert [transcribe_cmd] == COMMANDS
    monkeypatch.setenv("GROQ_API_KEY", "gsk")
    src = str(_audio_file(tmp_path))
    runner = CliRunner()
    res = runner.invoke(transcribe_cmd, [src])
    assert res.exit_code == 0, res.output
    assert "[00:00:00] groq hello 1" in res.output

    out = tmp_path / "t.json"
    res = runner.invoke(transcribe_cmd, [src, "--out", str(out), "--model", "groq/whisper-large-v3"])
    assert res.exit_code == 0, res.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["segments"][0]["text"].startswith("groq hello")
    assert stt.calls[-1][1] == "whisper-large-v3"


def test_cli_summary_and_errors(stt: FakeSTT, no_ffmpeg: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from prompture.cli.transcribe_cmd import transcribe as transcribe_cmd
    from prompture.media.understand import summarize as sum_mod

    runner = CliRunner()
    src = str(_audio_file(tmp_path))
    res = runner.invoke(transcribe_cmd, [src])
    assert res.exit_code == 1
    assert "GROQ_API_KEY" in res.output

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = FakeLLM()
    real = sum_mod.summarize_transcript
    monkeypatch.setattr(
        "prompture.media.understand.summarize_transcript",
        lambda t, **kw: real(t, **{**kw, "driver": llm}),
    )
    res = runner.invoke(transcribe_cmd, [src, "--summary", "--summary-model", "fake/llm", "--no-timestamps"])
    assert res.exit_code == 0, res.output
    assert "# Summary" in res.output and "## Key points" in res.output
    assert "openai chunk 1" in res.output

    stt.behaviour["openai"] = FakeAuthError("nope")
    res = runner.invoke(transcribe_cmd, [src, "--allow-provider-fallback"])
    assert res.exit_code == 1 and "TranscriptionError" in res.output


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_rows_registered() -> None:
    from prompture.capabilities.health import list_capabilities

    names = {c.name for c in list_capabilities("media")}
    assert {"transcription", "media_download"} <= names


def test_transcription_health_unconfigured(no_keys: None) -> None:
    row = health_mod.transcription_status(live=False)
    assert row.status == "unconfigured"
    assert "GROQ_API_KEY" in (row.fix_hint or "")


def test_transcription_health_degraded_without_ffmpeg(
    no_keys: None, no_ffmpeg: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    row = health_mod.transcription_status(live=False)
    assert row.status == "degraded"
    assert row.active_backend == "openai/whisper-1"
    assert row.fix_hint == "Install ffmpeg"


def test_transcription_health_ok_and_live(stt: FakeSTT, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ff_mod, "probe_tool", lambda name: ProbeResult("ok", name, path=f"/bin/{name}", output="v1"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk")
    row = health_mod.transcription_status(live=False)
    assert row.status == "ok" and row.active_backend == "groq/whisper-large-v3-turbo"
    assert stt.calls == []  # offline never calls the provider

    row = health_mod.transcription_status(live=True)
    assert row.status == "ok" and "live check passed" in row.message
    assert stt.calls[0][3]["filename"] == "silence.wav"

    stt.behaviour["groq"] = FakeAuthError("bad key")
    row = health_mod.transcription_status(live=True)
    assert row.status == "error" and "GROQ_API_KEY" in (row.fix_hint or "")


def test_media_download_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ff_mod,
        "probe_tool",
        lambda name: ProbeResult("broken", name, path="/shim", output="No Python at 'C:\\Python311'", hint="reinstall"),
    )
    row = health_mod.media_download_status()
    assert row.status == "broken" and row.fix_hint == "reinstall"
    assert "direct audio/video URLs" in row.message

    monkeypatch.setattr(
        ff_mod, "probe_tool", lambda name: ProbeResult("ok", name, path="/bin/yt-dlp", output="2026.09.01")
    )
    row = health_mod.media_download_status()
    assert row.status == "ok" and "2026.09.01" in row.message


# ---------------------------------------------------------------------------
# Live (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_transcription_health() -> None:
    if not transcription_available():
        pytest.skip("no STT provider configured")
    row = health_mod.transcription_status(live=True)
    assert row.status in ("ok", "degraded"), row.message
