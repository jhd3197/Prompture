"""URL / file → timestamped transcript.

:func:`transcribe` resolves the source (local file, direct media URL, or a
video/podcast page through ``yt-dlp``), normalizes the audio with ffmpeg
(mono, 16 kHz, low-bitrate MP3), splits it into ~10-minute chunks below the
25 MB upload limit, transcribes each chunk with a speech-to-text provider and
stitches the segments back together with offset timestamps.

Provider choice (``model="auto"``) is the first configured provider in the
order Groq Whisper → OpenAI Whisper → ElevenLabs Scribe (reorder with
``PROMPTURE_STT_PROVIDERS="openai,groq"``). Audio is only ever sent to that
one provider unless the caller passes ``allow_provider_fallback=True``; with
the opt-in, failures go through :class:`~prompture.capabilities.BackendChain`
(auth / quota / rate limit → next provider, transient → one retry).

Every STT call is recorded on the usage tracker (and therefore the SQLite
usage ledger) as a ``stt`` media event, and the per-call cost is summed into
:attr:`Transcript.cost`.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...capabilities.backends import BackendChain, BaseBackend
from ...capabilities.errors import AllBackendsFailedError
from ...security.redaction import scrub_secrets, scrub_url_credentials
from . import ffmpeg as _ff
from . import sources as _sources
from .errors import (
    MediaProcessingError,
    MediaTooLargeError,
    MediaToolMissingError,
    TranscriptionError,
    TranscriptionUnavailableError,
)

# ── Limits ─────────────────────────────────────────────────────────────────

#: Largest source file / download accepted (bytes).
DEFAULT_MAX_SOURCE_BYTES = 500 * 1024 * 1024
#: Most chunks per job (24 × 10 min ≈ 4 hours).
DEFAULT_MAX_CHUNKS = 24
#: Default chunk length in seconds.
DEFAULT_CHUNK_SECONDS = 600
#: Largest single upload (Whisper-style APIs reject > 25 MB).
DEFAULT_MAX_CHUNK_BYTES = 24 * 1024 * 1024
#: Cap on the sum of all chunk uploads.
DEFAULT_MAX_TOTAL_CHUNK_BYTES = 200 * 1024 * 1024
#: Timeout for each external command (yt-dlp download, ffmpeg transcode/split).
DEFAULT_TIMEOUT = 900.0
#: Timeout for each ffprobe call.
DEFAULT_FFPROBE_TIMEOUT = 30.0

#: Chunks shorter than this (a trailing sliver after splitting) are skipped.
MIN_CHUNK_SECONDS = 0.5

#: Formats every built-in STT provider accepts as-is (no ffmpeg needed).
DIRECT_UPLOAD_EXTENSIONS = frozenset({".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"})

#: Env var that reorders the providers ``auto`` considers.
STT_ORDER_ENV = "PROMPTURE_STT_PROVIDERS"


# ── Result types ───────────────────────────────────────────────────────────


def format_timestamp(seconds: float | None) -> str:
    """Format *seconds* as ``HH:MM:SS``."""
    total = max(0, int(seconds or 0))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def parse_timestamp(value: Any) -> float | None:
    """Parse ``HH:MM:SS`` / ``MM:SS`` / seconds into float seconds (``None`` if unparseable)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value >= 0 else None
    text = str(value).strip().strip("[]()")
    if not text:
        return None
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    if len(parts) > 3 or any(p < 0 for p in parts):
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


@dataclass
class TranscriptSegment:
    """A timed span of speech; ``start``/``end`` are seconds from the start of the source."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


@dataclass
class Transcript:
    """A stitched transcript of a whole source.

    Attributes:
        text: Full transcript text.
        segments: Timed segments with offsets relative to the source start.
        source: The source (URL credentials scrubbed) or file path.
        model: ``provider/model`` of the provider that served the first chunk.
        provider: Provider name (``groq`` / ``openai`` / ``elevenlabs`` / ...).
        duration_s: Audio duration in seconds (probed or reported).
        language: Detected or requested language code.
        cost: Total STT cost in USD across all chunks.
        usage: ``requests``, ``chunks``, ``audio_seconds``, ``by_provider``.
        meta: ``source_kind``, ``title``, ``chunked``, ``fallback``, ``route``.
    """

    text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    source: str = ""
    model: str = ""
    provider: str = ""
    duration_s: float | None = None
    language: str | None = None
    cost: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str | None:
        return self.meta.get("title")

    def to_markdown(self, timestamps: bool = True, *, header: bool = True) -> str:
        """Render as Markdown: a short header, then one ``[HH:MM:SS] text`` line per segment."""
        lines: list[str] = []
        if header:
            lines.append(f"# Transcript: {self.title or self.source or 'media'}")
            lines.append("")
            facts = [f"Source: {self.source}"] if self.source else []
            if self.model:
                facts.append(f"Model: {self.model}")
            if self.duration_s:
                facts.append(f"Duration: {format_timestamp(self.duration_s)}")
            if self.language:
                facts.append(f"Language: {self.language}")
            if facts:
                lines.extend(f"- {f}" for f in facts)
                lines.append("")
        if timestamps and self.segments:
            lines.extend(f"[{format_timestamp(s.start)}] {s.text.strip()}" for s in self.segments if s.text.strip())
        else:
            lines.append(self.text.strip())
        return "\n".join(lines).rstrip() + "\n"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["segments"] = [s.to_dict() for s in self.segments]
        return data


# ── STT providers ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class STTProviderSpec:
    """A speech-to-text provider ``auto`` may pick."""

    name: str
    env_var: str
    settings_attr: str
    default_model: str
    display: str


#: Default ``auto`` order: fastest/cheapest Whisper first.
STT_PROVIDERS: tuple[STTProviderSpec, ...] = (
    STTProviderSpec("groq", "GROQ_API_KEY", "groq_api_key", "whisper-large-v3-turbo", "Groq Whisper"),
    STTProviderSpec("openai", "OPENAI_API_KEY", "openai_api_key", "whisper-1", "OpenAI Whisper"),
    STTProviderSpec("elevenlabs", "ELEVENLABS_API_KEY", "elevenlabs_api_key", "scribe_v1", "ElevenLabs Scribe"),
)
_SPECS = {s.name: s for s in STT_PROVIDERS}


def provider_configured(name: str) -> bool:
    """``True`` when the provider's API key is set (env var or settings). Offline."""
    spec = _SPECS.get(name)
    if spec is None:
        return False
    if os.environ.get(spec.env_var, "").strip():
        return True
    try:
        from ...infra.settings import settings

        return bool(getattr(settings, spec.settings_attr, None))
    except Exception:  # pragma: no cover - settings import never fails in practice
        return False


def stt_setup_hint() -> str:
    """Exact next step when no STT provider is configured."""
    return "Set GROQ_API_KEY (Groq Whisper, cheapest), OPENAI_API_KEY or ELEVENLABS_API_KEY to enable transcription."


def make_stt_driver(provider: str, model: str) -> Any:
    """Instantiate the STT driver for ``provider/model`` (patched in tests)."""
    from ...drivers.audio_registry import get_stt_driver_for_model

    if provider == "groq":
        from .groq_stt import ensure_groq_stt_registered

        ensure_groq_stt_registered()
    return get_stt_driver_for_model(f"{provider}/{model}")


def _language_options(provider: str, language: str | None) -> dict[str, Any]:
    if not language:
        return {}
    if provider == "elevenlabs":
        return {"language_code": language}
    return {"language": language}


def _record_stt_usage(driver: Any, meta: dict[str, Any], elapsed_ms: float, error: Exception | None = None) -> None:
    from ...drivers._media_usage import record_media_usage

    record_media_usage(
        driver,
        meta,
        elapsed_ms,
        modality="stt",
        count_key="duration_seconds",
        status="error" if error else "success",
        error=error,
    )


class STTBackend(BaseBackend):
    """One STT provider wrapped as a :class:`~prompture.capabilities.BaseBackend`.

    ``run(audio, filename=..., language=...)`` returns the driver response
    (``text``, ``segments``, ``language``, ``meta``) and records usage.
    """

    category = "media"

    def __init__(self, provider: str, model: str | None = None) -> None:
        self.provider = provider
        spec = _SPECS.get(provider)
        self.spec = spec
        self.name = provider
        self.model = model or (spec.default_model if spec else "")
        self.requires = (spec.env_var,) if spec else ()

    @property
    def model_name(self) -> str:
        return f"{self.provider}/{self.model}" if self.model else self.provider

    def available(self) -> bool:
        if self.spec is not None:
            return provider_configured(self.provider)
        try:
            from ...drivers.registry import is_stt_driver_registered

            return is_stt_driver_registered(self.provider)
        except Exception:
            return False

    def unavailable_hint(self) -> str | None:
        if self.spec is not None:
            return f"Set {self.spec.env_var} to use {self.spec.display}."
        return f"No STT driver is registered for provider {self.provider!r}."

    def run(self, audio: bytes, *, filename: str = "audio.mp3", language: str | None = None) -> dict[str, Any]:
        driver = make_stt_driver(self.provider, self.model)
        options: dict[str, Any] = {"filename": filename, **_language_options(self.provider, language)}
        start = time.perf_counter()
        hook = getattr(driver, "transcribe_with_hooks", None)
        try:
            resp = hook(audio, options) if callable(hook) else driver.transcribe(audio, options)
        except Exception as exc:
            _record_stt_usage(driver, {"model_name": self.model_name}, (time.perf_counter() - start) * 1000, exc)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        resp = dict(resp or {})
        meta = dict(resp.get("meta") or {})
        meta.setdefault("model_name", self.model_name)
        resp["meta"] = meta
        _record_stt_usage(driver, meta, elapsed_ms)
        return resp

    def live_check(self) -> None:
        """Transcribe half a second of silence (a fraction of a cent)."""
        self.run(silent_wav(0.5), filename="silence.wav")


def silent_wav(seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    """A tiny mono 16-bit PCM WAV of silence (for live health checks)."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return buf.getvalue()


def _parse_model(model: str | None) -> tuple[str | None, str | None]:
    """``"auto"`` → (None, None); ``"groq"`` → ("groq", None); ``"openai/whisper-1"`` → both."""
    text = (model or "auto").strip()
    if not text or text.lower() == "auto":
        return None, None
    provider, _, model_id = text.partition("/")
    return provider.lower(), (model_id or None)


def stt_backends() -> list[STTBackend]:
    """All built-in STT backends in effective ``auto`` order (env override applied)."""
    chain = BackendChain([STTBackend(s.name) for s in STT_PROVIDERS], override_env=STT_ORDER_ENV, name="transcription")
    return chain.ordered()


def stt_chain() -> BackendChain[dict[str, Any]]:
    """The built-in STT providers as a :class:`BackendChain` (``auto`` order)."""
    return BackendChain([STTBackend(s.name) for s in STT_PROVIDERS], override_env=STT_ORDER_ENV, name="transcription")


def active_stt_provider() -> STTBackend | None:
    """The provider ``model="auto"`` would use right now (offline), or ``None``."""
    return stt_chain().active_backend()


def transcription_available() -> bool:
    """``True`` when an STT provider is configured. Offline.

    That is enough for small audio files in a provider-supported format;
    long or video sources additionally need ffmpeg (see health).
    """
    return active_stt_provider() is not None


def plan_stt_backends(model: str | None = "auto", *, allow_provider_fallback: bool = False) -> list[STTBackend]:
    """Backends a transcription may use, in order.

    Without ``allow_provider_fallback`` the plan has exactly one backend, so
    audio never reaches a second provider.

    Raises:
        TranscriptionUnavailableError: ``auto`` and nothing is configured.
    """
    provider, model_id = _parse_model(model)
    ordered = stt_backends()
    if provider is None:
        configured = [b for b in ordered if b.available()]
        if not configured:
            raise TranscriptionUnavailableError(f"no speech-to-text provider is configured. {stt_setup_hint()}")
        primary: STTBackend = configured[0]
    else:
        primary = STTBackend(provider, model_id)
    if not allow_provider_fallback:
        return [primary]
    rest = [b for b in ordered if b.name != primary.name and b.available()]
    return [primary, *rest]


# ── Pipeline ───────────────────────────────────────────────────────────────


@dataclass
class _Chunk:
    path: Path
    offset: float
    duration: float | None


def _prepare_chunks(
    media: _sources.LocalMedia,
    workdir: Path,
    *,
    chunk_seconds: float,
    max_chunks: int,
    max_chunk_bytes: int,
    max_total_chunk_bytes: int,
    timeout: float,
    ffprobe_timeout: float,
) -> tuple[list[_Chunk], float | None, bool]:
    """Turn a local media file into upload-ready chunks.

    Returns ``(chunks, duration, used_ffmpeg)``.
    """
    small = media.size <= max_chunk_bytes and media.extension in DIRECT_UPLOAD_EXTENSIONS
    if small and not _ff.tool_available("ffmpeg"):
        # A provider-supported file under the upload limit needs no ffmpeg.
        return [_Chunk(media.path, 0.0, None)], None, False

    try:
        duration = _ff.probe_duration(media.path, timeout=ffprobe_timeout)
    except (MediaProcessingError, MediaToolMissingError):
        if small:
            return [_Chunk(media.path, 0.0, None)], None, False
        raise
    if duration is not None and math.ceil(duration / chunk_seconds) > max_chunks:
        limit = max_chunks * chunk_seconds
        raise MediaTooLargeError(
            f"source is {format_timestamp(duration)} long; the limit is {format_timestamp(limit)} "
            f"({max_chunks} chunks × {chunk_seconds:g}s)",
            limit=limit,
        )
    if small and (duration is None or duration <= chunk_seconds):
        return [_Chunk(media.path, 0.0, duration)], duration, False

    speech = _ff.transcode_for_speech(media.path, workdir / "speech.mp3", timeout=timeout)
    if duration is not None and duration <= chunk_seconds and speech.stat().st_size <= max_chunk_bytes:
        paths = [speech]
    else:
        paths = _ff.split_audio(speech, workdir / "chunks", chunk_seconds=chunk_seconds, timeout=timeout)
    if len(paths) > max_chunks:
        raise MediaTooLargeError(f"{len(paths)} chunks exceeds the {max_chunks}-chunk cap", limit=max_chunks)
    total = 0
    for p in paths:
        size = p.stat().st_size
        if size > max_chunk_bytes:
            raise MediaTooLargeError(
                f"chunk {p.name} is {size} bytes, over the {max_chunk_bytes}-byte upload cap", limit=max_chunk_bytes
            )
        total += size
    if total > max_total_chunk_bytes:
        raise MediaTooLargeError(
            f"chunks total {total} bytes, over the {max_total_chunk_bytes}-byte cap", limit=max_total_chunk_bytes
        )
    if len(paths) == 1:
        return [_Chunk(paths[0], 0.0, duration)], duration, True
    offsets = _ff.chunk_offsets(paths, chunk_seconds=chunk_seconds, timeout=ffprobe_timeout)
    chunks = [_Chunk(p, off, dur) for p, (off, dur) in zip(paths, offsets, strict=False)]
    # A sliver at the end (segment boundaries land on frame edges) is rejected by
    # providers as "too short" and carries no speech; drop it.
    kept = [c for c in chunks if c.duration is None or c.duration >= MIN_CHUNK_SECONDS]
    return kept or chunks[:1], duration, True


def _segments_from_words(words: list[Any], max_span: float = 20.0) -> list[dict[str, Any]]:
    """Group word timings (ElevenLabs-style ``words``) into sentence-ish segments."""
    segments: list[dict[str, Any]] = []
    buf: list[str] = []
    seg_start: float | None = None
    seg_end = 0.0
    for w in words:
        if not isinstance(w, dict):
            continue
        if w.get("type") not in (None, "word", "spacing"):
            continue
        text = str(w.get("text", ""))
        start = w.get("start")
        end = w.get("end")
        if w.get("type") == "spacing":
            if buf:
                buf.append(text)
            continue
        if seg_start is None and isinstance(start, (int, float)):
            seg_start = float(start)
        buf.append(text)
        if isinstance(end, (int, float)):
            seg_end = float(end)
        stripped = text.strip()
        span = seg_end - (seg_start or 0.0)
        if stripped.endswith((".", "?", "!")) or span >= max_span:
            segments.append({"start": seg_start or 0.0, "end": seg_end, "text": "".join(buf).strip()})
            buf, seg_start = [], None
    if buf and "".join(buf).strip():
        segments.append({"start": seg_start or 0.0, "end": seg_end, "text": "".join(buf).strip()})
    return segments


def _chunk_segments(resp: dict[str, Any], chunk: _Chunk) -> list[TranscriptSegment]:
    raw_segments = resp.get("segments") or []
    if not raw_segments:
        raw = (resp.get("meta") or {}).get("raw_response") or {}
        words = raw.get("words") if isinstance(raw, dict) else None
        if isinstance(words, list) and words:
            raw_segments = _segments_from_words(words)
    out: list[TranscriptSegment] = []
    for s in raw_segments:
        get = s.get if isinstance(s, dict) else (lambda k, d=None, _s=s: getattr(_s, k, d))
        text = str(get("text", "") or "").strip()
        if not text:
            continue
        start = float(get("start", 0.0) or 0.0)
        end = float(get("end", start) or start)
        out.append(TranscriptSegment(chunk.offset + start, chunk.offset + max(end, start), text))
    if not out:
        text = str(resp.get("text") or "").strip()
        if text:
            meta_dur = (resp.get("meta") or {}).get("duration_seconds") or 0
            dur = chunk.duration or float(meta_dur) or 0.0
            out.append(TranscriptSegment(chunk.offset, chunk.offset + dur, text))
    return out


def _display_source(source: Any) -> str:
    text = os.fspath(source) if isinstance(source, os.PathLike) else str(source)
    return scrub_url_credentials(text) if _sources.is_url(text) else text


def transcribe(
    source: str | os.PathLike[str],
    *,
    model: str = "auto",
    allow_provider_fallback: bool = False,
    language: str | None = None,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
    max_total_chunk_bytes: int = DEFAULT_MAX_TOTAL_CHUNK_BYTES,
    timeout: float = DEFAULT_TIMEOUT,
    ffprobe_timeout: float = DEFAULT_FFPROBE_TIMEOUT,
    allow_private: bool | None = None,
) -> Transcript:
    """Transcribe a local file or URL into a timestamped :class:`Transcript`.

    Args:
        source: Local path, direct media URL, or a video/podcast page URL
            (needs a healthy ``yt-dlp``).
        model: ``"auto"``, a provider (``"groq"``) or ``"provider/model"``
            (``"openai/whisper-1"``).
        allow_provider_fallback: Permit sending the audio to another
            configured provider when the first one fails. Off by default.
        language: ISO language hint (``"en"``); auto-detected when omitted.
        max_source_bytes: Cap on the local file / download size.
        max_chunks: Cap on chunk count (duration cap = ``max_chunks × chunk_seconds``).
        chunk_seconds: Chunk length for long audio.
        max_chunk_bytes: Per-upload cap (Whisper rejects > 25 MB).
        max_total_chunk_bytes: Cap on the sum of all uploads.
        timeout: Timeout for each yt-dlp / ffmpeg command, in seconds.
        ffprobe_timeout: Timeout for each ffprobe call.
        allow_private: Permit private-network URLs (default:
            ``PROMPTURE_WEB_ALLOW_PRIVATE``).

    Raises:
        TranscriptionUnavailableError: No STT provider is configured.
        UnsafeURLError: The URL points at a private / non-http target.
        MediaToolMissingError: ffmpeg / ffprobe / yt-dlp is needed but missing or broken.
        MediaTooLargeError: A size, duration or chunk cap was exceeded.
        MediaSourceError: The source is missing or is not media.
        TranscriptionError: Every allowed provider failed.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    plan = plan_stt_backends(model, allow_provider_fallback=allow_provider_fallback)
    display = _display_source(source)
    workdir = Path(tempfile.mkdtemp(prefix="prompture-media-"))
    try:
        media = _sources.fetch_source(
            source,
            workdir,
            max_bytes=max_source_bytes,
            timeout=timeout,
            allow_private=allow_private,
        )
        chunks, duration, used_ffmpeg = _prepare_chunks(
            media,
            workdir,
            chunk_seconds=chunk_seconds,
            max_chunks=max_chunks,
            max_chunk_bytes=max_chunk_bytes,
            max_total_chunk_bytes=max_total_chunk_bytes,
            timeout=timeout,
            ffprobe_timeout=ffprobe_timeout,
        )
        return _transcribe_chunks(
            chunks,
            plan,
            language=language,
            source=display,
            media=media,
            duration=duration,
            used_ffmpeg=used_ffmpeg,
            allow_provider_fallback=allow_provider_fallback,
            chunk_seconds=chunk_seconds,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _transcribe_chunks(
    chunks: list[_Chunk],
    plan: list[STTBackend],
    *,
    language: str | None,
    source: str,
    media: _sources.LocalMedia,
    duration: float | None,
    used_ffmpeg: bool,
    allow_provider_fallback: bool,
    chunk_seconds: float,
) -> Transcript:
    chain: BackendChain[dict[str, Any]] = BackendChain(plan, name="transcription")
    order = [b.name for b in plan]
    by_name = {b.name: b for b in plan}
    segments: list[TranscriptSegment] = []
    texts: list[str] = []
    route: list[dict[str, Any]] = []
    by_provider: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    reported_seconds = 0.0
    detected_language: str | None = None
    first_model = ""
    first_provider = ""
    any_fallback = False

    for index, chunk in enumerate(chunks):
        audio = chunk.path.read_bytes()
        try:
            result = chain.run(audio, only=order, filename=chunk.path.name, language=language)
        except AllBackendsFailedError as exc:
            raise TranscriptionError(
                _failure_message(exc, plan, allow_provider_fallback), attempts=exc.attempts
            ) from exc
        except Exception as exc:
            attempts = list(getattr(exc, "attempts", []) or [])
            raise TranscriptionError(
                f"transcription failed on chunk {index + 1}/{len(chunks)}: {type(exc).__name__}: {scrub_secrets(str(exc))}",
                attempts=attempts,
            ) from exc

        resp = result.value
        served = result.served_by
        if result.fallback:
            any_fallback = True
        # Sticky failover: once a provider had to step in, start there next time.
        if order and order[0] != served:
            order = [served, *[n for n in order if n != served]]

        meta = resp.get("meta") or {}
        cost = float(meta.get("cost") or 0.0)
        secs = float(meta.get("duration_seconds") or 0.0) or float(chunk.duration or 0.0)
        total_cost += cost
        reported_seconds += secs
        model_name = str(meta.get("model_name") or by_name[served].model_name)
        if not first_model:
            first_model, first_provider = model_name, served
        if detected_language is None and resp.get("language"):
            detected_language = str(resp["language"])
        stats = by_provider.setdefault(served, {"requests": 0, "cost": 0.0, "audio_seconds": 0.0, "model": model_name})
        stats["requests"] += 1
        stats["cost"] = round(stats["cost"] + cost, 6)
        stats["audio_seconds"] = round(stats["audio_seconds"] + secs, 3)

        chunk_segments = _chunk_segments(resp, chunk)
        segments.extend(chunk_segments)
        text = str(resp.get("text") or "").strip() or " ".join(s.text for s in chunk_segments)
        if text:
            texts.append(text)
        route.append({"chunk": index, "offset": round(chunk.offset, 3), **result.route})

    total_duration = duration
    if total_duration is None:
        total_duration = segments[-1].end if segments and segments[-1].end > 0 else (reported_seconds or None)

    return Transcript(
        text="\n".join(texts).strip(),
        segments=segments,
        source=source,
        model=first_model,
        provider=first_provider,
        duration_s=round(total_duration, 3) if total_duration else None,
        language=detected_language or language,
        cost=round(total_cost, 6),
        usage={
            "requests": len(chunks),
            "chunks": len(chunks),
            "audio_seconds": round(reported_seconds, 3),
            "cost": round(total_cost, 6),
            "by_provider": by_provider,
        },
        meta={
            "source_kind": media.kind,
            "title": media.title,
            "source_bytes": media.size,
            "chunked": len(chunks) > 1,
            "chunk_seconds": chunk_seconds,
            "ffmpeg": used_ffmpeg,
            "fallback": any_fallback,
            "allow_provider_fallback": allow_provider_fallback,
            "route": route,
        },
    )


def _failure_message(exc: AllBackendsFailedError, plan: list[STTBackend], allow_fallback: bool) -> str:
    msg = str(exc)
    if not allow_fallback:
        others = [b.name for b in stt_backends() if b.name != plan[0].name and b.available()]
        if others:
            msg += (
                f". Other configured providers ({', '.join(others)}) were not tried because "
                "allow_provider_fallback is off; pass allow_provider_fallback=True to let the audio go to them."
            )
    return msg
