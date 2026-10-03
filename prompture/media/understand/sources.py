"""Resolve a transcription source (local path or URL) to a local media file.

* Local paths are used in place (never copied) after a size check.
* Direct media URLs (``.mp3``, ``.m4a``, ``.mp4``, ... or a response whose
  ``Content-Type`` is ``audio/*`` / ``video/*``) are downloaded with
  :func:`prompture.capabilities.safe_get`: public-URL guard on every redirect
  hop, size cap, challenge-page detection.
* Video / podcast pages (YouTube, Vimeo, SoundCloud, ...) are handed to
  ``yt-dlp`` for audio-only extraction, but only when ``yt-dlp`` passes its
  health probe. The URL is validated with the same public-URL guard first and
  passed after ``--`` so it can never be parsed as an option.
"""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from ...capabilities.errors import ResponseTooLargeError
from ...capabilities.http import safe_get
from ...capabilities.url_safety import normalize_public_http_url
from . import ffmpeg as _ff
from .errors import MediaProcessingError, MediaSourceError, MediaTooLargeError

#: Extensions treated as direct media downloads.
MEDIA_EXTENSIONS = frozenset(
    {
        ".mp3",
        ".m4a",
        ".mp4",
        ".m4v",
        ".mpeg",
        ".mpga",
        ".mpg",
        ".wav",
        ".ogg",
        ".oga",
        ".opus",
        ".flac",
        ".aac",
        ".webm",
        ".mkv",
        ".mov",
        ".avi",
        ".wma",
        ".aiff",
        ".aif",
        ".amr",
        ".3gp",
    }
)

#: Hosts whose pages need ``yt-dlp`` (the URL is a player page, not a file).
PLATFORM_HOSTS = (
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "vimeo.com",
    "soundcloud.com",
    "dailymotion.com",
    "twitch.tv",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "bilibili.com",
    "rumble.com",
    "mixcloud.com",
    "bandcamp.com",
    "podcasts.apple.com",
    "ted.com",
    "archive.org",
    "reddit.com",
)

DEFAULT_YTDLP_TIMEOUT = 900.0


@dataclass
class LocalMedia:
    """A media file ready for the ffmpeg / STT pipeline.

    Attributes:
        path: Local file path.
        source: The caller's original source (path or URL, credentials scrubbed by callers).
        kind: ``file`` | ``download`` | ``yt-dlp``.
        size: File size in bytes.
        content_type: Best-known MIME type.
        title: Title reported by the platform (``yt-dlp`` only).
        meta: Extra detail (final URL after redirects, yt-dlp version, ...).
    """

    path: Path
    source: str
    kind: str
    size: int
    content_type: str | None = None
    title: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()


def is_url(source: str) -> bool:
    """``True`` when *source* should be treated as a URL rather than a path."""
    text = str(source).strip()
    return "://" in text or text.lower().startswith(("http:", "https:", "www."))


def _host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def is_platform_url(url: str) -> bool:
    """``True`` for hosts that serve player pages (needs ``yt-dlp``)."""
    host = _host_of(url)
    return any(host == h or host.endswith("." + h) for h in PLATFORM_HOSTS)


def url_extension(url: str) -> str:
    """Lower-cased file extension of the URL path (``""`` when none)."""
    try:
        path = unquote(urlsplit(url).path or "")
    except ValueError:
        return ""
    return Path(path).suffix.lower()


def is_direct_media_url(url: str) -> bool:
    """``True`` when the URL path ends in a known media extension."""
    return url_extension(url) in MEDIA_EXTENSIONS


def _is_media_content_type(ctype: str) -> bool:
    ctype = (ctype or "").lower()
    return ctype.startswith(("audio/", "video/")) or ctype in {
        "application/ogg",
        "application/octet-stream",
        "binary/octet-stream",
    }


def local_file(path: str | os.PathLike[str], *, max_bytes: int) -> LocalMedia:
    """Validate a local media path against *max_bytes*."""
    p = Path(os.path.expanduser(os.fspath(path)))
    if not p.is_file():
        raise MediaSourceError(f"file not found: {p}")
    size = p.stat().st_size
    if size == 0:
        raise MediaSourceError(f"file is empty: {p}")
    if size > max_bytes:
        raise MediaTooLargeError(f"{p.name} is {size} bytes, over the {max_bytes}-byte source cap", limit=max_bytes)
    ctype = mimetypes.guess_type(p.name)[0]
    return LocalMedia(p, os.fspath(path), "file", size, ctype)


def download_direct(
    url: str,
    workdir: str | os.PathLike[str],
    *,
    max_bytes: int,
    timeout: float = 60.0,
    allow_private: bool | None = None,
    require_media: bool = False,
    session: Any = None,
) -> LocalMedia:
    """Download a direct media URL with :func:`safe_get` into *workdir*.

    Args:
        require_media: Raise :class:`MediaSourceError` when the response is not
            ``audio/*`` / ``video/*`` and the URL has no media extension
            (e.g. an HTML page).
    """
    try:
        resp = safe_get(
            url,
            session=session,
            timeout=timeout,
            max_bytes=max_bytes,
            allow_private=allow_private,
            headers={"Accept": "audio/*,video/*,application/octet-stream;q=0.9,*/*;q=0.5"},
            backend="media",
        )
    except ResponseTooLargeError as exc:
        raise MediaTooLargeError(f"download exceeds the {max_bytes}-byte source cap", limit=max_bytes) from exc

    ctype = resp.content_type
    ext = url_extension(resp.url) or url_extension(url)
    if ext not in MEDIA_EXTENSIONS:
        guessed = mimetypes.guess_extension(ctype) if ctype else None
        ext = guessed if guessed in MEDIA_EXTENSIONS else ""
    if require_media and not _is_media_content_type(ctype) and not ext:
        raise MediaSourceError(
            f"{resp.url} is not a media file (content-type {ctype or 'unknown'}); "
            "pages from video/podcast sites need a working yt-dlp"
        )
    if not resp.content:
        raise MediaSourceError(f"{resp.url} returned an empty body")

    out = Path(workdir) / f"source{ext or '.bin'}"
    out.write_bytes(resp.content)
    return LocalMedia(
        out,
        url,
        "download",
        len(resp.content),
        ctype or mimetypes.guess_type(out.name)[0],
        meta={"final_url": resp.url, "redirects": len(resp.history)},
    )


def ytdlp_download(
    url: str,
    workdir: str | os.PathLike[str],
    *,
    max_bytes: int,
    timeout: float = DEFAULT_YTDLP_TIMEOUT,
    allow_private: bool | None = None,
) -> LocalMedia:
    """Extract the best audio stream of a video/podcast page with ``yt-dlp``.

    Raises:
        MediaToolMissingError: yt-dlp is missing or broken (with the fix hint).
        UnsafeURLError: The URL is not a public http(s) address.
        MediaProcessingError: yt-dlp failed or timed out.
        MediaTooLargeError: The audio exceeds *max_bytes*.
    """
    safe_url = normalize_public_http_url(url, allow_private=allow_private)
    ytdlp = _ff.tool_path("yt-dlp")
    out_dir = Path(workdir) / "yt-dlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    template = os.fspath(out_dir / "source.%(ext)s")
    proc = _ff.run_media_command(
        [
            ytdlp,
            "--ignore-config",
            "--no-playlist",
            "--no-progress",
            "--no-warnings",
            "--quiet",
            "--no-mtime",
            "--no-part",
            "--no-cache-dir",
            "--socket-timeout",
            "30",
            "--format",
            "bestaudio/best",
            "--max-filesize",
            str(int(max_bytes)),
            "--no-simulate",
            "--print",
            "after_move:title",
            "--output",
            template,
            "--",
            safe_url,
        ],
        timeout=timeout,
        what="yt-dlp",
    )
    files = sorted(p for p in out_dir.iterdir() if p.is_file() and p.stat().st_size > 0)
    if not files:
        # yt-dlp exits 0 and writes nothing when --max-filesize rejects the file.
        raise MediaTooLargeError(
            f"yt-dlp produced no file (the audio may exceed the {max_bytes}-byte source cap)", limit=max_bytes
        )
    path = max(files, key=lambda p: p.stat().st_size)
    size = path.stat().st_size
    if size > max_bytes:
        raise MediaTooLargeError(f"downloaded audio is {size} bytes, over the {max_bytes}-byte cap", limit=max_bytes)
    title = (proc.stdout or b"").decode("utf-8", errors="replace").strip().splitlines()
    return LocalMedia(
        path,
        url,
        "yt-dlp",
        size,
        mimetypes.guess_type(path.name)[0],
        title=title[-1].strip() if title else None,
        meta={"final_url": safe_url},
    )


def fetch_source(
    source: str | os.PathLike[str],
    workdir: str | os.PathLike[str],
    *,
    max_bytes: int,
    timeout: float = DEFAULT_YTDLP_TIMEOUT,
    download_timeout: float = 60.0,
    allow_private: bool | None = None,
) -> LocalMedia:
    """Resolve *source* to a :class:`LocalMedia` inside *workdir*.

    Routing for URLs:

    1. Known video/podcast platform → ``yt-dlp`` (must be healthy).
    2. URL with a media extension → direct download.
    3. Anything else → direct download; if the response is a page rather
       than media, retry through ``yt-dlp`` when it is healthy, otherwise
       raise a :class:`MediaSourceError` naming the fix.
    """
    if isinstance(source, os.PathLike) or not is_url(str(source)):
        return local_file(source, max_bytes=max_bytes)

    url = str(source).strip()
    if url.lower().startswith("www."):
        url = "https://" + url
    # Validate before routing so a private URL is refused even on the yt-dlp path.
    normalize_public_http_url(url, allow_private=allow_private)

    if is_platform_url(url) and not is_direct_media_url(url):
        return ytdlp_download(url, workdir, max_bytes=max_bytes, timeout=timeout, allow_private=allow_private)
    if is_direct_media_url(url):
        return download_direct(url, workdir, max_bytes=max_bytes, timeout=download_timeout, allow_private=allow_private)
    try:
        return download_direct(
            url,
            workdir,
            max_bytes=max_bytes,
            timeout=download_timeout,
            allow_private=allow_private,
            require_media=True,
        )
    except MediaSourceError as page_error:
        probe = _ff.probe_tool("yt-dlp")
        if not probe.ok:
            raise MediaSourceError(
                f"{page_error} — yt-dlp is {probe.status}" + (f": {probe.hint}" if probe.hint else "")
            ) from page_error
        try:
            return ytdlp_download(url, workdir, max_bytes=max_bytes, timeout=timeout, allow_private=allow_private)
        except MediaProcessingError as exc:
            raise MediaSourceError(f"no media found at {url}: {exc}") from exc
