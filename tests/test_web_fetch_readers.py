"""Tests for web_fetch, read_url readers, search_platform, WebToolkit and health rows.

No network: HTTP goes to a fake ``requests.Session``, DNS is faked, and
binaries / optional packages / transcription are monkeypatched.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from prompture.capabilities import AllBackendsFailedError, UnsafeURLError
from prompture.tools.web import (
    FetchResult,
    ReadResult,
    WebToolkit,
    _common,
    clear_fetch_cache,
    get_reader,
    html2md,
    list_readers,
    read_url,
    register_reader,
    resolve_web_tools,
    search_platform,
    unregister_reader,
    web_fetch,
)
from prompture.tools.web import platform as platform_mod
from prompture.tools.web.readers import _media, matching_readers, search_anilist
from prompture.tools.web.readers import github as gh_mod
from prompture.tools.web.readers import podcasts as pod_mod
from prompture.tools.web.readers import youtube as yt_mod
from prompture.tools.web.readers.anilist import anilist_id, clean_description
from prompture.tools.web.readers.arxiv import arxiv_id
from prompture.tools.web.readers.feeds import looks_like_feed_url, parse_feed_stdlib
from prompture.tools.web.readers.github import parse_github_url
from prompture.tools.web.readers.hackernews import hn_item_id
from prompture.tools.web.readers.podcasts import apple_ids, is_audio_url
from prompture.tools.web.readers.wikipedia import wikipedia_title
from prompture.tools.web.readers.youtube import group_segments, parse_vtt, pick_caption_track, youtube_video_id

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResp:
    def __init__(
        self,
        status: int = 200,
        body: bytes | str = b"",
        *,
        json_data: Any = None,
        headers: dict[str, str] | None = None,
        content_type: str | None = None,
    ) -> None:
        self.headers = CaseInsensitiveDict(headers or {})
        if json_data is not None:
            body = json.dumps(json_data)
            self.headers.setdefault("Content-Type", "application/json")
        if content_type:
            self.headers["Content-Type"] = content_type
        self.content = body.encode() if isinstance(body, str) else body
        self.status_code = status
        self.reason = "OK" if status < 400 else "Error"
        self.encoding = "utf-8"
        self.url = ""

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)

    def iter_content(self, chunk_size: int = 65536):
        yield self.content

    def close(self) -> None:
        pass


Handler = Callable[[str, str, dict[str, Any]], FakeResp]


class FakeSession:
    def __init__(self, routes: list[tuple[str, str, FakeResp | Handler]] | None = None) -> None:
        self.routes = routes or []
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _dispatch(self, method: str, url: str, kw: dict[str, Any]) -> FakeResp:
        self.calls.append((method, url, kw))
        for m, needle, resp in self.routes:
            if m == method and needle in url:
                return resp(method, url, kw) if callable(resp) else resp
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url: str, **kw: Any) -> FakeResp:
        return self._dispatch("GET", url, kw)

    def post(self, url: str, **kw: Any) -> FakeResp:
        return self._dispatch("POST", url, kw)

    def called(self, needle: str) -> list[tuple[str, str, dict[str, Any]]]:
        return [c for c in self.calls if needle in c[1]]


class Exploding:
    """A session that fails the test if anything uses it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"network used: {name}")


PRIVATE_HOSTS = {"intranet.example": "10.0.0.5"}


def fake_resolve(host: str, port: int | None = None) -> list[str]:
    return [PRIVATE_HOSTS.get(host, "93.184.216.34")]


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for var in (
        "JINA_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "PROMPTURE_FETCH_BACKENDS",
        "PROMPTURE_SEARCH_PROVIDERS",
        "PROMPTURE_PROXY",
        "PROMPTURE_WEB_ALLOW_PRIVATE",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
        "SERPER_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "SEARXNG_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("prompture.capabilities.url_safety.resolve_host", fake_resolve)
    monkeypatch.setattr(_common, "settings", SimpleNamespace())
    monkeypatch.setattr(_common, "binary_ok", lambda cmd: False)
    monkeypatch.setattr(_common, "has_module", lambda name: False)
    monkeypatch.setattr(_media, "load_transcriber", lambda: None)
    monkeypatch.setattr(html2md, "_trafilatura_markdown", lambda html, base_url: None)
    clear_fetch_cache()
    pod_mod._episodes.clear()
    yield
    clear_fetch_cache()
    pod_mod._episodes.clear()


def jina_json(content: str, title: str = "Title", url: str = "https://example.com/") -> FakeResp:
    return FakeResp(json_data={"code": 200, "data": {"title": title, "url": url, "content": content}})


# ---------------------------------------------------------------------------
# web_fetch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://169.254.169.254/latest", "http://intranet.example/"])
def test_private_url_refused_before_any_backend(url):
    with pytest.raises(UnsafeURLError):
        web_fetch(url, session=Exploding())  # type: ignore[arg-type]


def test_jina_reader_serves_by_default_with_api_user_agent():
    sess = FakeSession([("GET", "r.jina.ai/https://example.com/", jina_json("Hello **world**"))])
    res = web_fetch("https://example.com", session=sess)  # type: ignore[arg-type]
    assert isinstance(res, FetchResult)
    assert res.served_by == "jina_reader"
    assert res.content == "Hello **world**"
    assert res.title == "Title"
    assert res.truncated is False and res.next_start is None
    headers = sess.calls[0][2]["headers"]
    assert "Authorization" not in headers
    assert headers["User-Agent"].startswith("prompture-web")
    assert "_served by jina_reader_" in res.to_markdown()


def test_jina_key_is_sent_when_configured(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "jina_abc")
    sess = FakeSession([("GET", "r.jina.ai", jina_json("x"))])
    web_fetch("https://example.com", session=sess)  # type: ignore[arg-type]
    assert sess.calls[0][2]["headers"]["Authorization"] == "Bearer jina_abc"


def test_truncation_paging_and_cache():
    body = "".join(f"line {i:02d}\n" for i in range(10))  # 80 chars
    sess = FakeSession([("GET", "r.jina.ai", jina_json(body))])
    first = web_fetch("https://example.com/long", max_chars=30, session=sess)  # type: ignore[arg-type]
    assert first.truncated is True
    assert first.next_start == 30
    assert first.total_chars == len(body.strip())
    assert first.content.endswith("[truncated — call again with start=30]")
    second = web_fetch("https://example.com/long", max_chars=30, start=first.next_start, session=sess)  # type: ignore[arg-type]
    assert second.cached is True
    assert len(sess.calls) == 1  # paging reused the cached page
    third = web_fetch("https://example.com/long", max_chars=30, start=60, session=sess)  # type: ignore[arg-type]
    assert third.truncated is False
    assert "[truncated" not in third.content


def test_challenge_from_jina_moves_to_direct():
    challenge = "<title>Just a moment...</title> cf_chl_opt challenge-platform checking your browser"
    html = "<html><head><title>Real Page</title></head><body><h1>Real</h1><p>Actual content here.</p></body></html>"
    sess = FakeSession(
        [
            ("GET", "r.jina.ai", jina_json(challenge, title="Just a moment...")),
            ("GET", "example.com/page", FakeResp(body=html, content_type="text/html; charset=utf-8")),
        ]
    )
    res = web_fetch("https://example.com/page", session=sess)  # type: ignore[arg-type]
    assert res.served_by == "direct"
    assert res.title == "Real Page"
    assert "# Real" in res.content
    assert res.route["fallback"] is True
    assert res.route["attempts"][0]["category"] == "challenge_page"


SHELL_MD = "Sorry, this site requires a modern browser.\nPlease upgrade to a newer web browser.\n\n[Home](/) [Search](/search)"


def test_js_shell_from_jina_moves_to_direct():
    html = "<html><head><title>Cast</title></head><body><h1>Cast</h1><p>Server-rendered list.</p></body></html>"
    sess = FakeSession(
        [
            ("GET", "r.jina.ai", jina_json(SHELL_MD, title="Some App")),
            ("GET", "example.com/cast", FakeResp(body=html, content_type="text/html; charset=utf-8")),
        ]
    )
    res = web_fetch("https://example.com/cast", session=sess)  # type: ignore[arg-type]
    assert res.served_by == "direct"
    assert "Server-rendered list." in res.content
    assert res.route["attempts"][0]["backend"] == "jina_reader"


def test_js_shell_everywhere_fails_and_is_not_cached():
    shell_html = "<html><body><p>Sorry, this site requires a modern browser.</p></body></html>"
    responses = [
        ("GET", "r.jina.ai", jina_json(SHELL_MD)),
        ("GET", "example.com/app", FakeResp(body=shell_html, content_type="text/html")),
    ]
    sess = FakeSession(list(responses))
    with pytest.raises(AllBackendsFailedError):
        web_fetch("https://example.com/app", session=sess)  # type: ignore[arg-type]
    sess = FakeSession(list(responses))
    with pytest.raises(AllBackendsFailedError):
        web_fetch("https://example.com/app", session=sess)  # type: ignore[arg-type]
    assert sess.calls  # the shell was not served from the cache

def test_override_env_and_direct_challenge_falls_to_jina(monkeypatch):
    monkeypatch.setenv("PROMPTURE_FETCH_BACKENDS", "direct,jina_reader")
    blocked = FakeResp(403, body="<title>Just a moment...</title><div id=cf_chl_opt></div>", content_type="text/html")
    sess = FakeSession(
        [
            ("GET", "r.jina.ai", jina_json("rendered by jina")),
            ("GET", "example.com/x", blocked),
        ]
    )
    res = web_fetch("https://example.com/x", session=sess)  # type: ignore[arg-type]
    assert res.served_by == "jina_reader"
    assert [a["backend"] for a in res.route["attempts"]] == ["direct", "jina_reader"]


def test_jina_rejection_fails_over():
    sess = FakeSession(
        [
            ("GET", "r.jina.ai", FakeResp(422, b'{"message": "cannot process"}')),
            ("GET", "example.com/doc.txt", FakeResp(body="plain text body", content_type="text/plain")),
        ]
    )
    res = web_fetch("https://example.com/doc.txt", session=sess)  # type: ignore[arg-type]
    assert res.served_by == "direct"
    assert res.content == "plain text body"
    assert res.content_type == "text/plain"


def test_direct_json_and_unsupported_binary():
    sess = FakeSession(
        [
            ("GET", "example.com/data.json", FakeResp(json_data={"a": 1})),
            ("GET", "example.com/bin", FakeResp(body=b"\x00\x01", content_type="application/octet-stream")),
        ]
    )
    res = web_fetch("https://example.com/data.json", backends=["direct"], session=sess)  # type: ignore[arg-type]
    assert '"a": 1' in res.content
    with pytest.raises(AllBackendsFailedError):
        web_fetch("https://example.com/bin", backends=["direct"], session=sess)  # type: ignore[arg-type]


def test_redirect_to_private_is_refused():
    sess = FakeSession([("GET", "example.com/r", FakeResp(302, b"", headers={"Location": "http://127.0.0.1/secret"}))])
    with pytest.raises(UnsafeURLError):
        web_fetch("https://example.com/r", backends=["direct"], session=sess)  # type: ignore[arg-type]


def test_compress_collapses_blank_lines():
    sess = FakeSession([("GET", "r.jina.ai", jina_json("a   \n\n\n\n\nb"))])
    res = web_fetch("https://example.com/c", compress=True, session=sess)  # type: ignore[arg-type]
    assert res.content == "a\n\nb"


def test_unknown_fetch_backend():
    with pytest.raises(ValueError):
        web_fetch("https://example.com", backends=["nope"])


def test_html_to_markdown_basics():
    html = (
        "<html><head><title>T</title></head><body><nav><a href='/x'>Nav</a></nav>"
        "<main><h2>Head</h2><p>See <a href='/doc'>docs</a> and <code>x()</code>.</p>"
        "<ul><li>one</li><li>two</li></ul><pre>a = 1\n  b = 2</pre><script>evil()</script></main></body></html>"
    )
    md = html2md.html_to_markdown(html, base_url="https://site.example/", use_trafilatura=False)
    assert "## Head" in md
    assert "[docs](https://site.example/doc)" in md
    assert "`x()`" in md
    assert "- one\n- two" in md
    assert "```\na = 1\n  b = 2\n```" in md
    assert "evil" not in md and "Nav" not in md


# ---------------------------------------------------------------------------
# URL matching
# ---------------------------------------------------------------------------


def test_url_matchers():
    assert youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1") == "dQw4w9WgXcQ"
    assert youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert youtube_video_id("https://m.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert youtube_video_id("https://www.youtube.com/channel/abc") is None

    assert parse_github_url("https://github.com/psf/requests").kind == "repo"
    t = parse_github_url("https://github.com/psf/requests/blob/main/src/requests/api.py")
    assert (t.kind, t.ref, t.path) == ("file", "main", "src/requests/api.py")
    assert parse_github_url("https://github.com/psf/requests/pull/12").number == 12
    assert parse_github_url("https://github.com/o/r/discussions/3").kind == "discussion"
    assert parse_github_url("https://github.com/settings/profile") is None
    assert parse_github_url("https://github.com/psf") is None

    assert hn_item_id("https://news.ycombinator.com/item?id=8863") == 8863
    assert hn_item_id("https://news.ycombinator.com/news") is None
    assert arxiv_id("https://arxiv.org/abs/1706.03762v7") == "1706.03762v7"
    assert arxiv_id("https://arxiv.org/pdf/1706.03762.pdf") == "1706.03762"
    assert arxiv_id("https://arxiv.org/abs/hep-th/9901001") == "hep-th/9901001"
    assert wikipedia_title("https://de.wikipedia.org/wiki/Berlin") == ("de", "Berlin")
    assert wikipedia_title("https://en.wikipedia.org/wiki/Special:Random") is None
    assert looks_like_feed_url("https://blog.example/feed/")
    assert looks_like_feed_url("https://x.example/index.xml")
    assert not looks_like_feed_url("https://x.example/sitemap.xml")
    assert is_audio_url("https://cdn.example/ep1.mp3?x=1")
    assert apple_ids("https://podcasts.apple.com/us/podcast/show/id123?i=456") == ("123", "456")


def test_reader_routing_order():
    names = [r.name for r in list_readers()]
    assert names[:8] == ["youtube", "github", "hackernews", "arxiv", "wikipedia", "anilist", "podcasts", "feeds"]
    assert [r.name for r in matching_readers("https://anilist.co/anime/1/x")] == ["anilist"]
    assert [r.name for r in matching_readers("https://youtu.be/dQw4w9WgXcQ")] == ["youtube"]
    assert matching_readers("https://example.com/") == []


# ---------------------------------------------------------------------------
# read_url — routing, fallback, paging, registry
# ---------------------------------------------------------------------------


def test_read_url_falls_back_to_web_fetch():
    sess = FakeSession([("GET", "r.jina.ai", jina_json("page body"))])
    res = read_url("https://example.com/article", session=sess)  # type: ignore[arg-type]
    assert isinstance(res, ReadResult)
    assert res.reader == "web_fetch"
    assert res.route["served_by"] == "web_fetch/jina_reader"


def test_read_url_refuses_private_url():
    with pytest.raises(UnsafeURLError):
        read_url("http://10.1.2.3/", session=Exploding())  # type: ignore[arg-type]


def test_custom_reader_first_wins_and_pages():
    class Echo:
        name = "echo"

        def can_handle(self, url: str) -> bool:
            return "example.com" in url

        def read(self, url: str, **kw: Any) -> ReadResult:
            return ReadResult(url, "Echo", "x" * 50, "echo", "custom")

        def check(self, live: bool = False):
            from prompture.capabilities import HealthStatus

            return HealthStatus("echo", "ok")

    register_reader(Echo(), first=True)
    try:
        res = read_url("https://example.com/a", max_chars=20, session=Exploding())  # type: ignore[arg-type]
        assert res.reader == "echo"
        assert res.truncated and res.next_start == 20
        assert res.content.endswith("[truncated — call again with start=20]")
        assert list_readers()[0].name == "echo"
    finally:
        unregister_reader("echo")
    assert get_reader("echo") is None


def test_register_reader_rejects_non_reader():
    with pytest.raises(TypeError):
        register_reader(object())


def test_unknown_forced_reader():
    with pytest.raises(ValueError, match="Unknown reader"):
        read_url("https://example.com", reader="nope", session=Exploding())  # type: ignore[arg-type]


def test_failing_reader_falls_back_and_records_route():
    sess = FakeSession(
        [
            ("GET", "hacker-news.firebaseio.com", FakeResp(404, b"null")),
            ("GET", "hn.algolia.com", FakeResp(404, b"{}")),
            ("GET", "r.jina.ai", jina_json("hn page via fetch")),
        ]
    )
    res = read_url("https://news.ycombinator.com/item?id=1", session=sess)  # type: ignore[arg-type]
    assert res.reader == "web_fetch"
    assert res.route["fallback"] is True
    assert res.route["attempts"][0]["backend"] == "hackernews"
    assert res.route["attempts"][0]["status"] == "error"


def test_unmatched_url_with_feed_body_is_rendered_as_feed():
    sess = FakeSession(
        [
            ("GET", "r.jina.ai", FakeResp(422, b"{}")),
            ("GET", "example.com/frontpage", FakeResp(body=RSS, content_type="application/rss+xml")),
        ]
    )
    res = read_url("https://example.com/frontpage", session=sess)  # type: ignore[arg-type]
    assert res.reader == "feeds"
    assert res.kind == "feed"


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

YT = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
OEMBED = FakeResp(json_data={"title": "Video Title", "author_name": "Channel"})


def _fake_transcript_module(snippets: list[tuple[float, str]]) -> types.ModuleType:
    mod = types.ModuleType("youtube_transcript_api")

    class Fetched(list):
        language_code = "en"

    class Api:
        def fetch(self, video_id: str, languages: list[str]):
            assert video_id == "dQw4w9WgXcQ"
            return Fetched(SimpleNamespace(start=s, text=t, duration=1.0) for s, t in snippets)

        def list(self, video_id: str):
            return []

    mod.YouTubeTranscriptApi = Api  # type: ignore[attr-defined]
    return mod


def test_youtube_transcript_api_path(monkeypatch):
    monkeypatch.setattr(_common, "has_module", lambda name: name == "youtube_transcript_api")
    snippets = [(0.0, "never gonna"), (2.0, "give you up"), (35.0, "never gonna let you down")]
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", _fake_transcript_module(snippets))
    sess = FakeSession([("GET", "youtube.com/oembed", OEMBED)])
    res = read_url(YT, session=sess)  # type: ignore[arg-type]
    assert res.reader == "youtube"
    assert res.kind == "video"
    assert res.title == "Video Title"
    assert res.route["served_by"] == "youtube/transcript_api"
    assert res.content.splitlines()[0] == "[00:00] never gonna give you up"
    assert "[00:35] never gonna let you down" in res.content
    assert res.meta["channel"] == "Channel"


def test_youtube_yt_dlp_captions_path(monkeypatch):
    monkeypatch.setattr(_common, "binary_ok", lambda cmd: cmd == "yt-dlp")
    info = {
        "title": "Captioned",
        "channel": "Chan",
        "duration": 60,
        "automatic_captions": {
            "de": [{"ext": "json3", "url": "https://www.youtube.com/api/timedtext?lang=de"}],
            "en": [{"ext": "vtt", "url": "https://www.youtube.com/api/timedtext?lang=en&fmt=vtt"}],
        },
    }
    seen: list[list[str]] = []

    def fake_run(argv, timeout=60.0, **kw):
        seen.append(list(argv))
        return json.dumps(info)

    monkeypatch.setattr(yt_mod, "run_command", fake_run)
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello <c>there</c>\n\n00:00:03.000 --> 00:00:05.000\nhello there\nworld\n"
    sess = FakeSession([("GET", "timedtext", FakeResp(body=vtt, content_type="text/vtt"))])
    res = read_url(YT, session=sess)  # type: ignore[arg-type]
    assert res.route["served_by"] == "youtube/yt_dlp"
    assert res.title == "Captioned"
    assert res.content == "[00:01] hello there world"
    assert res.meta["language"] == "en"
    assert seen[0][0] == "yt-dlp" and "--skip-download" in seen[0]


def test_youtube_transcription_only_when_available(monkeypatch):
    calls: list[str] = []

    def transcribe(source: str, **kw: Any):
        calls.append(source)
        return SimpleNamespace(text="spoken words", segments=[], to_markdown=lambda: "[00:00] spoken words")

    monkeypatch.setattr(_media, "load_transcriber", lambda: transcribe)
    sess = FakeSession([("GET", "youtube.com/oembed", OEMBED)])
    res = read_url(YT, session=sess)  # type: ignore[arg-type]
    assert res.route["served_by"] == "youtube/transcription"
    assert res.content == "[00:00] spoken words"
    assert res.meta["transcript_source"] == "speech_to_text"
    assert calls == [YT]


def test_youtube_falls_back_to_page_when_nothing_installed():
    sess = FakeSession([("GET", "r.jina.ai", jina_json("video page text", title="Watch page"))])
    res = read_url(YT, session=sess)  # type: ignore[arg-type]
    assert res.route["served_by"] == "youtube/web_fetch"
    assert "No transcript was available" in res.content
    skipped = [a["backend"] for a in res.route["attempts"][-1]["attempts"] if a["status"] == "skipped"]
    assert skipped == ["transcript_api", "yt_dlp", "transcription"]


def test_load_transcriber_respects_availability(monkeypatch):
    monkeypatch.undo()  # use the real loader
    fake = types.ModuleType("prompture.media.understand")
    fake.transcribe = lambda *a, **k: None  # type: ignore[attr-defined]
    fake.transcription_available = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prompture.media.understand", fake)
    assert _media.load_transcriber() is None
    fake.transcription_available = lambda: True  # type: ignore[attr-defined]
    assert _media.load_transcriber() is fake.transcribe


def test_caption_helpers():
    assert parse_vtt("WEBVTT\n\n1\n00:00:00.500 --> 00:00:01.000\na\n\n00:01:02.000 --> 00:01:03.000\nb\n") == [
        (0.5, "a"),
        (62.0, "b"),
    ]
    assert group_segments([(0, "a"), (10, "a"), (40, "b")], timestamps=False) == "a\n\nb"
    info = {"subtitles": {"fr": [{"ext": "vtt", "url": "https://x/fr"}]}, "automatic_captions": {}}
    assert pick_caption_track(info) == ("https://x/fr", "vtt", "fr")


# ---------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------

RSS = """<?xml version="1.0"?>
<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">
<channel><title>My Show</title><link>https://show.example/</link><description>About &amp; more</description>
<item><title>Episode 2</title><link>https://show.example/ep2</link><pubDate>Tue, 01 Sep 2026 10:00:00 GMT</pubDate>
<description>&lt;p&gt;Second &lt;b&gt;episode&lt;/b&gt;&lt;/p&gt;</description>
<enclosure url="https://cdn.example/ep2.mp3" type="audio/mpeg" length="123"/>
<podcast:transcript url="https://show.example/ep2.vtt" type="text/vtt"/></item>
<item><title>Episode 1</title><link>https://show.example/ep1</link><description>First</description>
<enclosure url="https://cdn.example/ep1.mp3" type="audio/mpeg"/></item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Atom Blog</title><link href="https://atom.example/"/>
<entry><title>Post</title><link href="https://atom.example/post"/><updated>2026-09-01T00:00:00Z</updated>
<summary>Summary text</summary></entry></feed>"""


def test_feed_reader_rss():
    sess = FakeSession([("GET", "show.example/feed.xml", FakeResp(body=RSS, content_type="application/rss+xml"))])
    res = read_url("https://show.example/feed.xml", session=sess)  # type: ignore[arg-type]
    assert res.reader == "feeds"
    assert res.route["served_by"] == "feeds/stdlib"
    assert res.title == "My Show"
    assert "## [Episode 2](https://show.example/ep2)" in res.content
    assert "Second **episode**" in res.content
    assert res.meta["entry_count"] == 2
    assert res.meta["entries"][0]["transcripts"][0]["url"] == "https://show.example/ep2.vtt"


def test_feed_parser_atom_and_entity_rejection():
    feed = parse_feed_stdlib(ATOM.encode())
    assert feed["title"] == "Atom Blog"
    assert feed["entries"][0]["link"] == "https://atom.example/post"
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><rss><channel></channel></rss>'
    with pytest.raises(Exception, match="entities"):
        parse_feed_stdlib(bomb)


def test_feed_reader_on_page_advertising_feed():
    page = '<html><head><title>Blog</title><link rel="alternate" type="application/atom+xml" href="/atom.xml"></head><body>hi</body></html>'
    sess = FakeSession(
        [
            ("GET", "atom.example/atom.xml", FakeResp(body=ATOM, content_type="application/atom+xml")),
            ("GET", "atom.example/", FakeResp(body=page, content_type="text/html")),
        ]
    )
    res = read_url("https://atom.example/", reader="feeds", session=sess)  # type: ignore[arg-type]
    assert res.title == "Atom Blog"
    assert res.meta["feed_url"] == "https://atom.example/atom.xml"


def test_feedparser_used_when_installed(monkeypatch):
    monkeypatch.setattr(_common, "has_module", lambda name: name == "feedparser")
    fake = types.ModuleType("feedparser")
    fake.parse = lambda data: {  # type: ignore[attr-defined]
        "feed": {"title": "FP Feed"},
        "entries": [{"title": "E", "link": "https://fp.example/e", "summary": "s", "enclosures": []}],
        "version": "rss20",
    }
    monkeypatch.setitem(sys.modules, "feedparser", fake)
    sess = FakeSession([("GET", "fp.example/rss", FakeResp(body=RSS, content_type="application/rss+xml"))])
    res = read_url("https://fp.example/rss", session=sess)  # type: ignore[arg-type]
    assert res.route["served_by"] == "feeds/feedparser"
    assert res.title == "FP Feed"


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


def test_github_repo_with_readme():
    repo = {
        "full_name": "psf/requests",
        "description": "HTTP for Humans",
        "stargazers_count": 50000,
        "forks_count": 9000,
        "open_issues_count": 200,
        "language": "Python",
        "license": {"spdx_id": "Apache-2.0"},
        "default_branch": "main",
        "topics": ["http"],
    }
    sess = FakeSession(
        [
            ("GET", "repos/psf/requests/readme", FakeResp(body="# Requests\n\nREADME body", content_type="text/plain")),
            ("GET", "api.github.com/repos/psf/requests", FakeResp(json_data=repo)),
        ]
    )
    res = read_url("https://github.com/psf/requests", session=sess)  # type: ignore[arg-type]
    assert res.route["served_by"] == "github/rest"
    assert res.kind == "repo"
    assert "Stars: 50000" in res.content
    assert "## README\n\n# Requests" in res.content
    readme_call = sess.called("readme")[0]
    assert readme_call[2]["headers"]["Accept"] == "application/vnd.github.raw"
    assert "Authorization" not in readme_call[2]["headers"]


def test_github_issue_with_comments_and_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "a" * 36)
    issue = {
        "title": "Bug",
        "state": "open",
        "user": {"login": "alice"},
        "created_at": "2026-01-02T00:00:00Z",
        "body": "It breaks",
        "labels": [{"name": "bug"}],
        "comments": 1,
    }
    comments = [{"user": {"login": "bob"}, "created_at": "2026-01-03T00:00:00Z", "body": "Same here"}]
    sess = FakeSession(
        [
            ("GET", "issues/7/comments", FakeResp(json_data=comments)),
            ("GET", "issues/7", FakeResp(json_data=issue)),
        ]
    )
    res = read_url("https://github.com/o/r/issues/7", session=sess)  # type: ignore[arg-type]
    assert res.title == "Bug"
    assert "Labels: bug" in res.content
    assert "### @bob — 2026-01-03\n\nSame here" in res.content
    assert sess.calls[0][2]["headers"]["Authorization"].startswith("Bearer ghp_")


def test_github_pr_and_file_and_tree():
    issue = {"title": "Add X", "state": "closed", "user": {"login": "a"}, "body": "desc"}
    pr = {
        "merged": True,
        "additions": 10,
        "deletions": 2,
        "changed_files": 3,
        "head": {"label": "a:x"},
        "base": {"label": "o:main"},
    }
    sess = FakeSession(
        [
            ("GET", "issues/5/comments", FakeResp(json_data=[])),
            ("GET", "issues/5", FakeResp(json_data=issue)),
            ("GET", "pulls/5", FakeResp(json_data=pr)),
            ("GET", "contents/src/app.py", FakeResp(body="print('hi')\n", content_type="text/plain")),
            (
                "GET",
                "contents/src",
                FakeResp(json_data=[{"name": "app.py", "type": "file", "size": 12}, {"name": "pkg", "type": "dir"}]),
            ),
        ]
    )
    res = read_url("https://github.com/o/r/pull/5", session=sess)  # type: ignore[arg-type]
    assert res.kind == "pr"
    assert "+10 −2 in 3 files · merged" in res.content
    f = read_url("https://github.com/o/r/blob/main/src/app.py", session=sess)  # type: ignore[arg-type]
    assert f.content == "```python\nprint('hi')\n```"
    t = read_url("https://github.com/o/r/tree/main/src", session=sess)  # type: ignore[arg-type]
    assert t.content.splitlines() == ["- pkg/", "- app.py (12 bytes)"]


def test_github_discussion_via_graphql(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "b" * 36)
    data = {
        "data": {
            "repository": {
                "discussion": {
                    "title": "Idea",
                    "body": "What if",
                    "createdAt": "2026-02-01T00:00:00Z",
                    "author": {"login": "carol"},
                    "category": {"name": "Ideas"},
                    "answer": {"body": "Yes", "author": {"login": "dave"}},
                    "comments": {"nodes": [{"body": "+1", "createdAt": "2026-02-02", "author": {"login": "erin"}}]},
                }
            }
        }
    }
    sess = FakeSession([("POST", "api.github.com/graphql", FakeResp(json_data=data))])
    res = read_url("https://github.com/o/r/discussions/9", session=sess)  # type: ignore[arg-type]
    assert res.title == "Idea"
    assert "## Accepted answer (@dave)" in res.content
    assert sess.calls[0][2]["json"]["variables"] == {"owner": "o", "name": "r", "number": 9}


def test_github_rate_limit_falls_to_gh_cli(monkeypatch):
    monkeypatch.setattr(_common, "binary_ok", lambda cmd: cmd == "gh")
    argv_seen: list[list[str]] = []

    def fake_run(argv, timeout=60.0, **kw):
        argv_seen.append(list(argv))
        if argv[2].endswith("/readme"):
            return "gh readme"
        return json.dumps({"full_name": "o/r", "description": "via gh"})

    monkeypatch.setattr(gh_mod, "run_command", fake_run)
    limited = FakeResp(403, b'{"message": "API rate limit exceeded for 1.2.3.4"}', content_type="application/json")
    sess = FakeSession([("GET", "api.github.com", limited)])
    res = read_url("https://github.com/o/r", session=sess)  # type: ignore[arg-type]
    assert res.route["served_by"] == "github/gh"
    assert "via gh" in res.content
    assert argv_seen[0][:3] == ["gh", "api", "repos/o/r"]


# ---------------------------------------------------------------------------
# Hacker News, arXiv, Wikipedia
# ---------------------------------------------------------------------------


def test_hackernews_firebase_thread():
    items = {
        1: {
            "id": 1,
            "type": "story",
            "title": "Show HN",
            "by": "pg",
            "score": 99,
            "descendants": 2,
            "url": "https://x.example",
            "kids": [2, 3],
        },
        2: {"id": 2, "by": "a", "text": "Great <i>work</i>", "kids": [4]},
        3: {"id": 3, "deleted": True},
        4: {"id": 4, "by": "b", "text": "Agreed"},
    }

    def handler(method: str, url: str, kw: dict[str, Any]) -> FakeResp:
        item_id = int(url.rsplit("/", 1)[-1].split(".")[0])
        return FakeResp(json_data=items[item_id])

    sess = FakeSession([("GET", "hacker-news.firebaseio.com", handler)])
    res = read_url("https://news.ycombinator.com/item?id=1", session=sess)  # type: ignore[arg-type]
    assert res.route["served_by"] == "hackernews/firebase"
    assert res.title == "Show HN"
    assert "**a**:\nGreat _work_" in res.content
    assert "> **b**:\n> Agreed" in res.content
    assert res.meta["comments_shown"] == 2


ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/1706.03762v7</id><updated>2023-08-02T00:00:00Z</updated>
<published>2017-06-12T17:57:34Z</published><title>Attention Is All
 You Need</title><summary>  The dominant sequence transduction models...  </summary>
<author><name>Ashish Vaswani</name></author><author><name>Noam Shazeer</name></author>
<arxiv:primary_category term="cs.CL"/><category term="cs.CL"/><category term="cs.LG"/>
<link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related" type="application/pdf"/></entry></feed>"""


def test_arxiv_reader_api():
    sess = FakeSession(
        [("GET", "export.arxiv.org/api/query", FakeResp(body=ARXIV_ATOM, content_type="application/atom+xml"))]
    )
    res = read_url("https://arxiv.org/abs/1706.03762", session=sess)  # type: ignore[arg-type]
    assert res.title == "Attention Is All You Need"
    assert res.kind == "paper"
    assert "**Authors:** Ashish Vaswani, Noam Shazeer" in res.content
    assert "## Abstract\n\nThe dominant sequence transduction models..." in res.content
    assert res.meta["primary_category"] == "cs.CL"
    assert sess.calls[0][2]["params"]["id_list"] == "1706.03762"


def test_wikipedia_reader_summary_and_article():
    summary = {
        "title": "Python (programming language)",
        "description": "General-purpose programming language",
        "extract": "Python is a high-level language.",
        "type": "standard",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python_(programming_language)"}},
    }
    html = '<html><body><section><h2>History</h2><p>Created by Guido<sup class="mw-ref reference"><a href="#c1">[1]</a></sup>.</p></section></body></html>'
    sess = FakeSession(
        [
            ("GET", "page/summary/", FakeResp(json_data=summary)),
            ("GET", "page/html/", FakeResp(body=html, content_type="text/html")),
        ]
    )
    res = read_url("https://en.wikipedia.org/wiki/Python_(programming_language)", session=sess)  # type: ignore[arg-type]
    assert res.reader == "wikipedia"
    assert res.content.startswith("_General-purpose programming language_\n\nPython is a high-level language.")
    assert "## History" in res.content
    assert "Created by Guido." in res.content and "[1]" not in res.content
    assert sess.calls[0][2]["headers"]["User-Agent"].startswith("prompture-web")
    short = read_url("https://en.wikipedia.org/wiki/Python_(programming_language)", full=False, session=sess)  # type: ignore[arg-type]
    assert "## History" not in short.content


def _anilist_edge(cid: int, name: str, role: str, gender: str | None, voice: str) -> dict:
    return {
        "role": role,
        "node": {
            "id": cid,
            "siteUrl": f"https://anilist.co/character/{cid}",
            "gender": gender,
            "age": None,
            "name": {"full": name, "native": None, "alternative": []},
            "image": {"medium": None},
            "description": f"{name} keeps the lighthouse.~!Secretly a ghost.!~ &quot;Calm&quot;",
        },
        "voiceActors": [{"name": {"full": voice, "native": None}, "languageV2": "Japanese"}],
    }


def test_anilist_url_parsing_and_description_cleanup():
    assert anilist_id("https://anilist.co/anime/123/Harbor-Lights/characters") == ("anime", 123)
    assert anilist_id("https://anilist.co/character/77") == ("character", 77)
    assert anilist_id("https://anilist.co/user/someone") is None
    assert anilist_id("https://example.com/anime/123") is None
    assert clean_description("A~!hidden!~ B &amp; C") == "A B & C"
    assert clean_description("A~!shown!~", spoilers=True) == "Ashown"


def test_anilist_reader_pages_the_cast():
    pages = {
        1: {"hasNextPage": True, "edges": [_anilist_edge(1, "Mira Tavel", "MAIN", "Female", "Voice One")]},
        2: {"hasNextPage": False, "edges": [_anilist_edge(2, "Oren Pask", "SUPPORTING", None, "Voice Two")]},
    }

    def answer(method: str, url: str, kw: dict) -> FakeResp:
        variables = kw["json"]["variables"]
        assert variables["id"] == 123 and variables["lang"] == "JAPANESE"
        block = pages[variables["page"]]
        media = {
            "id": 123,
            "type": "ANIME",
            "format": "TV",
            "status": "FINISHED",
            "episodes": 12,
            "chapters": None,
            "seasonYear": 2001,
            "siteUrl": "https://anilist.co/anime/123",
            "title": {"romaji": "Minato no Akari", "english": "Harbor Lights", "native": None},
            "description": "A quiet town.<br>By the sea.",
            "characters": {"pageInfo": {"hasNextPage": block["hasNextPage"]}, "edges": block["edges"]},
        }
        return FakeResp(json_data={"data": {"Media": media}})

    sess = FakeSession([("POST", "graphql.anilist.co", answer)])
    res = read_url("https://anilist.co/anime/123/Harbor-Lights/characters", session=sess)  # type: ignore[arg-type]
    assert res.reader == "anilist" and res.kind == "cast"
    assert res.title == "Harbor Lights"
    assert len(sess.calls) == 2
    assert [c["name"] for c in res.meta["characters"]] == ["Mira Tavel", "Oren Pask"]
    first = res.meta["characters"][0]
    assert first["role"] == "MAIN" and first["gender"] == "Female"
    assert first["voice_actors"] == [{"name": "Voice One", "native": "", "language": "Japanese"}]
    assert first["description"] == 'Mira Tavel keeps the lighthouse. "Calm"'
    assert "### Mira Tavel" in res.content and "Voice: Voice One (Japanese)" in res.content
    assert "ghost" not in res.content
    assert res.meta["more_characters"] is False


def test_anilist_reader_stops_at_max_characters():
    edges = [_anilist_edge(i, f"Extra {i}", "BACKGROUND", None, "V") for i in range(25)]
    media = {"id": 9, "title": {"romaji": "Long Show"}, "characters": {"pageInfo": {"hasNextPage": True}, "edges": edges}}
    sess = FakeSession([("POST", "graphql.anilist.co", FakeResp(json_data={"data": {"Media": media}}))])
    res = read_url("https://anilist.co/anime/9", max_characters=10, session=sess)  # type: ignore[arg-type]
    assert len(res.meta["characters"]) == 10
    assert len(sess.calls) == 1
    assert res.meta["more_characters"] is True


def test_anilist_character_and_search():
    character = {
        "id": 77,
        "siteUrl": "https://anilist.co/character/77",
        "gender": "Male",
        "age": "30s",
        "name": {"full": "Oren Pask", "native": None, "alternative": []},
        "image": {"medium": None},
        "description": "A ferryman.",
        "media": {
            "edges": [
                {
                    "characterRole": "SUPPORTING",
                    "node": {"id": 123, "type": "ANIME", "format": "TV", "seasonYear": 2001,
                             "siteUrl": "https://anilist.co/anime/123", "title": {"english": "Harbor Lights"}},
                    "voiceActors": [],
                }
            ]
        },
    }
    found = {"Page": {"media": [{"id": 123, "type": "ANIME", "format": "TV", "episodes": 12, "seasonYear": 2001,
                                  "siteUrl": "https://anilist.co/anime/123",
                                  "title": {"romaji": "Minato no Akari", "english": "Harbor Lights"},
                                  "synonyms": ["HL"]}]}}

    def answer(method: str, url: str, kw: dict) -> FakeResp:
        query = kw["json"]["query"]
        return FakeResp(json_data={"data": {"Character": character} if "Character(" in query else found})

    sess = FakeSession([("POST", "graphql.anilist.co", answer)])
    res = read_url("https://anilist.co/character/77", session=sess)  # type: ignore[arg-type]
    assert res.kind == "character" and res.title == "Oren Pask"
    assert res.meta["appearances"][0]["title"] == "Harbor Lights"
    assert "- Harbor Lights (TV, 2001, SUPPORTING)" in res.content
    hits = search_anilist("harbor", session=sess)
    assert hits[0]["id"] == 123 and hits[0]["title"] == "Harbor Lights" and hits[0]["year"] == 2001
    assert sess.calls[-1][2]["json"]["variables"]["type"] == "ANIME"


def test_anilist_missing_title_falls_back_to_fetch():
    errors = {"errors": [{"message": "Not Found.", "status": 404}], "data": {"Media": None}}
    sess = FakeSession(
        [
            ("POST", "graphql.anilist.co", FakeResp(404, json_data=errors)),
            ("GET", "r.jina.ai", jina_json("plain page")),
        ]
    )
    res = read_url("https://anilist.co/anime/999999", session=sess)  # type: ignore[arg-type]
    assert res.reader == "web_fetch"


# ---------------------------------------------------------------------------
# Podcasts
# ---------------------------------------------------------------------------

ITUNES = {
    "results": [
        {
            "wrapperType": "track",
            "kind": "podcast",
            "collectionName": "My Show",
            "feedUrl": "https://show.example/feed.xml",
        },
        {
            "wrapperType": "podcastEpisode",
            "trackId": 456,
            "trackName": "Episode 2",
            "episodeUrl": "https://cdn.example/ep2.mp3",
            "description": "Second episode notes",
            "releaseDate": "2026-09-01T10:00:00Z",
            "trackTimeMillis": 3_600_000,
        },
    ]
}


def test_podcast_apple_link_uses_publisher_transcript():
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nWelcome to the show\n"
    sess = FakeSession(
        [
            ("GET", "itunes.apple.com/lookup", FakeResp(json_data=ITUNES)),
            ("GET", "show.example/feed.xml", FakeResp(body=RSS, content_type="application/rss+xml")),
            ("GET", "show.example/ep2.vtt", FakeResp(body=vtt, content_type="text/vtt")),
        ]
    )
    res = read_url("https://podcasts.apple.com/us/podcast/my-show/id123?i=456", session=sess)  # type: ignore[arg-type]
    assert res.route["served_by"] == "podcasts/feed_transcript"
    assert res.title == "Episode 2"
    assert "[00:01] Welcome to the show" in res.content
    assert res.meta["transcript_source"] == "publisher"
    assert res.meta["audio_url"] == "https://cdn.example/ep2.mp3"


def test_podcast_audio_url_transcribed_when_configured(monkeypatch):
    seen: list[str] = []

    def transcribe(source: str, **kw: Any):
        seen.append(source)
        seg = SimpleNamespace(start=0.0, end=2.0, text="hello listeners")
        return SimpleNamespace(text="hello listeners", segments=[seg], to_markdown=None)

    monkeypatch.setattr(_media, "load_transcriber", lambda: transcribe)
    res = read_url("https://cdn.example/ep1.mp3", session=Exploding())  # type: ignore[arg-type]
    assert res.route["served_by"] == "podcasts/transcription"
    assert "[00:00] hello listeners" in res.content
    assert seen == ["https://cdn.example/ep1.mp3"]


def test_podcast_show_notes_when_no_transcript():
    itunes = {"results": [dict(ITUNES["results"][1])]}
    sess = FakeSession([("GET", "itunes.apple.com/lookup", FakeResp(json_data=itunes))])
    res = read_url("https://podcasts.apple.com/us/podcast/x/id123?i=456", session=sess)  # type: ignore[arg-type]
    assert res.route["served_by"] == "podcasts/show_notes"
    assert "Second episode notes" in res.content
    assert res.meta["transcript_source"] is None


# ---------------------------------------------------------------------------
# Platform search
# ---------------------------------------------------------------------------


def test_platform_youtube_via_yt_dlp(monkeypatch):
    monkeypatch.setattr(_common, "binary_ok", lambda cmd: cmd == "yt-dlp")
    lines = "\n".join(
        json.dumps(x)
        for x in (
            {"id": "aaaaaaaaaaa", "title": "Vid A", "channel": "C", "duration": 10},
            {"id": "bbbbbbbbbbb", "title": "Vid B", "url": "https://www.youtube.com/watch?v=bbbbbbbbbbb"},
        )
    )
    argv_seen: list[list[str]] = []

    def fake_run(argv, timeout=60.0, **kw):
        argv_seen.append(list(argv))
        return lines + "\nnot json\n"

    monkeypatch.setattr(platform_mod, "run_command", fake_run)
    results = search_platform("youtube", "lofi", max_results=2)
    assert [r.title for r in results] == ["Vid A", "Vid B"]
    assert results[0].url == "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    assert results[0].extra["served_by"] == "yt_dlp"
    assert argv_seen[0][:2] == ["yt-dlp", "ytsearch2:lofi"]
    assert "--flat-playlist" in argv_seen[0]


def test_platform_github_kinds(monkeypatch):
    repos = {
        "items": [{"full_name": "a/b", "html_url": "https://github.com/a/b", "description": "d", "stargazers_count": 5}]
    }
    issues = {"items": [{"title": "I", "html_url": "https://github.com/a/b/issues/1", "body": "x", "pull_request": {}}]}
    code = {
        "items": [
            {
                "path": "x.py",
                "name": "x.py",
                "html_url": "https://github.com/a/b/blob/main/x.py",
                "repository": {"full_name": "a/b"},
            }
        ]
    }
    sess = FakeSession(
        [
            ("GET", "search/repositories", FakeResp(json_data=repos)),
            ("GET", "search/issues", FakeResp(json_data=issues)),
            ("GET", "search/code", FakeResp(json_data=code)),
        ]
    )
    assert search_platform("github", "q", session=sess)[0].extra["stars"] == 5  # type: ignore[arg-type]
    assert search_platform("github", "q", kind="issues", session=sess)[0].extra["is_pr"] is True  # type: ignore[arg-type]
    with pytest.raises(AllBackendsFailedError):  # code search needs a token or gh
        search_platform("github", "q", kind="code", session=sess)  # type: ignore[arg-type]
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_" + "c" * 36)
    assert search_platform("github", "q", kind="code", session=sess)[0].title == "a/b: x.py"  # type: ignore[arg-type]


def test_platform_hackernews_and_arxiv():
    hn = {"hits": [{"objectID": "42", "title": "Rust", "url": "https://rust.example", "points": 10, "num_comments": 3}]}
    sess = FakeSession(
        [
            ("GET", "hn.algolia.com/api/v1/search", FakeResp(json_data=hn)),
            ("GET", "export.arxiv.org/api/query", FakeResp(body=ARXIV_ATOM, content_type="application/atom+xml")),
        ]
    )
    hits = search_platform("hn", "rust", session=sess)  # type: ignore[arg-type]
    assert hits[0].extra["hn_url"] == "https://news.ycombinator.com/item?id=42"
    assert sess.calls[0][2]["params"]["tags"] == "story"
    papers = search_platform("arxiv", "attention", max_results=1, session=sess)  # type: ignore[arg-type]
    assert papers[0].title == "Attention Is All You Need"
    assert sess.calls[1][2]["params"]["search_query"] == "all:attention"


def test_platform_unknown():
    with pytest.raises(ValueError, match="Unknown platform"):
        search_platform("myspace", "q")


# ---------------------------------------------------------------------------
# Toolkit, named namespace, health
# ---------------------------------------------------------------------------


def test_toolkit_tool_names_and_media_skip(monkeypatch):
    monkeypatch.setitem(sys.modules, "prompture.media.understand.tools", None)  # ImportError
    names = [t.name for t in WebToolkit().tools()]
    assert names == ["web_search", "web_fetch", "read_url", "search_platform"]
    td = WebToolkit(include=["platform"]).tools()[0]
    assert td.parameters["properties"]["platform"]["enum"] == ["youtube", "github", "hackernews", "arxiv"]


def test_toolkit_adds_media_tools_when_importable(monkeypatch):
    from prompture.agents.tools_schema import ToolDefinition

    fake = types.ModuleType("prompture.media.understand.tools")
    fake.transcribe_media_tool = lambda: ToolDefinition(
        "transcribe_media", "t", {"type": "object", "properties": {}}, lambda: ""
    )  # type: ignore[attr-defined]
    fake.summarize_media_tool = lambda model=None: ToolDefinition(
        "summarize_media", str(model), {"type": "object", "properties": {}}, lambda: ""
    )  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prompture.media.understand.tools", fake)
    tools = WebToolkit(summarize_model="openai/gpt-4o-mini").tools()
    assert [t.name for t in tools][-2:] == ["transcribe_media", "summarize_media"]
    assert tools[-1].description == "openai/gpt-4o-mini"


def test_toolkit_functions_never_raise():
    kit = WebToolkit(include=["fetch", "read", "platform", "search"])
    tools = {t.name: t for t in kit.tools()}
    out = tools["web_fetch"].function(url="http://127.0.0.1/secret")
    assert out.startswith("Error: web fetch failed") and "unsafe_url" in out
    out = tools["read_url"].function(url="file:///etc/passwd")
    assert out.startswith("Error:")
    out = tools["search_platform"].function(platform="nope", query="x")
    assert out.startswith("Error:")
    out = tools["web_search"].function(query="  ")
    assert out.startswith("Error:")


def test_toolkit_register_on_registry():
    from prompture import ToolRegistry

    reg = ToolRegistry()
    tools = WebToolkit(include=["search", "fetch"]).register_on(reg)
    assert [t.name for t in tools] == ["web_search", "web_fetch"]
    assert reg.get("web_fetch") is tools[1]


def test_resolve_web_tools_namespace(monkeypatch):
    monkeypatch.setitem(sys.modules, "prompture.media.understand.tools", None)
    assert [t.name for t in resolve_web_tools("all")] == ["web_search", "web_fetch", "read_url", "search_platform"]
    assert [t.name for t in resolve_web_tools("search+fetch")] == ["web_search", "web_fetch"]
    assert [t.name for t in resolve_web_tools("read")] == ["read_url"]
    assert [t.name for t in resolve_web_tools("platform")] == ["search_platform"]
    with pytest.raises(ValueError):
        resolve_web_tools("bogus")
    from prompture.tools.named import resolve_tool_spec

    assert [t.name for t in resolve_tool_spec("web:search")] == ["web_search"]


def test_health_rows_registered_and_offline(monkeypatch):
    import prompture.tools.web.health as health
    from prompture.capabilities import list_capabilities

    monkeypatch.setattr(_common, "default_session", lambda: Exploding())
    health.register_web_capabilities()
    names = {c.name for c in list_capabilities("tools")}
    assert {
        "web_search",
        "web_fetch",
        "read_url:youtube",
        "read_url:github",
        "read_url:feeds",
        "search_platform:github",
    } <= names
    caps = {c.name: c for c in list_capabilities("tools")}
    search_row = caps["web_search"].run(False)[0]
    assert search_row.status == "ok" and search_row.active_backend == "exa_mcp"
    yt_row = caps["read_url:youtube"].run(False)[0]
    assert yt_row.name == "read_url:youtube"
    assert yt_row.active_backend == "web_fetch"
    assert yt_row.fix_hint and "youtube-transcript-api" in yt_row.fix_hint
    fetch_row = caps["web_fetch"].run(False)[0]
    assert fetch_row.active_backend == "jina_reader"
