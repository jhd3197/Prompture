"""Tests for the capability foundation: URL guard, safe_get, challenge
detection, backend chains, probes, health registry and credential scrubbing."""

from __future__ import annotations

import sys

import pytest

from prompture.capabilities import (
    AllBackendsFailedError,
    BackendChain,
    BaseBackend,
    ChallengePageError,
    HealthStatus,
    HTTPStatusError,
    ResponseTooLargeError,
    UnsafeURLError,
    check_capabilities,
    is_challenge_page,
    normalize_public_http_url,
    probe_command,
    register_capability,
    resolve_proxy,
    safe_get,
    unregister_capability,
    worst_status,
)
from prompture.capabilities import probe as probe_mod
from prompture.capabilities.backends import order_backends, parse_override
from prompture.capabilities.url_safety import parse_legacy_ipv4
from prompture.resilience import ErrorAction, classify_error
from prompture.security import scrub_secrets, scrub_url_credentials

PUBLIC_IP = "93.184.216.34"

# ---------------------------------------------------------------------------
# URL guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://2130706433/",  # decimal 127.0.0.1
        "http://0177.0.0.1/",  # octal
        "http://0x7f.1/",  # hex, short form
        "http://127.1/",
        "http://10.0.0.5/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://100.64.0.1/",  # CGNAT
        "http://169.254.169.254/latest/meta-data/",  # metadata
        "http://0.0.0.0/",
        "http://224.0.0.1/",  # multicast
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped
        "http://[fe80::1]/",
        "http://[fd00:ec2::254]/",  # AWS IPv6 metadata
        "http://[2002:7f00:1::]/",  # 6to4 around 127.0.0.1
        "http://localhost:8080/",
        "http://foo.localhost/",
        "http://metadata.google.internal/",
        "http://intranet/",
        "ftp://example.com/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "",
    ],
)
def test_unsafe_urls_are_blocked(url):
    with pytest.raises(UnsafeURLError):
        normalize_public_http_url(url)


def test_public_ip_literal_is_normalized():
    out = normalize_public_http_url(f"HTTPS://user:pw@{PUBLIC_IP}:443/a?b=1#frag")
    assert out == f"https://{PUBLIC_IP}/a?b=1"


def test_hostname_resolving_to_private_is_blocked(monkeypatch):
    monkeypatch.setattr("prompture.capabilities.url_safety.resolve_host", lambda host, port=None: ["10.1.2.3"])
    with pytest.raises(UnsafeURLError, match="non-public"):
        normalize_public_http_url("https://rebind.example.com/")


def test_hostname_with_any_private_address_is_blocked(monkeypatch):
    monkeypatch.setattr(
        "prompture.capabilities.url_safety.resolve_host", lambda host, port=None: [PUBLIC_IP, "127.0.0.1"]
    )
    with pytest.raises(UnsafeURLError):
        normalize_public_http_url("https://mixed.example.com/")


def test_public_hostname_passes(monkeypatch):
    monkeypatch.setattr("prompture.capabilities.url_safety.resolve_host", lambda host, port=None: [PUBLIC_IP])
    assert normalize_public_http_url("example.com/x") == "https://example.com/x"


def test_allow_private_escape_hatch(monkeypatch):
    assert normalize_public_http_url("http://127.0.0.1:9000/", allow_private=True) == "http://127.0.0.1:9000/"
    monkeypatch.setenv("PROMPTURE_WEB_ALLOW_PRIVATE", "1")
    assert normalize_public_http_url("http://localhost/") == "http://localhost/"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("2130706433", "127.0.0.1"),
        ("0177.0.0.1", "127.0.0.1"),
        ("0x7f.0.0.1", "127.0.0.1"),
        ("127.1", "127.0.0.1"),
        ("10.0.258", "10.0.1.2"),
        ("example.com", None),
        ("08.0.0.1", None),  # invalid octal
        ("1.2.3.4.5", None),
    ],
)
def test_parse_legacy_ipv4(host, expected):
    ip = parse_legacy_ipv4(host)
    assert (str(ip) if ip else None) == expected


# ---------------------------------------------------------------------------
# safe_get
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, reason="OK"):
        self.status_code = status
        self._body = body
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}
        self.reason = reason
        self.encoding = "utf-8"
        self.closed = False

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses[url]


def test_safe_get_returns_body_and_never_follows_redirects_automatically():
    url = f"http://{PUBLIC_IP}/page"
    session = FakeSession({url: FakeResponse(body=b"<html>hello</html>")})
    resp = safe_get(url, session=session)
    assert resp.text == "<html>hello</html>"
    assert resp.content_type == "text/html"
    assert session.calls[0][1]["allow_redirects"] is False
    assert "Mozilla" in session.calls[0][1]["headers"]["User-Agent"]


def test_safe_get_rechecks_each_redirect_hop():
    start = f"http://{PUBLIC_IP}/start"
    session = FakeSession({start: FakeResponse(302, headers={"location": "http://127.0.0.1/admin"})})
    with pytest.raises(UnsafeURLError, match="redirect"):
        safe_get(start, session=session)
    assert len(session.calls) == 1


def test_safe_get_follows_public_redirect():
    start = f"http://{PUBLIC_IP}/a"
    final = f"http://{PUBLIC_IP}/b"
    session = FakeSession({start: FakeResponse(301, headers={"location": "/b"}), final: FakeResponse(body=b"ok")})
    resp = safe_get(start, session=session)
    assert resp.url == final
    assert resp.history == [start]


def test_safe_get_size_cap():
    url = f"http://{PUBLIC_IP}/big"
    session = FakeSession({url: FakeResponse(body=b"x" * 5000)})
    with pytest.raises(ResponseTooLargeError):
        safe_get(url, session=session, max_bytes=1000)
    resp = safe_get(url, session=session, max_bytes=1000, truncate=True)
    assert resp.truncated and len(resp.content) == 1000


def test_safe_get_declared_length_cap():
    url = f"http://{PUBLIC_IP}/big"
    session = FakeSession({url: FakeResponse(body=b"x", headers={"content-length": "999999"})})
    with pytest.raises(ResponseTooLargeError):
        safe_get(url, session=session, max_bytes=1000)


def test_safe_get_challenge_page():
    url = f"http://{PUBLIC_IP}/cf"
    body = b"<html><head><title>Just a moment...</title></head><script>window._cf_chl_opt={}</script>"
    session = FakeSession({url: FakeResponse(403, body=body)})
    with pytest.raises(ChallengePageError):
        safe_get(url, session=session)


def test_safe_get_http_error_is_classified():
    url = f"http://{PUBLIC_IP}/limited"
    session = FakeSession({url: FakeResponse(429, body=b"slow down", headers={"retry-after": "7"}, reason="Too Many")})
    with pytest.raises(HTTPStatusError) as exc:
        safe_get(url, session=session)
    info = classify_error(exc.value)
    assert info.status_code == 429
    assert info.action == ErrorAction.COOLDOWN
    assert info.retry_after == 7


def test_resolve_proxy(monkeypatch):
    monkeypatch.delenv("PROMPTURE_PROXY", raising=False)
    monkeypatch.delenv("PROMPTURE_JINA_READER_PROXY", raising=False)
    assert resolve_proxy("jina_reader") is None
    monkeypatch.setenv("PROMPTURE_PROXY", "http://proxy:1")
    assert resolve_proxy("jina_reader") == "http://proxy:1"
    monkeypatch.setenv("PROMPTURE_JINA_READER_PROXY", "http://special:2")
    assert resolve_proxy("jina_reader") == "http://special:2"


# ---------------------------------------------------------------------------
# Challenge detection
# ---------------------------------------------------------------------------


def test_challenge_true_positives():
    cf = "<!DOCTYPE html><title>Just a moment...</title><div id='challenge-platform'></div>"
    assert is_challenge_page(cf, status=503)
    assert is_challenge_page(cf, status=200)  # two strong markers
    assert is_challenge_page("anything", headers={"cf-mitigated": "challenge"})
    assert is_challenge_page("Checking your browser before accessing. Verify you are human", status=403)


def test_challenge_false_positives():
    login = "<form><div class='g-recaptcha'></div>Please enable cookies</form>"
    assert not is_challenge_page(login, status=200)
    assert not is_challenge_page("<p>An article about captchas and Cloudflare.</p>", status=200)
    # Markers past the first 4 KB are ignored.
    assert not is_challenge_page("x" * 5000 + "cf_chl_opt challenge-platform", status=403)


def test_capability_errors_classification():
    assert classify_error(UnsafeURLError("x")).action == ErrorAction.FATAL
    assert classify_error(ChallengePageError("x")).action == ErrorAction.FAILOVER
    assert classify_error(ResponseTooLargeError("x")).action == ErrorAction.FAILOVER


# ---------------------------------------------------------------------------
# Backend chains
# ---------------------------------------------------------------------------


class Ok(BaseBackend):
    def __init__(self, name, value=None):
        self.name = name
        self.value = value or name
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        return self.value


class Fails(BaseBackend):
    def __init__(self, name, exc_factory):
        self.name = name
        self.exc_factory = exc_factory
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        raise self.exc_factory()


class Off(BaseBackend):
    requires = ("SOME_KEY",)

    def __init__(self, name):
        self.name = name

    def available(self):
        return False


def test_chain_serves_first_available():
    chain = BackendChain([Off("a"), Ok("b"), Ok("c")], name="t")
    result = chain.run()
    assert result.value == "b"
    assert result.served_by == "b"
    assert result.fallback is False
    assert [a["status"] for a in result.attempts] == ["skipped", "ok"]


def test_chain_fails_over_on_auth_and_quota():
    bad = Fails("bad", lambda: HTTPStatusError("denied", status_code=401))
    quota = Fails("quota", lambda: HTTPStatusError("insufficient_quota", status_code=429))
    good = Ok("good")
    result = BackendChain([bad, quota, good], retry_delay=0).run()
    assert result.served_by == "good"
    assert result.route["fallback"] is True
    assert bad.calls == 1 and quota.calls == 1
    categories = [a.get("category") for a in result.attempts if a["status"] == "error"]
    assert categories == ["auth", "quota_exhausted"]


def test_chain_retries_transient_once():
    flaky = Fails("flaky", lambda: HTTPStatusError("boom", status_code=502))
    result = BackendChain([flaky, Ok("next")], retry_delay=0).run()
    assert flaky.calls == 2
    assert result.served_by == "next"


def test_chain_fatal_stops_and_reraises_original():
    unsafe = Fails("first", lambda: UnsafeURLError("private"))
    second = Ok("second")
    with pytest.raises(UnsafeURLError) as exc:
        BackendChain([unsafe, second]).run()
    assert second.calls == 0
    assert exc.value.attempts[0]["backend"] == "first"


def test_chain_all_failed_and_all_unavailable():
    with pytest.raises(AllBackendsFailedError) as exc:
        BackendChain([Fails("x", lambda: HTTPStatusError("no", status_code=403))], name="search").run()
    assert "all backends failed" in str(exc.value)
    assert exc.value.attempts[0]["category"] == "permission"

    with pytest.raises(AllBackendsFailedError, match="SOME_KEY"):
        BackendChain([Off("y")], name="search").run()


def test_chain_override_env_reorders_and_ignores_unknown(monkeypatch):
    a, b, c = Ok("a"), Ok("b"), Ok("c")
    chain = BackendChain([a, b, c], override_env="TEST_CHAIN_ORDER")
    monkeypatch.setenv("TEST_CHAIN_ORDER", "c, nope ,b")
    assert [x.name for x in chain.ordered()] == ["c", "b", "a"]
    assert chain.run().served_by == "c"
    monkeypatch.setenv("TEST_CHAIN_ORDER", "stale-name")
    assert [x.name for x in chain.ordered()] == ["a", "b", "c"]


def test_chain_only_restricts():
    chain = BackendChain([Ok("a"), Ok("b")])
    assert chain.run(only=["b"]).served_by == "b"
    with pytest.raises(AllBackendsFailedError):
        chain.run(only=["missing"])


def test_parse_and_order_helpers():
    assert parse_override(" A; b ,c ") == ["a", "b", "c"]
    assert parse_override(None) == []
    items = [Ok("x"), Ok("y"), Ok("z")]
    assert [i.name for i in order_backends(items, ["z", "x"])] == ["z", "x", "y"]


async def test_chain_arun():
    chain = BackendChain([Fails("bad", lambda: HTTPStatusError("nope", status_code=401)), Ok("good")])
    result = await chain.arun()
    assert result.served_by == "good"
    assert result.fallback


def test_chain_check_reports_active_backend():
    row = BackendChain([Off("keyed"), Ok("free")], name="web_search").check()
    assert row.status == "ok"
    assert row.active_backend == "free"
    assert "SOME_KEY" in (row.fix_hint or "")
    dead = BackendChain([Off("keyed")], name="web_search").check()
    assert dead.status == "unconfigured"


def test_chain_error_strings_are_scrubbed():
    leaky = Fails("leaky", lambda: RuntimeError("failed https://bob:secret@api.example.com/?token=abc"))
    result = BackendChain([leaky, Ok("ok")]).run()
    err = result.attempts[0]["error"]
    assert "secret" not in err and "abc" not in err


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _fake_which(monkeypatch, path):
    monkeypatch.setattr(probe_mod.shutil, "which", lambda cmd: path)


def test_probe_missing(monkeypatch):
    _fake_which(monkeypatch, None)
    result = probe_command("ffmpeg")
    assert result.status == "missing"
    assert "ffmpeg" in result.hint


def test_probe_ok(monkeypatch):
    _fake_which(monkeypatch, sys.executable)
    result = probe_command("fake", ["-c", "print('fake 1.2.3')"])
    assert result.status == "ok"
    assert result.version == "fake 1.2.3"


def test_probe_stale_shim_is_broken(monkeypatch):
    _fake_which(monkeypatch, sys.executable)
    script = "import sys; sys.stderr.write('No Python at C:\\\\old\\\\python.exe'); sys.exit(101)"
    result = probe_command("yt-dlp", ["-c", script])
    assert result.status == "broken"
    assert "reinstall" in result.hint


def test_probe_exit_127_is_broken(monkeypatch):
    _fake_which(monkeypatch, sys.executable)
    result = probe_command("tool", ["-c", "import sys; sys.exit(127)"])
    assert result.status == "broken"


def test_probe_exec_failure_is_broken(monkeypatch, tmp_path):
    _fake_which(monkeypatch, str(tmp_path / "does-not-exist.exe"))
    result = probe_command("ghost")
    assert result.status == "broken"


def test_probe_nonzero_is_error(monkeypatch):
    _fake_which(monkeypatch, sys.executable)
    result = probe_command("tool", ["-c", "import sys; print('usage'); sys.exit(2)"])
    assert result.status == "error"
    assert result.exit_code == 2


def test_probe_timeout(monkeypatch):
    _fake_which(monkeypatch, sys.executable)
    result = probe_command("slow", ["-c", "import time; time.sleep(5)"], timeout=0.5)
    assert result.status == "timeout"


def test_probe_strips_inherited_env(monkeypatch):
    _fake_which(monkeypatch, sys.executable)
    monkeypatch.setenv("PYTHONSTARTUP", "/nonexistent/startup.py")
    monkeypatch.setenv("LEAKY_VAR", "1")
    script = (
        "import os; print(os.environ.get('PYTHONSTARTUP'), os.environ.get('LEAKY_VAR'), os.environ['PYTHONIOENCODING'])"
    )
    result = probe_command("py", ["-c", script], strip_env=("PYTHONSTARTUP", "LEAKY_VAR"))
    assert result.output == "None None utf-8"


# ---------------------------------------------------------------------------
# Health registry
# ---------------------------------------------------------------------------


def test_health_registry_and_crashing_check():
    register_capability("test_ok", "tools", lambda live: HealthStatus("test_ok", "ok"))

    def boom(live):
        raise RuntimeError("kaboom")

    register_capability("test_boom", "media", boom)
    try:
        rows = {r.name: r for r in check_capabilities(only=["tools", "media"])}
        assert rows["test_ok"].category == "tools"
        assert rows["test_boom"].status == "error"
        assert rows["test_boom"].category == "media"
        assert worst_status([rows["test_ok"], rows["test_boom"]]) == "error"
    finally:
        unregister_capability("test_ok")
        unregister_capability("test_boom")


def test_health_status_scrubs_message():
    row = HealthStatus("x", "error", message="GET https://u:p@h.example.com/?api_key=sekrit failed")
    assert "sekrit" not in row.message and "u:p@" not in row.message


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "leaked"),
    [
        ("https://bob:hunter2@example.com/x", "hunter2"),
        ("proxy admin:s3cret@10.0.0.1:8080 refused", "s3cret"),
        ("https://api.example.com/v1?q=1&api_key=abc123", "abc123"),
        ("https://example.com/cb#access_token=tok123&state=x", "tok123"),
        ("https://b.s3.amazonaws.com/o?X-Amz-Signature=deadbeef", "deadbeef"),
        ("https://example.com/?session=sess42", "sess42"),
        ("https://example.com/?sig=s1g", "s1g"),
    ],
)
def test_scrub_url_credentials(raw, leaked):
    assert leaked not in scrub_url_credentials(raw)


def test_scrub_keeps_harmless_text():
    text = "Contact user@example.com at 10:30, see https://example.com/a?page=2&q=key+words"
    assert scrub_url_credentials(text) == text


def test_scrub_secrets_api_keys_and_bearer():
    out = scrub_secrets("Authorization: Bearer abcdefghijklmnop key=sk-" + "a" * 30)
    assert "abcdefghijklmnop" not in out
    assert "sk-aaaa" not in out


def test_driver_http_error_message_is_scrubbed():
    from prompture.drivers.base import DriverHTTPError

    err = DriverHTTPError("POST https://u:pw@api.example.com/v1?api_key=zzz failed", status_code=500)
    assert "pw@" not in str(err) and "zzz" not in str(err)


def test_logging_filter_scrubs(caplog):
    import logging

    from prompture.infra.logging import SecretScrubbingFilter

    record = logging.LogRecord("prompture", logging.INFO, __file__, 1, "fetch %s", ("https://a:b@x.example.com",), None)
    SecretScrubbingFilter().filter(record)
    assert "a:b@" not in record.getMessage()
