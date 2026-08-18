"""Tests for prompture.media.hosting (file hosting + resolve-to-bytes)."""

from __future__ import annotations

import base64

import pytest

from prompture.media.hosting import (
    InMemoryHost,
    LocalDiskHost,
    content_hash,
    host_media,
    resolve_to_bytes,
    save_media,
)


class TestContentHash:
    def test_stable_and_short(self):
        assert content_hash(b"abc") == content_hash(b"abc")
        assert content_hash(b"abc") != content_hash(b"abd")
        assert len(content_hash(b"abc")) == 16


class TestInMemoryHost:
    def test_put_get_roundtrip(self):
        h = InMemoryHost()
        url = h.put(b"payload", media_type="image/png")
        assert url.startswith("memory://")
        assert h.get(url) == b"payload"

    def test_resolve_to_bytes_via_host(self):
        h = InMemoryHost()
        url = h.put(b"data123")
        assert resolve_to_bytes(url, host=h) == b"data123"

    def test_resolve_memory_without_host_raises(self):
        with pytest.raises(ValueError):
            resolve_to_bytes("memory://deadbeef")


class TestLocalDiskHost:
    def test_put_returns_file_uri_and_reads_back(self, tmp_path):
        h = LocalDiskHost(tmp_path)
        url = h.put(b"file-bytes", media_type="image/png")
        assert url.startswith("file://")
        assert resolve_to_bytes(url) == b"file-bytes"

    def test_base_url_mode(self, tmp_path):
        h = LocalDiskHost(tmp_path, base_url="https://cdn.example.com/media")
        url = h.put(b"x", filename="pic.png")
        assert url == "https://cdn.example.com/media/pic.png"


class TestResolveToBytes:
    def test_bytes_passthrough(self):
        assert resolve_to_bytes(b"raw") == b"raw"

    def test_path_object(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"onsdisk")
        assert resolve_to_bytes(p) == b"onsdisk"

    def test_data_uri_base64(self):
        b64 = base64.b64encode(b"hello").decode()
        assert resolve_to_bytes(f"data:image/png;base64,{b64}") == b"hello"

    def test_data_uri_plain(self):
        assert resolve_to_bytes("data:text/plain,hello") == b"hello"

    def test_raw_base64_fallback(self):
        b64 = base64.b64encode(b"zzz").decode()
        assert resolve_to_bytes(b64) == b"zzz"

    def test_local_path_string(self, tmp_path):
        p = tmp_path / "g.bin"
        p.write_bytes(b"pathstr")
        assert resolve_to_bytes(str(p)) == b"pathstr"


class TestSaveAndHostMedia:
    def test_save_media(self, tmp_path):
        out = save_media(b"persisted", tmp_path / "nested" / "out.png")
        assert out.read_bytes() == b"persisted"

    def test_host_media_with_explicit_host(self):
        h = InMemoryHost()
        url = host_media(b"hm", host=h, media_type="image/png")
        assert resolve_to_bytes(url, host=h) == b"hm"
