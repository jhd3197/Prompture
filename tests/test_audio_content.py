"""Tests for prompture.media.audio — AudioContent and constructor functions."""

from __future__ import annotations

import base64
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from prompture.media.audio import (
    AudioContent,
    AudioInput,
    _guess_media_type,
    _guess_media_type_from_bytes,
    audio_from_base64,
    audio_from_bytes,
    audio_from_file,
    audio_from_url,
    make_audio,
)


# ── AudioContent dataclass ────────────────────────────────────────────────


class TestAudioContent:
    def test_frozen(self):
        ac = AudioContent(data="abc", media_type="audio/mpeg")
        with pytest.raises(AttributeError):
            ac.data = "xyz"  # type: ignore[misc]

    def test_defaults(self):
        ac = AudioContent(data="abc", media_type="audio/wav")
        assert ac.source_type == "base64"
        assert ac.url is None
        assert ac.duration_seconds is None

    def test_url_source(self):
        ac = AudioContent(data="", media_type="audio/mpeg", source_type="url", url="https://example.com/a.mp3")
        assert ac.source_type == "url"
        assert ac.url == "https://example.com/a.mp3"

    def test_duration(self):
        ac = AudioContent(data="abc", media_type="audio/wav", duration_seconds=3.5)
        assert ac.duration_seconds == 3.5


# ── Magic bytes detection ─────────────────────────────────────────────────


class TestMagicBytes:
    def test_mp3_frame_sync(self):
        data = b"\xff\xfb\x90\x00" + b"\x00" * 100
        assert _guess_media_type_from_bytes(data) == "audio/mpeg"

    def test_mp3_id3(self):
        data = b"ID3" + b"\x00" * 100
        assert _guess_media_type_from_bytes(data) == "audio/mpeg"

    def test_ogg(self):
        data = b"OggS" + b"\x00" * 100
        assert _guess_media_type_from_bytes(data) == "audio/ogg"

    def test_flac(self):
        data = b"fLaC" + b"\x00" * 100
        assert _guess_media_type_from_bytes(data) == "audio/flac"

    def test_wav(self):
        # RIFF....WAVE
        data = b"RIFF" + struct.pack("<I", 100) + b"WAVE" + b"\x00" * 100
        assert _guess_media_type_from_bytes(data) == "audio/wav"

    def test_m4a(self):
        # ftyp box at offset 4
        data = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100
        assert _guess_media_type_from_bytes(data) == "audio/mp4"

    def test_unknown_fallback(self):
        data = b"\x00\x00\x00\x00" + b"\x00" * 100
        assert _guess_media_type_from_bytes(data) == "audio/mpeg"


# ── MIME from path/URL ────────────────────────────────────────────────────


class TestGuessMediaType:
    def test_mp3(self):
        assert _guess_media_type("song.mp3") == "audio/mpeg"

    def test_wav(self):
        assert _guess_media_type("recording.wav") == "audio/wav"

    def test_ogg(self):
        assert _guess_media_type("clip.ogg") == "audio/ogg"

    def test_flac(self):
        assert _guess_media_type("lossless.flac") == "audio/flac"

    def test_m4a(self):
        assert _guess_media_type("podcast.m4a") == "audio/mp4"

    def test_opus(self):
        assert _guess_media_type("voice.opus") == "audio/opus"

    def test_url_with_query(self):
        assert _guess_media_type("https://cdn.example.com/clip.wav?token=abc") == "audio/wav"

    def test_unknown_fallback(self):
        assert _guess_media_type("file.xyz") == "audio/mpeg"


# ── Constructor: audio_from_bytes ─────────────────────────────────────────


class TestAudioFromBytes:
    def test_basic(self):
        raw = b"\xff\xfb" + b"\x00" * 50
        ac = audio_from_bytes(raw)
        assert ac.media_type == "audio/mpeg"
        assert ac.source_type == "base64"
        assert base64.b64decode(ac.data) == raw

    def test_explicit_mime(self):
        raw = b"\x00" * 50
        ac = audio_from_bytes(raw, media_type="audio/wav")
        assert ac.media_type == "audio/wav"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            audio_from_bytes(b"")


# ── Constructor: audio_from_base64 ────────────────────────────────────────


class TestAudioFromBase64:
    def test_raw_base64(self):
        b64 = base64.b64encode(b"hello audio").decode()
        ac = audio_from_base64(b64)
        assert ac.data == b64
        assert ac.media_type == "audio/mpeg"

    def test_data_uri(self):
        raw_b64 = base64.b64encode(b"hello audio").decode()
        data_uri = f"data:audio/wav;base64,{raw_b64}"
        ac = audio_from_base64(data_uri)
        assert ac.data == raw_b64
        assert ac.media_type == "audio/wav"

    def test_explicit_mime(self):
        ac = audio_from_base64("AAAA", media_type="audio/ogg")
        assert ac.media_type == "audio/ogg"


# ── Constructor: audio_from_file ──────────────────────────────────────────


class TestAudioFromFile:
    def test_reads_file(self, tmp_path: Path):
        p = tmp_path / "test.mp3"
        content = b"\xff\xfb" + b"\x00" * 50
        p.write_bytes(content)
        ac = audio_from_file(p)
        assert ac.media_type == "audio/mpeg"
        assert base64.b64decode(ac.data) == content

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            audio_from_file("/nonexistent/audio.mp3")

    def test_explicit_mime(self, tmp_path: Path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"\x00" * 50)
        ac = audio_from_file(p, media_type="audio/wav")
        assert ac.media_type == "audio/wav"


# ── Constructor: audio_from_url ───────────────────────────────────────────


class TestAudioFromUrl:
    def test_basic(self):
        ac = audio_from_url("https://example.com/clip.mp3")
        assert ac.source_type == "url"
        assert ac.url == "https://example.com/clip.mp3"
        assert ac.media_type == "audio/mpeg"
        assert ac.data == ""

    def test_explicit_mime(self):
        ac = audio_from_url("https://example.com/clip", media_type="audio/ogg")
        assert ac.media_type == "audio/ogg"


# ── Smart constructor: make_audio ─────────────────────────────────────────


class TestMakeAudio:
    def test_passthrough(self):
        ac = AudioContent(data="abc", media_type="audio/mpeg")
        assert make_audio(ac) is ac

    def test_bytes(self):
        raw = b"\xff\xfb" + b"\x00" * 50
        ac = make_audio(raw)
        assert ac.media_type == "audio/mpeg"

    def test_path(self, tmp_path: Path):
        p = tmp_path / "test.wav"
        p.write_bytes(b"RIFF" + struct.pack("<I", 100) + b"WAVE" + b"\x00" * 100)
        ac = make_audio(p)
        assert ac.media_type == "audio/wav"

    def test_url_string(self):
        ac = make_audio("https://example.com/song.mp3")
        assert ac.source_type == "url"
        assert ac.url == "https://example.com/song.mp3"

    def test_data_uri_string(self):
        b64 = base64.b64encode(b"audio data").decode()
        ac = make_audio(f"data:audio/wav;base64,{b64}")
        assert ac.media_type == "audio/wav"

    def test_file_path_string(self, tmp_path: Path):
        p = tmp_path / "test.flac"
        p.write_bytes(b"fLaC" + b"\x00" * 50)
        ac = make_audio(str(p))
        assert ac.media_type == "audio/flac"

    def test_raw_base64_string(self):
        b64 = base64.b64encode(b"audio data").decode()
        ac = make_audio(b64)
        assert ac.data == b64

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported"):
            make_audio(42)  # type: ignore[arg-type]
