"""Update awareness: version comparison, PyPI/GitHub parsing, 24h cache, offline, announce-once."""

from __future__ import annotations

import json

import pytest

from prompture.infra import updates
from prompture.infra.updates import (
    PYPI_URL,
    RELEASES_URL,
    UpdateInfo,
    changelog_since,
    check_for_update,
    compare_versions,
    is_newer,
    mark_announced,
    release_notes,
    should_announce,
)


def _pypi(version: str, releases: dict | None = None) -> dict:
    return {"info": {"version": version}, "releases": releases if releases is not None else {version: [{}]}}


RELEASES = [
    {
        "tag_name": "v1.3.0",
        "published_at": "2026-09-30T10:00:00Z",
        "html_url": "https://github.com/x/y/releases/tag/v1.3.0",
        "body": "## What's new\n- Add **doctor** command\n- Fix [proxy](https://x.y) handling\n- Version: v1.3.0\n",
    },
    {"tag_name": "v1.3.1rc1", "prerelease": True, "body": "- rc"},
    {"tag_name": "v1.2.0", "published_at": "2026-09-01T00:00:00Z", "body": "Plain paragraph note."},
    {"tag_name": "v1.1.0", "body": "- already installed"},
    {"tag_name": "v1.4.0", "draft": True, "body": "- draft"},
]


class FakeFetch:
    def __init__(self, pypi=None, releases=None, fail: Exception | None = None):
        self.pypi, self.releases, self.fail = pypi, releases, fail
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float):
        self.calls.append(url)
        if self.fail:
            raise self.fail
        if url == PYPI_URL:
            return self.pypi
        if url == RELEASES_URL:
            if isinstance(self.releases, Exception):
                raise self.releases
            return self.releases
        raise AssertionError(url)


@pytest.fixture
def state(tmp_path):
    return tmp_path / "update_check.json"


class TestVersions:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("1.2.0", "1.10.0", -1),
            ("1.10.0", "1.2.0", 1),
            ("1.2", "1.2.0", 0),
            ("v1.2.0", "1.2.0", 0),
            ("1.2.0rc1", "1.2.0", -1),
            ("1.2.1.dev3+g1234", "1.2.0", 1),
            ("1.2.1.dev3", "1.2.1", -1),
        ],
    )
    def test_compare(self, a, b, expected):
        assert compare_versions(a, b) == expected

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [("1.2.0", "1.10.0", -1), ("1.2", "1.2.0", 0), ("1.2.0rc1", "1.2.0", -1), ("1.2.0.post1", "1.2.0", 1)],
    )
    def test_fallback_parser_without_packaging(self, a, b, expected):
        ka, kb = updates._fallback_key(a), updates._fallback_key(b)
        assert ((ka > kb) - (ka < kb)) == expected

    def test_is_newer(self):
        assert is_newer("1.13.4", "1.12.5.dev1")
        assert not is_newer("1.13.0", "1.13.0")


class TestChangelog:
    def test_release_notes_prefers_bullets_and_drops_boilerplate(self):
        assert release_notes(RELEASES[0]["body"]) == ["Add doctor command", "Fix proxy handling"]

    def test_release_notes_falls_back_to_paragraphs(self):
        assert release_notes("# Title\n\nPlain paragraph note.") == ["Plain paragraph note."]
        assert release_notes(None) == []

    def test_changelog_since_filters_and_orders(self):
        out = changelog_since(RELEASES, "1.1.0", "1.3.0")
        assert [r["version"] for r in out] == ["1.3.0", "1.2.0"]  # no draft, no pre-release, nothing installed
        assert out[0]["date"] == "2026-09-30" and out[0]["url"].endswith("v1.3.0")

    def test_changelog_since_tolerates_garbage(self):
        assert changelog_since({"message": "rate limited"}, "1.0", "2.0") == []
        assert changelog_since([None, {"tag_name": "latest"}], "1.0", "2.0") == []


class TestCheckForUpdate:
    def test_update_available_with_highlights(self, state):
        fetch = FakeFetch(_pypi("1.3.0"), RELEASES)
        info = check_for_update(installed="1.1.0", state_path=state, fetch_json=fetch, now=1000.0)
        assert info.update_available and info.latest == "1.3.0" and info.source == "network"
        assert [h["version"] for h in info.highlights] == ["1.3.0", "1.2.0"]
        assert info.upgrade_command == "pip install -U prompture"
        assert "1.3.0 is available" in info.summary()
        assert fetch.calls == [PYPI_URL, RELEASES_URL]

    def test_up_to_date_skips_changelog(self, state):
        fetch = FakeFetch(_pypi("1.3.0"), RELEASES)
        info = check_for_update(installed="1.3.0", state_path=state, fetch_json=fetch)
        assert not info.update_available and info.highlights == []
        assert fetch.calls == [PYPI_URL]
        assert "up to date" in info.summary()

    def test_prereleases_and_yanked_are_ignored(self, state):
        data = _pypi("2.0.0b1", {"1.3.0": [{}], "1.4.0": [{"yanked": True}], "2.0.0b1": [{}]})
        info = check_for_update(installed="1.1.0", state_path=state, fetch_json=FakeFetch(data, []))
        assert info.latest == "1.3.0"
        info = check_for_update(
            installed="1.1.0", state_path=state, fetch_json=FakeFetch(data, []), include_prereleases=True, force=True
        )
        assert info.latest == "2.0.0b1"

    def test_cached_for_a_day(self, state):
        fetch = FakeFetch(_pypi("1.3.0"), RELEASES)
        check_for_update(installed="1.1.0", state_path=state, fetch_json=fetch, now=1000.0)
        again = check_for_update(installed="1.1.0", state_path=state, fetch_json=fetch, now=1000.0 + 3600)
        assert again.source == "cache" and again.latest == "1.3.0" and len(again.highlights) == 2
        assert fetch.calls == [PYPI_URL, RELEASES_URL]  # no second call
        stale = check_for_update(installed="1.1.0", state_path=state, fetch_json=fetch, now=1000.0 + 25 * 3600)
        assert stale.source == "network" and len(fetch.calls) == 4

    def test_force_and_new_install_bypass_cache(self, state):
        fetch = FakeFetch(_pypi("1.3.0"), RELEASES)
        check_for_update(installed="1.1.0", state_path=state, fetch_json=fetch, now=1000.0)
        check_for_update(installed="1.1.0", state_path=state, fetch_json=fetch, now=1001.0, force=True)
        check_for_update(installed="1.3.0", state_path=state, fetch_json=fetch, now=1002.0)
        assert fetch.calls.count(PYPI_URL) == 3

    def test_offline_without_cache(self, state):
        info = check_for_update(installed="1.1.0", state_path=state, fetch_json=FakeFetch(fail=OSError("no route")))
        assert info.source == "offline" and info.latest is None and not info.update_available
        assert "no route" in info.error and "unknown" in info.summary()
        assert not state.exists()

    def test_offline_falls_back_to_stale_cache(self, state):
        check_for_update(installed="1.1.0", state_path=state, fetch_json=FakeFetch(_pypi("1.3.0"), []), now=1000.0)
        info = check_for_update(
            installed="1.1.0", state_path=state, fetch_json=FakeFetch(fail=OSError("down")), now=1000.0 + 90000
        )
        assert info.source == "cache" and info.latest == "1.3.0" and info.update_available and "down" in info.error

    def test_changelog_failure_is_best_effort(self, state):
        info = check_for_update(
            installed="1.1.0", state_path=state, fetch_json=FakeFetch(_pypi("1.3.0"), RuntimeError("403"))
        )
        assert info.update_available and info.highlights == []

    def test_bad_pypi_payload(self, state):
        info = check_for_update(installed="1.1.0", state_path=state, fetch_json=FakeFetch({"info": {}}, []))
        assert info.source == "offline" and info.error

    def test_cache_file_is_json_and_atomic(self, state):
        check_for_update(installed="1.1.0", state_path=state, fetch_json=FakeFetch(_pypi("1.3.0"), []), now=5.0)
        data = json.loads(state.read_text(encoding="utf-8"))
        assert data["latest"] == "1.3.0" and data["installed"] == "1.1.0" and data["checked_at"] == 5.0
        assert [p.name for p in state.parent.iterdir()] == [state.name]  # no temp files left behind

    def test_to_dict_schema(self, state):
        data = UpdateInfo(installed="1.0", latest="1.1", update_available=True).to_dict()
        assert data["schema"] == "prompture.update/1"
        assert {"installed", "latest", "update_available", "checked_at", "source", "error", "highlights"} <= set(data)


class TestAnnounceOnce:
    def test_mention_once_per_version(self, state):
        assert should_announce("1.3.0", state_path=state)
        mark_announced("1.3.0", state_path=state)
        assert not should_announce("1.3.0", state_path=state)
        assert should_announce("1.4.0", state_path=state)
        assert not should_announce(None, state_path=state)

    def test_announce_state_survives_update_checks(self, state):
        mark_announced("1.3.0", state_path=state)
        check_for_update(installed="1.1.0", state_path=state, fetch_json=FakeFetch(_pypi("1.3.0"), []))
        assert not should_announce("1.3.0", state_path=state)
        mark_announced("1.3.0", state_path=state)
        assert json.loads(state.read_text(encoding="utf-8"))["announced"] == ["1.3.0"]
