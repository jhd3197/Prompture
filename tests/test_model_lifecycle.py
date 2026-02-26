"""Tests for model lifecycle/deprecation heuristics (prompture.infra.model_rates)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from prompture.infra.model_rates import (
    _compute_family_status,
    _lifecycle_cache,
    _lifecycle_lock,
    get_model_lifecycle,
)


def _make_entry(
    *,
    name: str = "",
    family: str = "",
    release_date: str | None = None,
    status: str | None = None,
) -> dict:
    """Build a minimal models.dev-style entry dict."""
    entry: dict = {"name": name, "family": family}
    if release_date is not None:
        entry["release_date"] = release_date
    if status is not None:
        entry["status"] = status
    return entry


def _date_ago(months: int) -> str:
    """Return an ISO date string approximately *months* ago from today."""
    d = datetime.now(timezone.utc) - timedelta(days=months * 30)
    return d.strftime("%Y-%m-%d")


def _build_provider_data(models: dict) -> dict:
    """Wrap a models dict into the models.dev provider structure."""
    return {"test_provider": {"models": models}}


class TestComputeFamilyStatus:
    """Unit tests for _compute_family_status heuristics."""

    def test_latest_marker_identifies_current(self):
        """A model with '(latest)' in its name is flagged as current."""
        models = {
            "claude-3-haiku": _make_entry(
                name="Claude 3 Haiku",
                family="claude-haiku",
                release_date=_date_ago(12),
            ),
            "claude-haiku-4-5": _make_entry(
                name="Claude Haiku 4.5 (latest)",
                family="claude-haiku",
                release_date=_date_ago(2),
            ),
        }
        data = _build_provider_data(models)
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = _compute_family_status("test_provider")

        assert result["claude-haiku-4-5"]["status"] == "current"

    def test_most_recent_release_is_current(self):
        """Without a (latest) marker, the most recently released model is current."""
        models = {
            "gpt-4o-mini": _make_entry(
                name="GPT-4o Mini",
                family="gpt-4o",
                release_date=_date_ago(1),
            ),
            "gpt-4o": _make_entry(
                name="GPT-4o",
                family="gpt-4o",
                release_date=_date_ago(8),
            ),
        }
        data = _build_provider_data(models)
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = _compute_family_status("test_provider")

        assert result["gpt-4o-mini"]["status"] == "current"

    def test_older_sibling_marked_legacy(self):
        """A model 6-18 months old with a newer sibling is 'legacy'."""
        models = {
            "model-v2": _make_entry(
                name="Model v2",
                family="model",
                release_date=_date_ago(1),
            ),
            "model-v1": _make_entry(
                name="Model v1",
                family="model",
                release_date=_date_ago(10),
            ),
        }
        data = _build_provider_data(models)
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = _compute_family_status("test_provider")

        assert result["model-v1"]["status"] == "legacy"
        assert result["model-v2"]["status"] == "current"

    def test_old_sibling_marked_deprecated(self):
        """A model >18 months old with a newer sibling is 'deprecated'."""
        models = {
            "model-v2": _make_entry(
                name="Model v2",
                family="model",
                release_date=_date_ago(1),
            ),
            "model-v1": _make_entry(
                name="Model v1",
                family="model",
                release_date=_date_ago(20),
            ),
        }
        data = _build_provider_data(models)
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = _compute_family_status("test_provider")

        assert result["model-v1"]["status"] == "deprecated"

    def test_explicit_status_from_models_dev_respected(self):
        """When models.dev provides status='deprecated', it overrides heuristics."""
        models = {
            "model-v2": _make_entry(
                name="Model v2",
                family="model",
                release_date=_date_ago(1),
            ),
            "model-v1": _make_entry(
                name="Model v1",
                family="model",
                release_date=_date_ago(3),  # only 3 months old — heuristic would say "current"
                status="deprecated",
            ),
        }
        data = _build_provider_data(models)
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = _compute_family_status("test_provider")

        assert result["model-v1"]["status"] == "deprecated"

    def test_superseded_by_chain(self):
        """Older models point superseded_by to the next newer sibling."""
        models = {
            "model-v3": _make_entry(
                name="Model v3",
                family="model",
                release_date=_date_ago(1),
            ),
            "model-v2": _make_entry(
                name="Model v2",
                family="model",
                release_date=_date_ago(10),
            ),
            "model-v1": _make_entry(
                name="Model v1",
                family="model",
                release_date=_date_ago(20),
            ),
        }
        data = _build_provider_data(models)
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = _compute_family_status("test_provider")

        assert result["model-v3"]["status"] == "current"
        assert result["model-v3"]["superseded_by"] is None
        assert result["model-v2"]["superseded_by"] == "model-v3"
        assert result["model-v1"]["superseded_by"] == "model-v2"

    def test_single_model_family_is_current(self):
        """A family with only one model is always 'current'."""
        models = {
            "solo-model": _make_entry(
                name="Solo Model",
                family="solo",
                release_date=_date_ago(24),
            ),
        }
        data = _build_provider_data(models)
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = _compute_family_status("test_provider")

        assert result["solo-model"]["status"] == "current"
        assert result["solo-model"]["superseded_by"] is None

    def test_end_of_support_estimated(self):
        """Legacy/deprecated models get end_of_support = release_date + ~24 months."""
        models = {
            "model-v2": _make_entry(
                name="Model v2",
                family="model",
                release_date=_date_ago(1),
            ),
            "model-v1": _make_entry(
                name="Model v1",
                family="model",
                release_date="2023-06-15",
            ),
        }
        data = _build_provider_data(models)
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = _compute_family_status("test_provider")

        assert result["model-v1"]["end_of_support"] is not None
        assert result["model-v2"]["end_of_support"] is None  # current model


class TestGetModelLifecycle:
    """Tests for the public get_model_lifecycle() API."""

    def setup_method(self):
        """Clear lifecycle cache before each test."""
        with _lifecycle_lock:
            _lifecycle_cache.clear()

    def test_returns_lifecycle_for_known_model(self):
        """get_model_lifecycle returns a dict for known models."""
        models = {
            "gpt-4o": _make_entry(
                name="GPT-4o",
                family="gpt-4o",
                release_date=_date_ago(3),
            ),
        }
        data = {"openai": {"models": models}}
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = get_model_lifecycle("openai", "gpt-4o")

        assert result is not None
        assert result["status"] == "current"
        assert result["family"] == "gpt-4o"

    def test_returns_none_for_unknown_model(self):
        """get_model_lifecycle returns None for models not in models.dev."""
        data = {"openai": {"models": {}}}
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = get_model_lifecycle("openai", "nonexistent-model")

        assert result is None

    def test_alias_models_inherit_status(self):
        """Date-versioned models fall back to the base model's lifecycle."""
        models = {
            "claude-sonnet-4": _make_entry(
                name="Claude Sonnet 4",
                family="claude-sonnet",
                release_date=_date_ago(2),
            ),
        }
        data = {"anthropic": {"models": models}}
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            # Query with date suffix that strips to "claude-sonnet-4"
            result = get_model_lifecycle("claude", "claude-sonnet-4-20250514")

        assert result is not None
        assert result["status"] == "current"
        assert result["family"] == "claude-sonnet"

    def test_provider_mapping(self):
        """The prompture provider name is mapped to models.dev API name."""
        models = {
            "grok-2": _make_entry(
                name="Grok 2",
                family="grok",
                release_date=_date_ago(4),
            ),
        }
        data = {"xai": {"models": models}}
        with patch("prompture.infra.model_rates._ensure_loaded", return_value=data):
            result = get_model_lifecycle("grok", "grok-2")

        assert result is not None
        assert result["family"] == "grok"
