"""Tests for prompture.infra.cache module."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from prompture.infra.cache import (
    MemoryCacheBackend,
    ResponseCache,
    SQLiteCacheBackend,
    _reset_cache,
    configure_cache,
    get_cache,
    make_cache_key,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the module-level singleton is clean for every test."""
    _reset_cache()
    yield
    _reset_cache()


# ---------------------------------------------------------------------------
# TestMakeCacheKey
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    """Tests for the cache key derivation function."""

    def test_deterministic(self):
        """Same inputs always produce the same key."""
        k1 = make_cache_key("hello", "openai/gpt-4", {"type": "object"}, {"temperature": 0.5})
        k2 = make_cache_key("hello", "openai/gpt-4", {"type": "object"}, {"temperature": 0.5})
        assert k1 == k2

    def test_different_prompt_different_key(self):
        k1 = make_cache_key("hello", "openai/gpt-4")
        k2 = make_cache_key("world", "openai/gpt-4")
        assert k1 != k2

    def test_different_model_different_key(self):
        k1 = make_cache_key("hello", "openai/gpt-4")
        k2 = make_cache_key("hello", "claude/claude-3")
        assert k1 != k2

    def test_different_schema_different_key(self):
        k1 = make_cache_key("hello", "m", schema={"type": "object"})
        k2 = make_cache_key("hello", "m", schema={"type": "array"})
        assert k1 != k2

    def test_irrelevant_options_ignored(self):
        """Options not in the allowlist (e.g. 'model') should not affect the key."""
        k1 = make_cache_key("hello", "m", options={"temperature": 0.7, "model": "x"})
        k2 = make_cache_key("hello", "m", options={"temperature": 0.7, "model": "y"})
        assert k1 == k2

    def test_relevant_options_affect_key(self):
        k1 = make_cache_key("hello", "m", options={"temperature": 0.7})
        k2 = make_cache_key("hello", "m", options={"temperature": 0.9})
        assert k1 != k2

    def test_option_order_independence(self):
        k1 = make_cache_key("p", "m", options={"temperature": 0.5, "max_tokens": 100})
        k2 = make_cache_key("p", "m", options={"max_tokens": 100, "temperature": 0.5})
        assert k1 == k2

    def test_pydantic_qualname_included(self):
        k1 = make_cache_key("p", "m", pydantic_qualname="Foo")
        k2 = make_cache_key("p", "m", pydantic_qualname="Bar")
        assert k1 != k2

    def test_pydantic_qualname_none_excluded(self):
        k1 = make_cache_key("p", "m")
        k2 = make_cache_key("p", "m", pydantic_qualname=None)
        assert k1 == k2

    def test_output_format_affects_key(self):
        k1 = make_cache_key("p", "m", output_format="json")
        k2 = make_cache_key("p", "m", output_format="toon")
        assert k1 != k2


# ---------------------------------------------------------------------------
# TestMemoryCacheBackend
# ---------------------------------------------------------------------------


class TestMemoryCacheBackend:
    """Tests for the in-memory LRU cache backend."""

    def test_round_trip(self):
        be = MemoryCacheBackend()
        be.set("k", {"data": 1})
        assert be.get("k") == {"data": 1}

    def test_miss_returns_none(self):
        be = MemoryCacheBackend()
        assert be.get("nonexistent") is None

    def test_has(self):
        be = MemoryCacheBackend()
        assert be.has("k") is False
        be.set("k", "v")
        assert be.has("k") is True

    def test_delete(self):
        be = MemoryCacheBackend()
        be.set("k", "v")
        be.delete("k")
        assert be.get("k") is None

    def test_clear(self):
        be = MemoryCacheBackend()
        be.set("a", 1)
        be.set("b", 2)
        be.clear()
        assert be.get("a") is None
        assert be.get("b") is None

    def test_ttl_expiration(self):
        be = MemoryCacheBackend()
        be.set("k", "v", ttl=1)
        assert be.get("k") == "v"
        # Simulate time passing
        with patch("prompture.infra.cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 2
            assert be.get("k") is None

    def test_lru_eviction(self):
        be = MemoryCacheBackend(maxsize=2)
        be.set("a", 1)
        be.set("b", 2)
        be.set("c", 3)  # should evict "a"
        assert be.get("a") is None
        assert be.get("b") == 2
        assert be.get("c") == 3

    def test_access_refreshes_lru(self):
        be = MemoryCacheBackend(maxsize=2)
        be.set("a", 1)
        be.set("b", 2)
        be.get("a")  # refresh "a", "b" becomes LRU
        be.set("c", 3)  # should evict "b"
        assert be.get("a") == 1
        assert be.get("b") is None
        assert be.get("c") == 3

    def test_thread_safety(self):
        be = MemoryCacheBackend(maxsize=1000)
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    be.set(f"{n}-{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# TestSQLiteCacheBackend
# ---------------------------------------------------------------------------


class TestSQLiteCacheBackend:
    """Tests for the SQLite cache backend."""

    def test_round_trip(self, tmp_path):
        db = str(tmp_path / "test.db")
        be = SQLiteCacheBackend(db_path=db)
        be.set("k", {"data": 1})
        assert be.get("k") == {"data": 1}

    def test_miss_returns_none(self, tmp_path):
        db = str(tmp_path / "test.db")
        be = SQLiteCacheBackend(db_path=db)
        assert be.get("missing") is None

    def test_has(self, tmp_path):
        db = str(tmp_path / "test.db")
        be = SQLiteCacheBackend(db_path=db)
        assert be.has("k") is False
        be.set("k", "v")
        assert be.has("k") is True

    def test_delete(self, tmp_path):
        db = str(tmp_path / "test.db")
        be = SQLiteCacheBackend(db_path=db)
        be.set("k", "v")
        be.delete("k")
        assert be.get("k") is None

    def test_clear(self, tmp_path):
        db = str(tmp_path / "test.db")
        be = SQLiteCacheBackend(db_path=db)
        be.set("a", 1)
        be.set("b", 2)
        be.clear()
        assert be.get("a") is None
        assert be.get("b") is None

    def test_ttl_expiration(self, tmp_path):
        db = str(tmp_path / "test.db")
        be = SQLiteCacheBackend(db_path=db)
        be.set("k", "v", ttl=1)
        assert be.get("k") == "v"
        # Simulate expiration by patching time
        with patch("prompture.infra.cache.time") as mock_time:
            mock_time.time.return_value = time.time() + 2
            assert be.get("k") is None

    def test_persistence_across_instances(self, tmp_path):
        db = str(tmp_path / "test.db")
        be1 = SQLiteCacheBackend(db_path=db)
        be1.set("k", {"persisted": True})
        # New instance, same file
        be2 = SQLiteCacheBackend(db_path=db)
        assert be2.get("k") == {"persisted": True}

    def test_overwrite(self, tmp_path):
        db = str(tmp_path / "test.db")
        be = SQLiteCacheBackend(db_path=db)
        be.set("k", "old")
        be.set("k", "new")
        assert be.get("k") == "new"


# ---------------------------------------------------------------------------
# TestResponseCache
# ---------------------------------------------------------------------------


class TestResponseCache:
    """Tests for the ResponseCache orchestrator."""

    def test_disabled_always_misses(self):
        be = MemoryCacheBackend()
        be.set("k", "v")
        rc = ResponseCache(backend=be, enabled=False)
        assert rc.get("k") is None

    def test_disabled_does_not_store(self):
        be = MemoryCacheBackend()
        rc = ResponseCache(backend=be, enabled=False)
        rc.set("k", "v")
        # Even if we enable later, nothing was stored
        assert be.get("k") is None

    def test_enabled_stores_and_retrieves(self):
        rc = ResponseCache(backend=MemoryCacheBackend(), enabled=True)
        rc.set("k", {"result": 42})
        assert rc.get("k") == {"result": 42}

    def test_stats_tracking(self):
        rc = ResponseCache(backend=MemoryCacheBackend(), enabled=True)
        rc.get("miss1")  # miss
        rc.set("k", "v")  # set
        rc.get("k")  # hit
        rc.get("miss2")  # miss
        stats = rc.stats()
        assert stats == {"hits": 1, "misses": 2, "sets": 1}

    def test_invalidate(self):
        rc = ResponseCache(backend=MemoryCacheBackend(), enabled=True)
        rc.set("k", "v")
        rc.invalidate("k")
        assert rc.get("k") is None

    def test_clear_resets_stats(self):
        rc = ResponseCache(backend=MemoryCacheBackend(), enabled=True)
        rc.set("k", "v")
        rc.get("k")
        rc.clear()
        assert rc.stats() == {"hits": 0, "misses": 0, "sets": 0}
        assert rc.get("k") is None


# ---------------------------------------------------------------------------
# TestCachedAskForJson
# ---------------------------------------------------------------------------


def _make_mock_driver(response_text='{"name": "Alice"}'):
    """Create a mock driver that returns a fixed response."""
    driver = Mock()
    driver.model = "test-model"
    _response = {
        "text": response_text,
        "meta": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.001,
        },
    }
    driver.generate.return_value = _response
    driver.generate_with_hooks.return_value = _response
    driver.generate_messages_with_hooks.return_value = _response
    return driver


class TestCachedAskForJson:
    """Tests for cache integration in ask_for_json."""

    def test_cache_miss_calls_driver(self):
        from prompture.extraction.core import ask_for_json

        configure_cache(backend="memory", enabled=True, ttl=60)
        driver = _make_mock_driver()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        result = ask_for_json(driver, "Tell me about Alice", schema)

        driver.generate_with_hooks.assert_called_once()
        assert result["json_object"]["name"] == "Alice"

    def test_cache_hit_skips_driver(self):
        from prompture.extraction.core import ask_for_json

        configure_cache(backend="memory", enabled=True, ttl=60)
        driver = _make_mock_driver()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        # First call — miss
        ask_for_json(driver, "Tell me about Alice", schema)
        # Second call — hit
        result = ask_for_json(driver, "Tell me about Alice", schema)

        assert driver.generate_with_hooks.call_count == 1  # only the first call
        assert result["usage"].get("cache_hit") is True

    def test_cache_hit_strips_raw_response(self):
        from prompture.extraction.core import ask_for_json

        configure_cache(backend="memory", enabled=True, ttl=60)
        driver = _make_mock_driver()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        ask_for_json(driver, "Tell me about Alice", schema)
        result = ask_for_json(driver, "Tell me about Alice", schema)

        assert result["usage"]["raw_response"] == {}

    def test_cache_false_bypasses(self):
        from prompture.extraction.core import ask_for_json

        configure_cache(backend="memory", enabled=True, ttl=60)
        driver = _make_mock_driver()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        ask_for_json(driver, "prompt", schema)
        ask_for_json(driver, "prompt", schema, cache=False)

        assert driver.generate_with_hooks.call_count == 2  # both calls hit the driver

    def test_cache_true_overrides_disabled(self):
        from prompture.extraction.core import ask_for_json

        configure_cache(backend="memory", enabled=False, ttl=60)
        driver = _make_mock_driver()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        ask_for_json(driver, "prompt", schema, cache=True)
        result = ask_for_json(driver, "prompt", schema, cache=True)

        assert driver.generate_with_hooks.call_count == 1
        assert result["usage"].get("cache_hit") is True


# ---------------------------------------------------------------------------
# TestCachedExtractWithModel
# ---------------------------------------------------------------------------


class _PersonModel(BaseModel):
    name: str
    age: int = 0


class TestCachedExtractWithModel:
    """Tests for cache integration in extract_with_model."""

    def test_pydantic_model_reconstructed_on_hit(self):
        from prompture.extraction.core import extract_with_model

        configure_cache(backend="memory", enabled=True, ttl=60)

        mock_result = {
            "json_string": '{"name": "Alice", "age": 30}',
            "json_object": {"name": "Alice", "age": 30},
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.001,
                "raw_response": {},
                "model_name": "test",
            },
            "output_format": "json",
        }

        with patch("prompture.extraction.core.extract_and_jsonify", return_value=mock_result):
            # First call — miss
            extract_with_model(_PersonModel, "Alice is 30", "openai/gpt-4")
            # Second call — hit
            r2 = extract_with_model(_PersonModel, "Alice is 30", "openai/gpt-4")

        assert r2["model"].name == "Alice"
        assert r2["model"].age == 30
        assert r2["usage"].get("cache_hit") is True

    def test_different_classes_different_keys(self):
        class _OtherModel(BaseModel):
            name: str
            age: int = 0

        k1 = make_cache_key("p", "m", pydantic_qualname=_PersonModel.__qualname__)
        k2 = make_cache_key("p", "m", pydantic_qualname=_OtherModel.__qualname__)
        assert k1 != k2


# ---------------------------------------------------------------------------
# TestConfigureCache
# ---------------------------------------------------------------------------


class TestConfigureCache:
    """Tests for configure_cache and get_cache."""

    def test_default_is_disabled(self):
        c = get_cache()
        assert c.enabled is False

    def test_configure_memory(self):
        c = configure_cache(backend="memory", enabled=True, ttl=120, maxsize=32)
        assert c.enabled is True
        assert c.default_ttl == 120
        assert isinstance(c.backend, MemoryCacheBackend)

    def test_configure_sqlite(self, tmp_path):
        db = str(tmp_path / "cache.db")
        c = configure_cache(backend="sqlite", enabled=True, db_path=db)
        assert isinstance(c.backend, SQLiteCacheBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown cache backend"):
            configure_cache(backend="memcached")

    def test_reconfigure_replaces_singleton(self):
        c1 = configure_cache(backend="memory", enabled=True)
        c2 = configure_cache(backend="memory", enabled=False)
        assert c1 is not c2
        assert get_cache() is c2

    def test_get_cache_returns_same_instance(self):
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2


# ---------------------------------------------------------------------------
# TestCacheSettings
# ---------------------------------------------------------------------------


class TestCacheSettings:
    """Tests for cache-related settings fields."""

    def test_default_values(self):
        from prompture.infra.settings import Settings

        s = Settings()
        assert s.cache_enabled is False
        assert s.cache_backend == "memory"
        assert s.cache_ttl_seconds == 3600
        assert s.cache_memory_maxsize == 256
        assert s.cache_sqlite_path is None
        assert s.cache_redis_url is None

    def test_env_var_override(self, monkeypatch):
        from prompture.infra.settings import Settings

        monkeypatch.setenv("CACHE_ENABLED", "true")
        monkeypatch.setenv("CACHE_BACKEND", "sqlite")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "7200")
        monkeypatch.setenv("CACHE_MEMORY_MAXSIZE", "512")
        monkeypatch.setenv("CACHE_SQLITE_PATH", "/tmp/test.db")
        monkeypatch.setenv("CACHE_REDIS_URL", "redis://custom:6380/1")

        s = Settings()
        assert s.cache_enabled is True
        assert s.cache_backend == "sqlite"
        assert s.cache_ttl_seconds == 7200
        assert s.cache_memory_maxsize == 512
        assert s.cache_sqlite_path == "/tmp/test.db"
        assert s.cache_redis_url == "redis://custom:6380/1"
