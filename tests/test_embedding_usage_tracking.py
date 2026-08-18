"""Embedding spend: sub-cent costs survive, calls reach the tracker, and the
vector stores speak the current driver contract."""

from __future__ import annotations

import asyncio
from typing import Any

from prompture.drivers.async_embedding_base import AsyncEmbeddingDriver
from prompture.drivers.embedding_base import EmbeddingDriver
from prompture.drivers.openai_embedding_driver import OpenAIEmbeddingDriver
from prompture.infra.tracker import UsageEvent, configure_tracker
from prompture.rag.vectorstores.base import embed_texts


def _capture_events(tmp_path) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    configure_tracker(db_path=str(tmp_path / "u.db"), persist=False, sinks=[events.append])
    return events


class StubEmbeddingDriver(EmbeddingDriver):
    model = "stub-embed-1"

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        return {
            "embeddings": [[0.1, 0.2] for _ in texts],
            "meta": {
                "model_name": "stub/stub-embed-1",
                "dimensions": 2,
                "total_tokens": 7 * len(texts),
                "cost": 1.4e-7 * len(texts),
            },
        }


class StubAsyncEmbeddingDriver(AsyncEmbeddingDriver):
    model = "stub-embed-1"

    async def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        return StubEmbeddingDriver().embed(texts, options)


class LegacyEmbeddingDriver:
    """The old one-arg contract some toy drivers still use."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.3, 0.4] for _ in texts]


def test_embed_with_hooks_records_usage(tmp_path):
    events = _capture_events(tmp_path)

    StubEmbeddingDriver().embed_with_hooks(["a", "b"], {})

    assert len(events) == 1
    assert events[0].total_tokens == 14
    assert events[0].prompt_tokens == 14
    assert events[0].cost > 0.0
    assert events[0].metadata == {"modality": "embedding", "count": 2}


def test_async_embed_with_hooks_records_usage(tmp_path):
    events = _capture_events(tmp_path)

    asyncio.run(StubAsyncEmbeddingDriver().embed_with_hooks(["a"], {}))

    assert len(events) == 1
    assert events[0].total_tokens == 7


def test_embedding_cost_is_not_floored_to_zero():
    # A typical single ingest is a handful of tokens; round(cost, 6) used to
    # floor every such call to $0.00, so embedding spend never summed to
    # anything anywhere.
    d = OpenAIEmbeddingDriver(api_key="x")
    cost = d._calculate_embedding_cost("openai", "text-embedding-3-small", total_tokens=8)
    assert cost > 0.0


def test_embed_texts_handles_current_contract():
    vectors = embed_texts(StubEmbeddingDriver(), ["a", "b"])
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]


def test_embed_texts_handles_legacy_contract():
    vectors = embed_texts(LegacyEmbeddingDriver(), ["a"])
    assert vectors == [[0.3, 0.4]]
