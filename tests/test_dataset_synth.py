"""Tests for ``prompture.dataset.synth.generate_qa_dataset`` and friends.

Mocks ``extract_with_model`` so the suite runs without a live LLM.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from prompture.dataset import (
    InstructionPair,
    QAPair,
    agenerate_qa_dataset,
    generate_qa_dataset,
    to_alpaca,
    to_jsonl,
    to_sharegpt,
    write_dataset,
)
from prompture.dataset.schemas import QAPairBatch
from prompture.rag.documents import Document

# ---------------------------------------------------------------------------
# Formatter tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pairs():
    return [
        QAPair(question="What is 2+2?", answer="4."),
        QAPair(question="What is 3+3?", answer="6."),
    ]


def test_to_jsonl_shape(sample_pairs):
    out = to_jsonl(sample_pairs)
    assert out == [
        {"question": "What is 2+2?", "answer": "4."},
        {"question": "What is 3+3?", "answer": "6."},
    ]


def test_to_sharegpt_shape(sample_pairs):
    out = to_sharegpt(sample_pairs)
    assert out[0]["conversations"][0] == {"from": "human", "value": "What is 2+2?"}
    assert out[0]["conversations"][1] == {"from": "gpt", "value": "4."}
    assert len(out) == 2


def test_to_alpaca_shape_from_qapairs(sample_pairs):
    out = to_alpaca(sample_pairs)
    assert out == [
        {"instruction": "What is 2+2?", "input": "", "output": "4."},
        {"instruction": "What is 3+3?", "input": "", "output": "6."},
    ]


def test_to_alpaca_from_instruction_pairs():
    pairs = [
        InstructionPair(
            instruction="Translate.",
            input="hello",
            output="bonjour",
        )
    ]
    out = to_alpaca(pairs)
    assert out == [{"instruction": "Translate.", "input": "hello", "output": "bonjour"}]


def test_write_dataset_jsonl():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "out.jsonl"
        records = [{"a": 1}, {"b": "héllo"}]
        result_path = write_dataset(records, path)
        assert result_path == path
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert "héllo" in lines[1]  # unicode preserved


def test_write_dataset_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nested" / "dir" / "out.jsonl"
        write_dataset([{"x": 1}], path)
        assert path.exists()


# ---------------------------------------------------------------------------
# generate_qa_dataset — mocked extract
# ---------------------------------------------------------------------------


@pytest.fixture
def two_docs():
    return [
        Document(content="Paris is the capital of France.", metadata={"source": "doc1"}),
        Document(
            content="The Eiffel Tower is 330m tall. It was completed in 1889.",
            metadata={"source": "doc2"},
        ),
    ]


def _mock_extract(model_cls, prompt, **kwargs):
    """Return one synthesised batch keyed off the prompt content."""
    if "Paris" in prompt:
        batch = QAPairBatch(
            pairs=[
                QAPair(question="What is the capital of France?", answer="Paris."),
                QAPair(question="Is Paris a city?", answer="Yes."),
            ]
        )
    else:
        batch = QAPairBatch(
            pairs=[
                QAPair(question="How tall is the Eiffel Tower?", answer="330m."),
                QAPair(question="When was the Eiffel Tower completed?", answer="1889."),
            ]
        )
    return {"model": batch, "usage": {}}


def test_generate_returns_qapairs_from_documents(two_docs):
    with patch("prompture.dataset.synth.extract_with_model", side_effect=_mock_extract):
        pairs = generate_qa_dataset(
            two_docs,  # list of Document
            model="mock/x",
            n_per_chunk=2,
        )
    assert len(pairs) == 4
    questions = {p.question for p in pairs}
    assert "What is the capital of France?" in questions
    assert "How tall is the Eiffel Tower?" in questions


def test_dedupe_removes_duplicate_questions(two_docs):
    def _dup_extract(model_cls, prompt, **kwargs):
        return {
            "model": QAPairBatch(
                pairs=[
                    QAPair(question="Same question?", answer="A"),
                    QAPair(question="same question?  ", answer="B"),
                    QAPair(question="Different?", answer="C"),
                ]
            ),
            "usage": {},
        }

    with patch("prompture.dataset.synth.extract_with_model", side_effect=_dup_extract):
        pairs = generate_qa_dataset(two_docs, model="mock/x", dedupe=True)

    questions = [p.question for p in pairs]
    # Each of the 2 docs runs the extractor; with dedupe we keep the
    # first occurrence of 'Same question?' and 'Different?' only.
    assert "Same question?" in questions
    assert questions.count("Different?") == 1


def test_dedupe_false_keeps_duplicates(two_docs):
    def _dup_extract(model_cls, prompt, **kwargs):
        return {
            "model": QAPairBatch(pairs=[QAPair(question="Q?", answer="A"), QAPair(question="Q?", answer="B")]),
            "usage": {},
        }

    with patch("prompture.dataset.synth.extract_with_model", side_effect=_dup_extract):
        pairs = generate_qa_dataset(two_docs, model="mock/x", dedupe=False)
    assert len(pairs) >= 4  # 2 chunks × 2 dup-pairs


def test_max_chunks_caps_extraction(two_docs):
    call_count = {"n": 0}

    def _count_extract(model_cls, prompt, **kwargs):
        call_count["n"] += 1
        return {
            "model": QAPairBatch(pairs=[QAPair(question=f"Q{call_count['n']}", answer="A")]),
            "usage": {},
        }

    with patch("prompture.dataset.synth.extract_with_model", side_effect=_count_extract):
        # Use a chunker that produces 1 chunk per document → 2 chunks total,
        # but cap at 1.
        from prompture.rag.chunkers import RecursiveCharacterChunker

        generate_qa_dataset(
            two_docs,
            model="mock/x",
            chunker=RecursiveCharacterChunker(chunk_size=10_000),
            max_chunks=1,
        )
    assert call_count["n"] == 1


def test_on_chunk_callback_receives_progress(two_docs):
    received: list[tuple[int, int]] = []

    def _cb(idx, total, chunk):
        received.append((idx, total))

    with patch("prompture.dataset.synth.extract_with_model", side_effect=_mock_extract):
        from prompture.rag.chunkers import RecursiveCharacterChunker

        generate_qa_dataset(
            two_docs,
            model="mock/x",
            chunker=RecursiveCharacterChunker(chunk_size=10_000),
            on_chunk=_cb,
        )
    # Two chunks → two callbacks
    assert len(received) == 2
    assert received[0] == (0, 2)
    assert received[1] == (1, 2)


def test_writes_output_jsonl_format(two_docs):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.jsonl"
        with patch("prompture.dataset.synth.extract_with_model", side_effect=_mock_extract):
            generate_qa_dataset(
                two_docs,
                model="mock/x",
                output_path=out,
                output_format="jsonl",
            )
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines
        record = json.loads(lines[0])
        assert "question" in record and "answer" in record


def test_writes_output_sharegpt_format(two_docs):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.jsonl"
        with patch("prompture.dataset.synth.extract_with_model", side_effect=_mock_extract):
            generate_qa_dataset(
                two_docs,
                model="mock/x",
                output_path=out,
                output_format="sharegpt",
            )
        record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert "conversations" in record
        assert record["conversations"][0]["from"] == "human"


def test_writes_output_alpaca_format(two_docs):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.jsonl"
        with patch("prompture.dataset.synth.extract_with_model", side_effect=_mock_extract):
            generate_qa_dataset(
                two_docs,
                model="mock/x",
                output_path=out,
                output_format="alpaca",
            )
        record = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert set(record.keys()) == {"instruction", "input", "output"}


def test_unknown_output_format_raises(two_docs):
    with patch("prompture.dataset.synth.extract_with_model", side_effect=_mock_extract):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(ValueError, match="output_format"):
                generate_qa_dataset(
                    two_docs,
                    model="mock/x",
                    output_path=Path(td) / "o.jsonl",
                    output_format="parquet",  # not supported
                )


def test_path_source_loads_text_file():
    """A real .txt path goes through the loader registry."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "facts.txt"
        p.write_text("Paris is the capital of France.\n", encoding="utf-8")
        with patch("prompture.dataset.synth.extract_with_model", side_effect=_mock_extract):
            pairs = generate_qa_dataset(p, model="mock/x")
        assert pairs


def test_empty_source_returns_empty_list():
    with patch("prompture.dataset.synth.extract_with_model", side_effect=_mock_extract):
        pairs = generate_qa_dataset([], model="mock/x")
    assert pairs == []


def test_chunk_failure_is_logged_and_skipped(two_docs):
    """A failing chunk shouldn't kill the whole run."""
    call = {"n": 0}

    def _flaky(model_cls, prompt, **kwargs):
        call["n"] += 1
        if call["n"] == 1:
            raise RuntimeError("rate limit")
        return {
            "model": QAPairBatch(pairs=[QAPair(question="Q?", answer="A")]),
            "usage": {},
        }

    with patch("prompture.dataset.synth.extract_with_model", side_effect=_flaky):
        from prompture.rag.chunkers import RecursiveCharacterChunker

        pairs = generate_qa_dataset(
            two_docs,
            model="mock/x",
            chunker=RecursiveCharacterChunker(chunk_size=10_000),
        )
    # Second chunk succeeded → at least one pair.
    assert pairs


# ---------------------------------------------------------------------------
# Async variant
# ---------------------------------------------------------------------------


def test_async_generate_returns_pairs(two_docs):
    async def _fake(model_cls, prompt, **kwargs):
        return {
            "model": QAPairBatch(pairs=[QAPair(question="AsyncQ?", answer="AsyncA")]),
            "usage": {},
        }

    with patch("prompture.dataset.synth._async_extract_with_model", side_effect=_fake):
        pairs = asyncio.run(agenerate_qa_dataset(two_docs, model="mock/x", concurrency=2))
    questions = {p.question for p in pairs}
    assert "AsyncQ?" in questions
