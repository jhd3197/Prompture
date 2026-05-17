#!/usr/bin/env python3
"""Generate a fine-tuning dataset from a document corpus.

Demonstrates ``prompture.dataset.generate_qa_dataset`` — load a file (or
glob), chunk it, ask an LLM to emit Q&A pairs grounded in each chunk,
and write the result as JSONL ready to feed into Unsloth, Axolotl,
TRL, or any other fine-tuning toolchain.

Run::

    pip install prompture
    export OPENAI_API_KEY=sk-...
    python examples/dataset_generation_example.py

Requires:
    - A configured LLM provider (defaults to openai/gpt-4o-mini).
    - Loader extras for non-text files: ``prompture[rag-pdf]``, etc.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from prompture import generate_qa_dataset

MODEL = os.environ.get("PROMPTURE_EXAMPLE_MODEL", "openai/gpt-4o-mini")


# Drop a small text file to demo on. In practice, point at a PDF, DOCX,
# CSV, HTML, EPUB, XLSX, JSON, JSONL, or Markdown file.
SAMPLE_TEXT = """\
Prompture is a Python library for structured LLM extraction.

It supports 36+ providers, including OpenAI, Claude, Google, Groq, Grok,
Azure, Ollama, LM Studio, OpenRouter, HuggingFace, and many more.

The library exposes a unified `provider/model` routing system so calls
look the same regardless of which backend is in use.

Built-in features include schema-enforced JSON extraction, Pydantic
model integration, conversation state management, tool/function calling
with automatic simulation for non-native models, deep agents with
planning and virtual filesystem, RAG (loaders, chunkers, vector stores,
retrievers), and an OpenAI-compatible server.
"""


with tempfile.TemporaryDirectory() as workspace:
    src = Path(workspace) / "prompture_overview.txt"
    src.write_text(SAMPLE_TEXT, encoding="utf-8")

    out = Path(workspace) / "training.jsonl"

    print("Generating Q&A pairs...")

    def show_progress(i: int, total: int, chunk) -> None:
        print(f"  chunk {i + 1}/{total} ({len(chunk.content)} chars)")

    pairs = generate_qa_dataset(
        src,
        model=MODEL,
        n_per_chunk=4,
        output_path=out,
        output_format="sharegpt",  # try 'jsonl' or 'alpaca'
        on_chunk=show_progress,
    )

    print(f"\nGenerated {len(pairs)} pairs → {out}")
    print("\nFirst 3 pairs (raw QAPair objects):")
    for p in pairs[:3]:
        print(f"  Q: {p.question}")
        print(f"  A: {p.answer}\n")

    print("File preview:")
    for line in out.read_text(encoding="utf-8").splitlines()[:3]:
        print(f"  {line}")
