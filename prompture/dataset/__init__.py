"""Synthetic dataset generation from documents.

Turn a corpus of PDFs, DOCX, HTML, or any RAG-loader-supported format
into a training-ready JSONL/ShareGPT/Alpaca dataset by composing
Prompture's existing loaders, chunkers, and structured extraction::

    from prompture.dataset import generate_qa_dataset

    pairs = generate_qa_dataset(
        "policy.pdf",
        model="openai/gpt-4o-mini",
        n_per_chunk=3,
        output_path="training.jsonl",
        output_format="sharegpt",
    )

The output is ready to feed into Unsloth, Axolotl, TRL, or any other
fine-tuning framework that consumes one of the three formats.
"""

from .formats import to_alpaca, to_jsonl, to_sharegpt, write_dataset
from .schemas import ChatTurn, InstructionPair, QAPair
from .synth import agenerate_qa_dataset, generate_qa_dataset

__all__ = [
    "ChatTurn",
    "InstructionPair",
    "QAPair",
    "agenerate_qa_dataset",
    "generate_qa_dataset",
    "to_alpaca",
    "to_jsonl",
    "to_sharegpt",
    "write_dataset",
]
