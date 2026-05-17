"""Output-format converters for synthesised datasets.

Three formats are supported, mapping cleanly onto the major fine-tuning
toolchains:

* ``jsonl`` — a flat ``{"question": ..., "answer": ...}`` record per line.
  Simplest format; works with any custom training loop.
* ``sharegpt`` — ``{"conversations": [{"from": "human", "value": q},
  {"from": "gpt", "value": a}]}``. The default for Unsloth fine-tunes
  and Axolotl's ``sharegpt`` plugin.
* ``alpaca`` — ``{"instruction": ..., "input": ..., "output": ...}``.
  The classic Stanford Alpaca shape, accepted by Axolotl, TRL, and most
  Hugging Face notebooks.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .schemas import InstructionPair, QAPair


def to_jsonl(pairs: Sequence[QAPair]) -> list[dict[str, Any]]:
    """Return Q&A pairs as plain ``{"question", "answer"}`` records."""
    return [{"question": p.question, "answer": p.answer} for p in pairs]


def to_sharegpt(pairs: Sequence[QAPair]) -> list[dict[str, Any]]:
    """Return Q&A pairs as ShareGPT-style conversations.

    Each example contains a single human → gpt exchange::

        {"conversations": [
            {"from": "human", "value": "<question>"},
            {"from": "gpt",   "value": "<answer>"},
        ]}
    """
    return [
        {
            "conversations": [
                {"from": "human", "value": p.question},
                {"from": "gpt", "value": p.answer},
            ]
        }
        for p in pairs
    ]


def to_alpaca(
    pairs: Sequence[QAPair | InstructionPair],
) -> list[dict[str, Any]]:
    """Return pairs as Stanford-Alpaca ``{instruction, input, output}`` records.

    ``QAPair`` items are mapped as ``instruction=question`` / ``output=answer``
    with an empty ``input`` field.  ``InstructionPair`` items pass through
    directly.
    """
    out: list[dict[str, Any]] = []
    for p in pairs:
        if isinstance(p, InstructionPair):
            out.append({"instruction": p.instruction, "input": p.input, "output": p.output})
        else:
            out.append({"instruction": p.question, "input": "", "output": p.answer})
    return out


def write_dataset(
    pairs: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write *pairs* as JSON-Lines to *output_path*.

    Creates parent directories as needed.  Each record is encoded with
    ``ensure_ascii=False`` so non-ASCII text remains readable.

    Returns the resolved :class:`Path` for convenience.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in pairs:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
    return path
