"""Pydantic schemas used when generating synthetic training data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QAPair(BaseModel):
    """A single question/answer pair grounded in a source chunk."""

    question: str = Field(..., description="A self-contained question a reader might ask.")
    answer: str = Field(..., description="A complete, accurate answer drawn from the source.")


class QAPairBatch(BaseModel):
    """A batch of Q&A pairs from a single source chunk.

    The LLM is asked to emit several pairs per call so we amortise the
    prompt overhead across the chunk.
    """

    pairs: list[QAPair] = Field(
        ...,
        description="Multiple distinct Q&A pairs covering different facts in the source.",
    )


class InstructionPair(BaseModel):
    """A single instruction/output pair, optionally with extra input context."""

    instruction: str = Field(..., description="A clear instruction the model must follow.")
    input: str = Field("", description="Optional supporting input for the instruction. Empty if none.")
    output: str = Field(..., description="The ideal model response.")


class InstructionPairBatch(BaseModel):
    """A batch of instruction/output pairs from one source chunk."""

    pairs: list[InstructionPair] = Field(
        ...,
        description="Multiple distinct instruction/output pairs derived from the source.",
    )


class ChatTurn(BaseModel):
    """A single chat-style message turn."""

    role: Literal["system", "user", "assistant"]
    content: str
