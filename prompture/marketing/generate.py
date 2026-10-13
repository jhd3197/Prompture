"""Generate a marketing model from a task, with the brand in front of the model.

A thin layer over :func:`~prompture.extraction.core.extract_with_model`: the
instruction says *create*, not *extract*, the brand block goes in the system
prompt, and validation failures are retried with the exact rule that broke::

    from prompture.marketing import BrandContext, SocialPostSet, generate

    brand = BrandContext.load("brand.md")
    posts = generate(
        SocialPostSet,
        "Write 4 LinkedIn posts: one insight, one trend, one question, one contrarian.",
        "openai/gpt-4o",
        brand=brand,
    )
    for post in posts.posts:
        print(post.hook, post.image_hook)
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

__all__ = ["GENERATION_INSTRUCTION", "agenerate", "generate"]

T = TypeVar("T", bound=BaseModel)

#: Replaces the default "Extract information from the following text" instruction.
GENERATION_INSTRUCTION = (
    "Complete the task below for the business described in the system prompt. "
    "Follow the field descriptions and limits in the schema exactly."
)


def generate(
    output_model: type[T],
    task: str,
    model_name: str,
    *,
    brand: Any = None,
    instruction: str = GENERATION_INSTRUCTION,
    max_retries: int = 2,
    **kwargs: Any,
) -> T:
    """Create an ``output_model`` instance for ``task``.

    Args:
        output_model: A Pydantic model, typically one from :mod:`prompture.marketing`.
        task: What to create, in plain language.
        model_name: ``"provider/model"``.
        brand: A :class:`~prompture.marketing.BrandContext`, dict, or brand file path.
        instruction: Overrides :data:`GENERATION_INSTRUCTION`.
        max_retries: Validation retries (each retry shows the model what failed).
        **kwargs: Forwarded to ``extract_with_model`` (``options``, ``system_prompt``, ...).
    """
    from ..extraction.core import extract_with_model

    result = extract_with_model(
        output_model,
        task,
        model_name,
        instruction_template=instruction,
        brand=brand,
        max_retries=max_retries,
        **kwargs,
    )
    return result["model"]


async def agenerate(
    output_model: type[T],
    task: str,
    model_name: str,
    *,
    brand: Any = None,
    instruction: str = GENERATION_INSTRUCTION,
    max_retries: int = 2,
    **kwargs: Any,
) -> T:
    """Async version of :func:`generate`."""
    from ..extraction.async_core import extract_with_model

    result = await extract_with_model(
        output_model,
        task,
        model_name,
        instruction_template=instruction,
        brand=brand,
        max_retries=max_retries,
        **kwargs,
    )
    return result["model"]
