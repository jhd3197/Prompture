"""The suite only protects drivers it knows about, so every tool-capable driver
must be in it, or be listed here with the reason it is not yet."""

from __future__ import annotations

import pytest
from conformance_harness import ANTHROPIC_WIRE, OPENAI_WIRE

from prompture.drivers._openai_chat import _ChatToolsBase
from prompture.drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP, _resolve_cls

#: Tool-capable drivers with no conformance scenarios yet. Each speaks its own
#: wire format, so each needs its own fake transport and scenario file.
NOT_YET_COVERED = {
    "GoogleDriver": "Gemini wire format (shared in _google_genai.py)",
    "AsyncGoogleDriver": "Gemini wire format (shared in _google_genai.py)",
    "OllamaDriver": "Ollama native chat format",
    "AsyncOllamaDriver": "Ollama native chat format",
    "CohereDriver": "Cohere v2 chat format",
    "AsyncCohereDriver": "Cohere v2 chat format",
}

#: OpenAI's own driver keeps its tool path for the Responses API, prompt-cache
#: keys and usage reporting; it calls the shared helpers rather than inheriting.
HAND_WRITTEN_OPENAI_WIRE = {"OpenAIDriver", "AsyncOpenAIDriver"}


def _tool_drivers() -> dict[str, type]:
    found: dict[str, type] = {}
    for desc in PROVIDER_DESCRIPTOR_MAP.values():
        if desc.alias_for:
            continue
        for spec in (desc.llm_sync, desc.llm_async):
            if spec is None:
                continue
            try:
                cls = _resolve_cls(spec.cls_path)
            except Exception:  # optional SDK missing
                continue
            if getattr(cls, "supports_tool_use", False):
                found[cls.__name__] = cls
    return found


def test_every_tool_driver_is_covered():
    covered = {target.cls for target in OPENAI_WIRE + ANTHROPIC_WIRE}
    missing = sorted(set(_tool_drivers()) - covered - set(NOT_YET_COVERED))
    assert not missing, (
        f"{missing} support tool use but are not in the conformance suite. "
        "Add them to OPENAI_WIRE or ANTHROPIC_WIRE in conformance_harness.py."
    )


def test_not_yet_covered_list_is_current():
    stale = sorted(set(NOT_YET_COVERED) & {target.cls for target in OPENAI_WIRE + ANTHROPIC_WIRE})
    assert not stale, f"{stale} are covered now; remove them from NOT_YET_COVERED"


@pytest.mark.parametrize(
    "target", [t for t in OPENAI_WIRE if t.cls not in HAND_WRITTEN_OPENAI_WIRE], ids=lambda t: t.id
)
def test_openai_wire_drivers_share_the_codec(target):
    """A driver that writes its own tool-call body is how the copies came back."""
    cls = target.load()
    assert issubclass(cls, _ChatToolsBase), f"{cls.__name__} must build tool calls through _openai_chat.py"
