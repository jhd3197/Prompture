"""Contextual chunking — Anthropic Sep 2024 technique.

Wraps any base chunker and uses an LLM to generate a short situating
context (50–100 tokens) for each chunk *within its parent document*.
The context is prepended to the chunk text before downstream
embedding, which Anthropic reports reduces failed retrievals by ~49%
on their own evaluation set.

Cost shape
~~~~~~~~~~

One extra LLM call per chunk. Because every call for the same document
shares the document content as its prefix, this technique is **designed
to be paired with prompt caching**:

* The full document is sent in the system role so the Claude driver's
  cache_control logic (Phase 1) wraps it as a cacheable prefix.
* The per-chunk question goes in the user role and varies every call.
* The first chunk pays the full cache-creation cost (~1.25× input);
  every subsequent chunk for the same document pays cache-read price
  (~0.1× input).

For LLMs that don't expose ``generate_messages`` the wrapper falls back
to a single combined prompt via ``generate()``, which still works
correctly but loses the caching benefit on Anthropic.

Design notes
~~~~~~~~~~~~

* The technique inherently needs the *whole document* alongside the
  chunk — calling :meth:`split_text` on a raw string can't produce
  contextualised output, so that path silently degrades to the base
  chunker's behaviour.
* ``fail_open=True`` (the default) makes the chunker resilient: any
  per-chunk LLM failure prepends an empty context (no-op) instead of
  raising. Set ``False`` if you want hard failure on LLM trouble.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ..documents import Document
from .base import TextChunker

logger = logging.getLogger(__name__)


#: Default user-prompt template. ``{chunk}`` is substituted with the
#: chunk text. The full document is sent separately via the system role
#: (or prepended in the fallback path) so this prompt stays short and
#: cache-friendly.
DEFAULT_CONTEXTUAL_PROMPT = (
    "Here is a chunk taken from the document above:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, succinct context (1–2 sentences) that situates this "
    "chunk within the overall document. The goal is to improve retrieval "
    "of this chunk when someone searches for related concepts. Answer "
    "with the context only — no preamble, no quotes, no labels."
)

#: Fallback combined-prompt template, used when the driver doesn't
#: expose ``generate_messages``. Less cache-friendly than the
#: messages path but functionally equivalent.
_FALLBACK_COMBINED_PROMPT = "<document>\n{document}\n</document>\n\n{user_prompt}"


class _LLM(Protocol):
    """Minimal LLM contract used by :class:`ContextualChunker`."""

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        ...


def _response_text(response: Any) -> str:
    """Extract the text payload from a driver response dict or stringable object."""
    if isinstance(response, dict):
        return str(response.get("text") or "").strip()
    return str(response or "").strip()


class ContextualChunker(TextChunker):
    """Wrap a base chunker; per-chunk LLM-generated context is prepended.

    Args:
        base: The underlying chunker (e.g. :class:`RecursiveCharacterChunker`).
            Its ``split_text`` is what we contextualise.
        driver: Any LLM driver. Best results come from one that supports
            prompt caching (Claude / OpenAI). When the driver exposes
            ``generate_messages`` the chunker uses the system-role path
            for caching; otherwise it falls back to a single combined
            prompt sent through ``generate``.
        prompt: User-prompt template containing a ``{chunk}`` placeholder.
            Defaults to :data:`DEFAULT_CONTEXTUAL_PROMPT`. Do NOT put the
            document into this template — the chunker handles that.
        llm_options: Forwarded to the driver. Defaults set a tight
            ``max_tokens`` budget (the context should be short) and
            leave ``cache_prompt=True`` so Claude drivers cache the
            document prefix automatically.
        max_context_chars: Cap on the per-chunk context length (chars).
            Defaults to 400 (≈100 tokens). Long contexts dilute the
            embedding signal of the actual chunk text.
        max_document_chars: Cap on the document text fed to the LLM.
            Defaults to 200000 — covers all common model context windows
            while keeping cache_creation costs bounded.
        fail_open: When True (default), per-chunk LLM failures fall
            through to an empty context (= behaviour of the base
            chunker). When False, the first failure is re-raised.
        skip_if_only_one_chunk: When True (default), single-chunk
            documents are returned as-is — no LLM call. Multi-chunk
            documents always get contextualised.
    """

    # ContextualChunker doesn't introduce its own chunk_size; it inherits
    # from the base chunker for any caller that needs to inspect those.
    def __init__(
        self,
        base: TextChunker,
        driver: _LLM,
        *,
        prompt: str = DEFAULT_CONTEXTUAL_PROMPT,
        llm_options: dict[str, Any] | None = None,
        max_context_chars: int = 400,
        max_document_chars: int = 200_000,
        fail_open: bool = True,
        skip_if_only_one_chunk: bool = True,
    ) -> None:
        if "{chunk}" not in prompt:
            raise ValueError(
                f"ContextualChunker prompt must include a '{{chunk}}' placeholder; got: {prompt[:80]!r}..."
            )
        # NOTE: we deliberately do *not* call super().__init__ with chunk_size
        # because that base check rejects chunk_overlap >= chunk_size, but the
        # base chunker already enforces sizing. We mirror the base's values so
        # any caller inspecting them sees something sensible.
        self.chunk_size = getattr(base, "chunk_size", 1000)
        self.chunk_overlap = getattr(base, "chunk_overlap", 200)
        self.base = base
        self.driver = driver
        self.prompt = prompt
        self.llm_options = (
            llm_options
            if llm_options is not None
            else {
                "temperature": 0.0,
                "max_tokens": 200,
                "cache_prompt": True,
            }
        )
        self.max_context_chars = max_context_chars
        self.max_document_chars = max_document_chars
        self.fail_open = fail_open
        self.skip_if_only_one_chunk = skip_if_only_one_chunk

    # ------------------------------------------------------------------
    # TextChunker contract
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> list[str]:
        """Delegate to the base chunker.

        ``split_text`` operates on a raw string and has no access to the
        surrounding document context, so contextualisation is not
        applied on this path — call :meth:`split_documents` to get
        contextualised output.
        """
        return self.base.split_text(text)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split each document and prepend an LLM-generated context per chunk."""
        result: list[Document] = []
        for doc in documents:
            doc_text = doc.content
            chunks = self.base.split_text(doc_text)
            count = len(chunks)
            parent_source = doc.metadata.get("source")

            # Avoid the LLM call entirely when there's nothing to contextualise.
            do_contextualise = count > 1 or not self.skip_if_only_one_chunk

            # Truncate document for the LLM call (chunks themselves are unaffected).
            doc_for_llm = doc_text
            if len(doc_for_llm) > self.max_document_chars:
                doc_for_llm = doc_for_llm[: self.max_document_chars]
                logger.debug(
                    "ContextualChunker: document truncated to %d chars for LLM context generation.",
                    self.max_document_chars,
                )

            for i, chunk in enumerate(chunks):
                context = ""
                if do_contextualise:
                    context = self._generate_context(doc_for_llm, chunk)
                final_content = f"{context}\n\n{chunk}" if context else chunk
                new_meta = dict(doc.metadata)
                new_meta["chunk_index"] = i
                new_meta["chunk_count"] = count
                if parent_source is not None and "parent_source" not in new_meta:
                    new_meta["parent_source"] = parent_source
                if context:
                    new_meta["contextual_context"] = context
                result.append(Document(content=final_content, metadata=new_meta))
        return result

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------

    def _generate_context(self, document: str, chunk: str) -> str:
        """Produce a 1–2 sentence situating context for ``chunk`` in ``document``.

        Returns an empty string when the LLM is unavailable / fails /
        produces no usable text and ``fail_open`` is True. Re-raises
        otherwise.
        """
        user_prompt = self.prompt.format(chunk=chunk)

        # Preferred path: system role gets the document so Claude can cache it.
        gm = getattr(self.driver, "generate_messages", None)
        if callable(gm):
            messages = [
                {"role": "system", "content": document},
                {"role": "user", "content": user_prompt},
            ]
            try:
                response = gm(messages, self.llm_options)
            except Exception:
                logger.warning(
                    "ContextualChunker: generate_messages failed; falling back to generate().",
                    exc_info=True,
                )
            else:
                return self._postprocess(_response_text(response))

        # Fallback: single combined prompt.
        combined = _FALLBACK_COMBINED_PROMPT.format(document=document, user_prompt=user_prompt)
        try:
            response = self.driver.generate(combined, self.llm_options)
        except Exception:
            if self.fail_open:
                logger.warning(
                    "ContextualChunker: LLM call failed; emitting un-contextualised chunk.",
                    exc_info=True,
                )
                return ""
            raise
        return self._postprocess(_response_text(response))

    def _postprocess(self, text: str) -> str:
        """Trim, strip quotes / common label prefixes, cap length."""
        if not text:
            return ""
        cleaned = text.strip().strip(" \"'")
        # Remove common label prefixes the model might add.
        for prefix in ("Context:", "Situating context:", "Here is the context:"):
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].lstrip(" :\"'")
        if len(cleaned) > self.max_context_chars:
            cleaned = cleaned[: self.max_context_chars].rstrip()
        return cleaned
