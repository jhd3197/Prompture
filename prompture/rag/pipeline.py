"""End-to-end RAG pipeline composing loader, chunker, vector store,
retriever, optional reranker, and an LLM driver.

Two flavours are provided:

* :class:`RAGPipeline` — synchronous orchestration.
* :class:`AsyncRAGPipeline` — async orchestration; native-async
  subcomponents (async retrievers / async LLMs) are used when present,
  with :func:`asyncio.to_thread` as a fallback for sync subs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..drivers.base import Driver
from .documents import Document
from .retrievers.base import AsyncRetriever, Retriever
from .vectorstores.base import VectorSearchResult

#: Default prompt template used by :class:`RAGPipeline`.  Callers can
#: override via the ``prompt_template`` constructor argument.
DEFAULT_RAG_PROMPT = """You are a helpful assistant. Use the following retrieved context to answer the user's question. If the context doesn't contain the answer, say so honestly.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGAnswer:
    """Wraps the result of a :meth:`RAGPipeline.query` call.

    Attributes:
        answer: The LLM-generated answer text.
        sources: The :class:`Document` objects fed to the LLM as context.
        retrieval_results: The raw :class:`VectorSearchResult` list from
            the retriever (post-rerank if a reranker was configured).
        usage: Token / cost metadata from the LLM call.
    """

    answer: str
    sources: list[Document] = field(default_factory=list)
    retrieval_results: list[VectorSearchResult] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def _format_context(results: list[VectorSearchResult]) -> str:
    """Join retrieved chunks into a single context string for the prompt."""
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        parts.append(f"[{i}] {r.document.content}")
    return "\n\n".join(parts)


def _apply_reranker_sync(
    reranker: Any,
    query: str,
    results: list[VectorSearchResult],
    top_n: int,
) -> list[VectorSearchResult]:
    """Re-order ``results`` using a rerank driver, return up to ``top_n``."""
    if not results:
        return []
    docs = [r.document.content for r in results]
    rerank_results = reranker.rerank(query=query, documents=docs, top_n=top_n, return_documents=False)
    return [results[rr.index] for rr in rerank_results]


async def _apply_reranker_async(
    reranker: Any,
    query: str,
    results: list[VectorSearchResult],
    top_n: int,
) -> list[VectorSearchResult]:
    if not results:
        return []
    docs = [r.document.content for r in results]
    rerank_fn = reranker.rerank
    if asyncio.iscoroutinefunction(rerank_fn):
        rerank_results = await rerank_fn(query=query, documents=docs, top_n=top_n, return_documents=False)
    else:
        rerank_results = await asyncio.to_thread(
            rerank_fn, query=query, documents=docs, top_n=top_n, return_documents=False
        )
    return [results[rr.index] for rr in rerank_results]


def _resolve_reranker(reranker: Any, *, prefer_async: bool = False) -> Any:
    """Coerce a ``provider/model`` string into an instantiated rerank driver.

    Already-instantiated drivers and ``None`` are returned unchanged so
    callers can mix-and-match. When ``prefer_async`` is true (the async
    pipeline), tries the async registry first and falls back to the sync
    registry — :func:`_apply_reranker_async` handles either uniformly.
    """
    if reranker is None or not isinstance(reranker, str):
        return reranker
    if "/" not in reranker:
        raise ValueError(
            f"Reranker string {reranker!r} must be in 'provider/model' form "
            "(e.g. 'cohere/rerank-v3.5'). Pass a pre-instantiated driver instead "
            "if you need custom construction."
        )
    if prefer_async:
        try:
            from ..drivers.rerank_registry import get_async_rerank_driver_for_model

            return get_async_rerank_driver_for_model(reranker)
        except Exception:
            # Async driver may not exist for this provider; fall through to sync.
            pass
    from ..drivers.rerank_registry import get_rerank_driver_for_model

    return get_rerank_driver_for_model(reranker)


# ── Sync pipeline ────────────────────────────────────────────────────────────


class RAGPipeline:
    """End-to-end RAG composition.

    Composes a retriever, an optional reranker, and an LLM driver into a
    single object exposing :meth:`query` for Q&A and :meth:`extract` for
    structured extraction.  Use :meth:`ingest` to load + chunk + embed
    documents into the retriever's underlying vector store.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: Driver,
        *,
        reranker: Any = None,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        top_n_after_rerank: int = 4,
        system_prompt: str | None = None,
    ) -> None:
        """Construct a RAG pipeline.

        Args:
            retriever: The retriever providing dense / hybrid / sparse search.
            llm: An LLM driver for the generation step.
            reranker: Optional reranker. Accepts either a pre-instantiated
                rerank driver or a ``"provider/model"`` string such as
                ``"cohere/rerank-v3.5"`` — the string is resolved via
                :func:`prompture.drivers.rerank_registry.get_rerank_driver_for_model`.
            prompt_template: Template containing ``{context}`` and ``{question}``.
            top_n_after_rerank: How many results to keep after reranking.
                Ignored when ``reranker`` is ``None``.
            system_prompt: Optional system prompt for the LLM call.
        """
        self.retriever = retriever
        self.llm = llm
        self.reranker = _resolve_reranker(reranker, prefer_async=False)
        self.prompt_template = prompt_template
        self.top_n_after_rerank = top_n_after_rerank
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: Any,
        *,
        loader: Any = None,
        chunker: Any = None,
        vector_store: Any = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Load → chunk → store documents from ``source``.

        Args:
            source: Path-like or other loader-compatible source.
            loader: Optional :class:`DocumentLoader`.  Defaults to
                ``get_loader_for_path(source)``.
            chunker: Optional :class:`TextChunker`.  Defaults to a
                :class:`RecursiveCharacterChunker` with ``chunk_size=1000``
                and ``chunk_overlap=200``.
            vector_store: Optional :class:`VectorStore`.  Defaults to
                ``self.retriever.vector_store``.
            ids: Optional explicit IDs to pass through to the vector store.

        Returns:
            The list of IDs added to the vector store.
        """
        if loader is None:
            from .loader_registry import get_loader_for_path

            loader = get_loader_for_path(str(source))
        if chunker is None:
            from .chunkers import RecursiveCharacterChunker

            chunker = RecursiveCharacterChunker(chunk_size=1000, chunk_overlap=200)
        if vector_store is None:
            vector_store = getattr(self.retriever, "vector_store", None)
            if vector_store is None:
                raise ValueError(
                    "ingest() needs a vector_store; the retriever does not expose one. "
                    "Pass vector_store=... explicitly."
                )

        documents = loader.load(source)
        chunks = chunker.split_documents(documents)
        return vector_store.add_documents(chunks, ids=ids)

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def _retrieve_and_rerank(self, question: str, k: int) -> list[VectorSearchResult]:
        results = self.retriever.retrieve(question, k=k)
        if self.reranker is not None and results:
            results = _apply_reranker_sync(self.reranker, question, results, self.top_n_after_rerank)
        return results

    def _format_prompt(self, question: str, results: list[VectorSearchResult]) -> str:
        return self.prompt_template.format(context=_format_context(results), question=question)

    def _llm_generate(self, prompt: str) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.system_prompt:
            options["system_prompt"] = self.system_prompt
        return self.llm.generate(prompt, options)

    def query(
        self,
        question: str,
        *,
        k: int = 4,
        return_sources: bool = True,
    ) -> RAGAnswer:
        """Retrieve relevant chunks and ask the LLM, returning a :class:`RAGAnswer`."""
        results = self._retrieve_and_rerank(question, k=k)
        prompt = self._format_prompt(question, results)
        response = self._llm_generate(prompt)
        answer_text = response.get("text", "") if isinstance(response, dict) else str(response)
        usage = response.get("usage") if isinstance(response, dict) else {}
        if usage is None and isinstance(response, dict):
            usage = {
                k: response[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost") if k in response
            }
        return RAGAnswer(
            answer=answer_text,
            sources=[r.document for r in results] if return_sources else [],
            retrieval_results=results,
            usage=usage or {},
        )

    def extract(
        self,
        question: str,
        model_class: type,
        *,
        k: int = 4,
    ) -> tuple[Any, RAGAnswer]:
        """Run RAG retrieval, then structured extraction over the retrieved context.

        The retrieved chunk text is used as the extraction source; the
        ``question`` becomes the instruction template.  Returns a tuple
        of ``(pydantic_instance, RAGAnswer)``.  The :class:`RAGAnswer`
        carries the raw LLM JSON in ``.answer`` for transparency.
        """
        from ..extraction.core import extract_with_model

        results = self._retrieve_and_rerank(question, k=k)
        context = _format_context(results)

        # Resolve the LLM driver back to a model_name string for
        # extract_with_model — fall back to passing the driver via options
        # if the driver doesn't carry a model_name.
        model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None)
        if not model_name:
            raise ValueError(
                "RAGPipeline.extract() requires self.llm to expose `.model_name` "
                "or `.model` so extract_with_model can route the call."
            )

        extraction = extract_with_model(
            model_class,
            context,
            model_name=model_name,
            instruction_template=question,
            system_prompt=self.system_prompt,
        )
        instance = extraction.get("model") if isinstance(extraction, dict) else extraction
        usage = extraction.get("usage", {}) if isinstance(extraction, dict) else {}

        # Serialize the structured answer back to a string for the RAGAnswer.
        try:
            answer_str = instance.model_dump_json() if instance is not None else ""
        except Exception:
            answer_str = str(instance)

        rag_answer = RAGAnswer(
            answer=answer_str,
            sources=[r.document for r in results],
            retrieval_results=results,
            usage=usage,
        )
        return instance, rag_answer


# ── Async pipeline ───────────────────────────────────────────────────────────


class AsyncRAGPipeline:
    """Async counterpart to :class:`RAGPipeline`.

    Mirrors the sync API with ``aingest``, ``aquery``, and ``aextract``.
    Native-async subcomponents (async retrievers / async LLMs) take
    precedence; sync subcomponents are awaited via
    :func:`asyncio.to_thread`.
    """

    def __init__(
        self,
        retriever: Retriever | AsyncRetriever,
        llm: Any,
        *,
        reranker: Any = None,
        prompt_template: str = DEFAULT_RAG_PROMPT,
        top_n_after_rerank: int = 4,
        system_prompt: str | None = None,
    ) -> None:
        """Construct an async RAG pipeline.

        Args:
            retriever: Sync or async retriever; the pipeline detects and adapts.
            llm: A sync or async LLM driver.
            reranker: Optional reranker. Accepts a pre-instantiated rerank
                driver (sync or async) or a ``"provider/model"`` string. When
                given a string, the async rerank driver is preferred and a
                sync one is used as a fallback.
            prompt_template: Template containing ``{context}`` and ``{question}``.
            top_n_after_rerank: How many results to keep after reranking.
            system_prompt: Optional system prompt for the LLM call.
        """
        self.retriever = retriever
        self.llm = llm
        self.reranker = _resolve_reranker(reranker, prefer_async=True)
        self.prompt_template = prompt_template
        self.top_n_after_rerank = top_n_after_rerank
        self.system_prompt = system_prompt

    async def _aretrieve(self, question: str, k: int) -> list[VectorSearchResult]:
        retrieve_fn = self.retriever.retrieve
        if asyncio.iscoroutinefunction(retrieve_fn):
            return await retrieve_fn(question, k=k)
        return await asyncio.to_thread(retrieve_fn, question, k=k)

    async def _aretrieve_and_rerank(self, question: str, k: int) -> list[VectorSearchResult]:
        results = await self._aretrieve(question, k=k)
        if self.reranker is not None and results:
            results = await _apply_reranker_async(self.reranker, question, results, self.top_n_after_rerank)
        return results

    async def _allm_generate(self, prompt: str) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.system_prompt:
            options["system_prompt"] = self.system_prompt
        gen_fn = self.llm.generate
        if asyncio.iscoroutinefunction(gen_fn):
            return await gen_fn(prompt, options)
        return await asyncio.to_thread(gen_fn, prompt, options)

    async def aingest(
        self,
        source: Any,
        *,
        loader: Any = None,
        chunker: Any = None,
        vector_store: Any = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Async version of :meth:`RAGPipeline.ingest`."""
        if loader is None:
            from .loader_registry import get_loader_for_path

            loader = get_loader_for_path(str(source))
        if chunker is None:
            from .chunkers import RecursiveCharacterChunker

            chunker = RecursiveCharacterChunker(chunk_size=1000, chunk_overlap=200)
        if vector_store is None:
            vector_store = getattr(self.retriever, "vector_store", None)
            if vector_store is None:
                raise ValueError("aingest() needs a vector_store; the retriever does not expose one.")

        load_fn = loader.load
        documents = (
            await load_fn(source) if asyncio.iscoroutinefunction(load_fn) else await asyncio.to_thread(load_fn, source)
        )
        split_fn = chunker.split_documents
        chunks = (
            await split_fn(documents)
            if asyncio.iscoroutinefunction(split_fn)
            else await asyncio.to_thread(split_fn, documents)
        )
        add_fn = vector_store.add_documents
        if asyncio.iscoroutinefunction(add_fn):
            return await add_fn(chunks, ids=ids)
        return await asyncio.to_thread(add_fn, chunks, ids)

    async def aquery(
        self,
        question: str,
        *,
        k: int = 4,
        return_sources: bool = True,
    ) -> RAGAnswer:
        """Async version of :meth:`RAGPipeline.query`."""
        results = await self._aretrieve_and_rerank(question, k=k)
        prompt = self.prompt_template.format(context=_format_context(results), question=question)
        response = await self._allm_generate(prompt)
        answer_text = response.get("text", "") if isinstance(response, dict) else str(response)
        usage = response.get("usage") if isinstance(response, dict) else {}
        if usage is None and isinstance(response, dict):
            usage = {
                k: response[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost") if k in response
            }
        return RAGAnswer(
            answer=answer_text,
            sources=[r.document for r in results] if return_sources else [],
            retrieval_results=results,
            usage=usage or {},
        )

    async def aextract(
        self,
        question: str,
        model_class: type,
        *,
        k: int = 4,
    ) -> tuple[Any, RAGAnswer]:
        """Async version of :meth:`RAGPipeline.extract`."""
        from ..extraction.async_core import extract_with_model as async_extract_with_model

        results = await self._aretrieve_and_rerank(question, k=k)
        context = _format_context(results)
        model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None)
        if not model_name:
            raise ValueError("AsyncRAGPipeline.aextract() requires self.llm to expose `.model_name` or `.model`.")
        extraction = await async_extract_with_model(
            model_class,
            context,
            model_name=model_name,
            instruction_template=question,
            system_prompt=self.system_prompt,
        )
        instance = extraction.get("model") if isinstance(extraction, dict) else extraction
        usage = extraction.get("usage", {}) if isinstance(extraction, dict) else {}
        try:
            answer_str = instance.model_dump_json() if instance is not None else ""
        except Exception:
            answer_str = str(instance)
        rag_answer = RAGAnswer(
            answer=answer_str,
            sources=[r.document for r in results],
            retrieval_results=results,
            usage=usage,
        )
        return instance, rag_answer
