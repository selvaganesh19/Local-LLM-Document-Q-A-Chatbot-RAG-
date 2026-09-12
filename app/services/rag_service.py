"""Application service layer.

Sits between the HTTP routes and the retrieval/generation machinery, and owns
the pieces of policy that do not belong to either:

* semantic-cache look-up and population,
* the short-circuit when retrieval returns nothing (never call the model with
  an empty context),
* cache invalidation whenever the corpus changes,
* assembling the single response shape returned by both the blocking and
  streaming endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from app.container import ServiceContainer
from app.generation.generator import GeneratedAnswer
from app.generation.prompts import INSUFFICIENT_CONTEXT_ANSWER
from app.ingestion.loaders import Document
from app.observability import tracing

logger = logging.getLogger(__name__)


class RagService:
    """Coordinates retrieval, caching and grounded generation."""

    def __init__(self, container: ServiceContainer) -> None:
        """Bind the service to a container.

        Args:
            container: Holds the retriever, generator and cache.
        """
        self._container = container

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _empty_answer_payload(question: str, reason: str) -> Dict[str, Any]:
        """Build the response used when there is nothing to ground an answer in."""
        return {
            "question": question,
            "answer": INSUFFICIENT_CONTEXT_ANSWER,
            "citations": [],
            "retrieved": [],
            "insufficient_context": True,
            "uncited": False,
            "grounded": True,
            "cached": False,
            "cache_similarity": None,
            "matched_query": None,
            "model": "",
            "trace_id": tracing.current_trace_id(),
            "timings": {},
            "stats": {},
            "reason": reason,
        }

    def _build_payload(
        self,
        question: str,
        result: GeneratedAnswer,
        timings: Dict[str, float],
    ) -> Dict[str, Any]:
        """Merge generation output, timings and trace id into a response body."""
        return {
            "question": question,
            "answer": result.answer,
            "citations": [citation.to_dict() for citation in result.citations],
            "retrieved": list(result.retrieved),
            "insufficient_context": result.insufficient_context,
            "uncited": result.uncited,
            "grounded": result.grounded,
            "cached": False,
            "cache_similarity": None,
            "matched_query": None,
            "model": self._container.client.model,
            "trace_id": tracing.current_trace_id(),
            "timings": {key: round(value, 2) for key, value in timings.items()},
            "stats": result.stats.to_dict(),
            "reason": None,
        }

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------
    async def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Answer ``question`` from the indexed corpus.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve; defaults to ``TOP_K_FINAL``.
            source_filter: Restrict retrieval to a single source document.
            use_cache: Consult and populate the semantic cache.

        Returns:
            The response body shared by the chat endpoints.

        Raises:
            OllamaError: If the model is unreachable or misconfigured.
        """
        with tracing.trace_query(question):
            if use_cache:
                hit = self._container.cache.get(question)
                if hit is not None:
                    payload = dict(hit.payload)
                    payload.update(
                        {
                            "cached": True,
                            "cache_similarity": round(hit.similarity, 4),
                            "matched_query": hit.matched_query,
                            "trace_id": tracing.current_trace_id(),
                        }
                    )
                    logger.info(
                        "Served answer from semantic cache",
                        extra={"similarity": round(hit.similarity, 4), "age_seconds": hit.age_seconds},
                    )
                    return payload

            retrieval = self._container.retriever.retrieve(
                question, top_k=top_k, source_filter=source_filter
            )
            timings: Dict[str, float] = dict(retrieval.timings)

            if retrieval.is_empty:
                logger.info("No chunks retrieved - skipping generation")
                payload = self._empty_answer_payload(question, "no_documents_retrieved")
                payload["cached"] = False
                return payload

            with tracing.Stopwatch() as generation_timer:
                generated = await self._container.generator.generate(question, retrieval.chunks)
            timings["generation_ms"] = generation_timer.elapsed_ms
            timings["total_ms"] = retrieval.total_ms + generation_timer.elapsed_ms

            payload = self._build_payload(question, generated, timings)

            if use_cache and not generated.uncited:
                # Never cache an answer the model produced without citations;
                # caching an ungrounded response would propagate it.
                self._container.cache.set(question, payload)

            return payload

    async def stream_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream an answer as event dicts.

        The cache is not consulted for streaming requests because the value of
        this endpoint is watching tokens arrive; a cache hit is available
        instantly through the blocking endpoint.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve.
            source_filter: Restrict retrieval to a single source document.

        Yields:
            Event dicts mirroring the SSE protocol documented in the route.
        """
        with tracing.trace_query(question):
            retrieval = self._container.retriever.retrieve(
                question, top_k=top_k, source_filter=source_filter
            )

            yield {"type": "timings", "timings": {k: round(v, 2) for k, v in retrieval.timings.items()}}
            yield {"type": "trace", "trace_id": tracing.current_trace_id()}

            if retrieval.is_empty:
                logger.info("No chunks retrieved - skipping streamed generation")
                yield {
                    "type": "done",
                    "answer": INSUFFICIENT_CONTEXT_ANSWER,
                    "citations": [],
                    "retrieved": [],
                    "insufficient_context": True,
                    "uncited": False,
                    "grounded": True,
                    "cached": False,
                    "model": self._container.client.model,
                    "reason": "no_documents_retrieved",
                }
                return

            async for event in self._container.generator.stream(question, retrieval.chunks):
                yield event

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_documents(self, documents: Sequence[Document]) -> Dict[str, Any]:
        """Ingest pre-loaded documents and invalidate cached answers."""
        result = self._container.pipeline.ingest_documents(documents)
        self._invalidate_cache()
        return result.to_dict()

    def ingest_text(self, text: str, source_name: str) -> Dict[str, Any]:
        """Ingest raw text and invalidate cached answers."""
        result = self._container.pipeline.ingest_text(text, source_name)
        self._invalidate_cache()
        return result.to_dict()

    def ingest_files(self, paths: Sequence[str]) -> Dict[str, Any]:
        """Ingest files and invalidate cached answers."""
        result = self._container.pipeline.ingest_files(paths)
        self._invalidate_cache()
        return result.to_dict()

    def ingest_directory(self, directory: str, recursive: bool = True) -> Dict[str, Any]:
        """Ingest a directory and invalidate cached answers."""
        result = self._container.pipeline.ingest_directory(directory, recursive=recursive)
        self._invalidate_cache()
        return result.to_dict()

    def delete_source(self, source: str) -> Dict[str, Any]:
        """Delete a source document and invalidate cached answers."""
        result = self._container.pipeline.delete_source(source)
        self._invalidate_cache()
        return result

    def reset_index(self) -> Dict[str, Any]:
        """Drop the whole index and invalidate cached answers."""
        self._container.pipeline.reset()
        self._invalidate_cache()
        return {"status": "reset"}

    def _invalidate_cache(self) -> None:
        """Drop cached answers after the corpus changes."""
        removed = self._container.cache.clear()
        if removed:
            logger.info("Cache invalidated after corpus change", extra={"entries": removed})

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def list_documents(self) -> List[Dict[str, Any]]:
        """List indexed source documents with chunk counts."""
        return self._container.vector_store.list_sources()

    def index_stats(self) -> Dict[str, Any]:
        """Summarise index and cache state."""
        return {
            **self._container.retriever.index_stats(),
            "cache": self._container.cache.stats(),
        }
