"""Tests for the hybrid retriever, exercised against a real index."""

from __future__ import annotations

import pytest

from app.container import ServiceContainer


def test_retrieve_returns_ranked_chunks(ingested_container: ServiceContainer) -> None:
    """A query over an indexed corpus produces ordered candidates."""
    result = ingested_container.retriever.retrieve("What is the notice period for senior staff?", top_k=3)

    assert result.chunks
    assert len(result.chunks) <= 3
    assert result.total_ms >= 0
    assert result.is_empty is False


def test_retrieve_populates_both_legs(ingested_container: ServiceContainer) -> None:
    """Both dense and lexical retrieval run for every query."""
    result = ingested_container.retriever.retrieve("authentication password rotation", top_k=3)

    assert result.dense_hits, "expected vector results"
    assert "dense_ms" in result.timings
    assert "sparse_ms" in result.timings
    assert "fusion_ms" in result.timings
    assert "rerank_ms" in result.timings
    assert "embedding_ms" in result.timings


def test_retrieve_ranks_the_matching_document_first(ingested_container: ServiceContainer) -> None:
    """Vocabulary overlap with one document pulls that document to the top."""
    result = ingested_container.retriever.retrieve("HTTP 429 Retry-After rate limit", top_k=3)

    assert result.chunks
    assert result.chunks[0].source == "api.txt"


def test_retrieve_marks_fused_candidates(ingested_container: ServiceContainer) -> None:
    """Candidates carry a fusion score and a display relevance."""
    result = ingested_container.retriever.retrieve("notice period", top_k=3)

    for chunk in result.chunks:
        assert chunk.fusion_score > 0
        assert 0.0 <= chunk.relevance <= 1.0


def test_retrieve_with_a_source_filter(ingested_container: ServiceContainer) -> None:
    """Filtering restricts results to one document."""
    result = ingested_container.retriever.retrieve(
        "rate limit requests", top_k=5, source_filter="handbook.txt"
    )

    assert result.chunks, "expected the filtered document to be retrievable"
    assert all(chunk.source == "handbook.txt" for chunk in result.chunks)


def test_retrieve_on_empty_query_returns_nothing(ingested_container: ServiceContainer) -> None:
    """A blank query short-circuits without touching the indexes."""
    result = ingested_container.retriever.retrieve("   ", top_k=3)

    assert result.is_empty
    assert result.timings == {}


def test_retrieve_on_an_empty_index(container: ServiceContainer) -> None:
    """With nothing indexed retrieval returns nothing rather than raising."""
    result = container.retriever.retrieve("anything at all", top_k=3)

    assert result.is_empty


def test_retrieve_respects_top_k(ingested_container: ServiceContainer) -> None:
    """The returned candidate count honours the request."""
    assert len(ingested_container.retriever.retrieve("notice", top_k=1).chunks) == 1
    assert len(ingested_container.retriever.retrieve("notice", top_k=3).chunks) <= 3


def test_index_stats_summarises_state(ingested_container: ServiceContainer) -> None:
    """Stats report matching counts, the collection name and the reranker."""
    stats = ingested_container.retriever.index_stats()

    assert stats["vectors"] == stats["lexical_chunks"] == 3
    assert stats["collection"] == "test_documents"
    assert stats["reranker"] == "none"


def test_deleted_document_is_no_longer_retrievable(ingested_container: ServiceContainer) -> None:
    """Deletion removes a document from both retrieval legs."""
    ingested_container.pipeline.delete_source("api.txt")

    result = ingested_container.retriever.retrieve("HTTP 429 Retry-After rate limit", top_k=5)

    assert all(chunk.source != "api.txt" for chunk in result.chunks)


@pytest.mark.asyncio
async def test_service_answers_and_caches(ingested_container: ServiceContainer) -> None:
    """The service layer caches an answer and replays it on a repeat."""
    from app.services.rag_service import RagService

    service = RagService(ingested_container)
    question = "What is the notice period for senior staff?"

    first = await service.answer(question)
    second = await service.answer(question)

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]
    assert second["trace_id"] == first["trace_id"]


@pytest.mark.asyncio
async def test_service_invalidates_the_cache_after_ingestion(ingested_container: ServiceContainer) -> None:
    """Ingesting new material drops stale cached answers."""
    from app.services.rag_service import RagService

    service = RagService(ingested_container)
    question = "What is the notice period for senior staff?"

    await service.answer(question)
    assert ingested_container.cache.stats()["entries"] == 1

    service.ingest_text("A newly added document about bicycles.", "bikes.txt")

    assert ingested_container.cache.stats()["entries"] == 0


@pytest.mark.asyncio
async def test_service_returns_a_refusal_on_an_empty_index(container: ServiceContainer) -> None:
    """With nothing indexed the model is never called."""
    from app.services.rag_service import RagService

    result = await RagService(container).answer("Any question at all")

    assert result["insufficient_context"] is True
    assert result["reason"] == "no_documents_retrieved"
    assert container.client.calls == []


@pytest.mark.asyncio
async def test_service_does_not_cache_uncited_answers(ingested_container: ServiceContainer) -> None:
    """An answer the model produced without citations is never cached."""
    from app.generation.ollama_client import ChatResponse, GenerationStats
    from app.services.rag_service import RagService

    async def uncited_chat(messages, temperature=None, num_ctx=None):
        """Return an answer containing no citation markers."""
        return ChatResponse(
            content="An answer with no citations whatsoever.",
            stats=GenerationStats(model="fake-model"),
        )

    ingested_container.client.chat = uncited_chat  # type: ignore[method-assign]
    service = RagService(ingested_container)

    result = await service.answer("What is the notice period?")

    assert result["uncited"] is True
    assert ingested_container.cache.stats()["entries"] == 0
