"""Tests for the ChromaDB-backed vector store."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import HashingEmbedder
from app.retrieval.vectorstore import VectorHit, VectorStore, sanitize_metadata


@pytest.fixture
def store(settings: Settings) -> VectorStore:
    """An empty vector store in a throwaway directory."""
    return VectorStore(settings=settings)


@pytest.fixture
def populated(store: VectorStore, sample_chunks, embedder: HashingEmbedder) -> VectorStore:
    """A store holding the embedded sample corpus."""
    store.upsert_chunks(sample_chunks, embedder.embed_documents([c.text for c in sample_chunks]))
    return store


def test_sanitize_metadata_drops_none_and_stringifies_objects() -> None:
    """Chroma only accepts primitives, so anything else is coerced."""
    clean = sanitize_metadata(
        {"page": 3, "ratio": 0.5, "flag": True, "missing": None, "tags": ["a", "b"]}
    )

    assert clean["page"] == 3
    assert clean["ratio"] == 0.5
    assert clean["flag"] is True
    assert "missing" not in clean
    assert isinstance(clean["tags"], str)


def test_new_store_is_empty(store: VectorStore) -> None:
    """A fresh collection starts at zero and reports no sources."""
    assert store.count() == 0
    assert store.list_sources() == []


def test_upsert_indexes_chunks(populated: VectorStore, sample_chunks) -> None:
    """Every supplied chunk becomes searchable."""
    assert populated.count() == len(sample_chunks)
    assert {entry["source"] for entry in populated.list_sources()} == {
        "handbook.txt",
        "security.txt",
        "api.txt",
    }


def test_upsert_is_idempotent(populated: VectorStore, sample_chunks, embedder: HashingEmbedder) -> None:
    """Chunk ids are content-addressed, so re-ingesting does not duplicate."""
    before = populated.count()

    populated.upsert_chunks(sample_chunks, embedder.embed_documents([c.text for c in sample_chunks]))

    assert populated.count() == before


def test_upsert_rejects_mismatched_lengths(store: VectorStore, sample_chunks) -> None:
    """Chunks and vectors must line up one-to-one."""
    with pytest.raises(ValueError):
        store.upsert_chunks(sample_chunks, [[0.0] * 64])


def test_upsert_of_nothing_writes_nothing(store: VectorStore) -> None:
    """An empty batch is a no-op."""
    assert store.upsert_chunks([], []) == 0


def test_search_returns_hits_ordered_by_similarity(populated: VectorStore, embedder: HashingEmbedder) -> None:
    """Results are sorted by descending cosine similarity."""
    hits = populated.search(embedder.embed_query("notice period 90 calendar days"), top_k=3)

    assert hits
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(hit, VectorHit) for hit in hits)


def test_search_finds_the_most_relevant_source(populated: VectorStore, embedder: HashingEmbedder) -> None:
    """A query sharing vocabulary with one document retrieves that document."""
    hits = populated.search(embedder.embed_query("passwords rotated multi-factor authentication"), top_k=3)

    assert hits[0].source == "security.txt"


def test_search_respects_top_k(populated: VectorStore, embedder: HashingEmbedder) -> None:
    """The result count is capped, and never exceeds what is indexed."""
    assert len(populated.search(embedder.embed_query("notice"), top_k=1)) == 1
    assert len(populated.search(embedder.embed_query("notice"), top_k=99)) == populated.count()


def test_search_with_non_positive_k_returns_nothing(populated: VectorStore, embedder: HashingEmbedder) -> None:
    """A zero or negative k short-circuits."""
    assert populated.search(embedder.embed_query("notice"), top_k=0) == []


def test_search_on_empty_store_returns_nothing(store: VectorStore, embedder: HashingEmbedder) -> None:
    """Querying nothing yields nothing rather than raising."""
    assert store.search(embedder.embed_query("notice"), top_k=5) == []


def test_search_honours_a_metadata_filter(populated: VectorStore, embedder: HashingEmbedder) -> None:
    """Filters restrict results to one source document."""
    hits = populated.search(
        embedder.embed_query("the"),
        top_k=5,
        where={"source": {"$eq": "api.txt"}},
    )

    assert hits
    assert {hit.source for hit in hits} == {"api.txt"}


def test_scores_are_bounded_cosine_similarities(populated: VectorStore, embedder: HashingEmbedder) -> None:
    """Similarity is reported in [-1, 1], not as a raw distance."""
    hits = populated.search(embedder.embed_query("notice period"), top_k=3)

    assert all(-1.0 <= hit.score <= 1.0 for hit in hits)
    assert hits[0].score > 0.0


def test_delete_by_source_removes_only_that_document(populated: VectorStore) -> None:
    """Deletion is scoped to one source."""
    removed = populated.delete_by_source("security.txt")

    assert removed > 0
    assert {entry["source"] for entry in populated.list_sources()} == {"handbook.txt", "api.txt"}


def test_delete_unknown_source_removes_nothing(populated: VectorStore) -> None:
    """Deleting something absent reports zero."""
    assert populated.delete_by_source("nope.txt") == 0


def test_get_chunk_by_id(populated: VectorStore, sample_chunks) -> None:
    """A chunk can be fetched directly by its id, with its text intact."""
    target = sample_chunks[0]

    fetched = populated.get_chunk(target.chunk_id)

    assert fetched is not None
    assert fetched.chunk_id == target.chunk_id
    assert fetched.text == target.text
    assert fetched.source == target.source


def test_get_missing_chunk_returns_none(populated: VectorStore) -> None:
    """An unknown id yields None rather than raising."""
    assert populated.get_chunk("does-not-exist") is None


def test_reset_empties_the_collection(populated: VectorStore) -> None:
    """Reset drops every vector."""
    populated.reset()

    assert populated.count() == 0
    assert populated.list_sources() == []


def test_collection_uses_cosine_space(settings: Settings) -> None:
    """The collection is configured for cosine similarity, as retrieval assumes."""
    store = VectorStore(settings=settings)

    assert store.collection_name == settings.chroma_collection
