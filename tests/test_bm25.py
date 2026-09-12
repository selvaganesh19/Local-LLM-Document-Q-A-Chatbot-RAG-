"""Tests for the BM25 lexical index."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_text
from app.retrieval.bm25_index import BM25Index, tokenize


@pytest.fixture
def index(sample_chunks) -> BM25Index:
    """A BM25 index populated with the sample corpus."""
    built = BM25Index()
    built.add_chunks(sample_chunks)
    return built


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    """Common words and single characters never reach the index."""
    assert tokenize("The Quick Brown FOX") == ["quick", "brown", "fox"]
    assert tokenize("a I of and the") == []


def test_tokenize_keeps_identifiers_and_numbers() -> None:
    """Codes and numbers survive, which is why BM25 complements embeddings."""
    assert "http" in tokenize("HTTP 429 Retry-After")
    assert "3600" in tokenize("tokens expire after 3600 seconds")


def test_search_finds_the_rare_token() -> None:
    """A distinctive literal token retrieves its document at rank one."""
    results = BM25Index()
    results.add_chunks(
        chunk_text(
            "Access tokens expire after 3600 seconds and refresh tokens last 30 days.",
            source="api.txt",
            chunk_size=500,
        )
        + chunk_text(
            "Employees may work remotely up to three days per week.",
            source="handbook.txt",
            chunk_size=500,
        )
    )

    hits = results.search("3600 seconds", top_k=5)

    assert hits
    assert hits[0].metadata["source"] == "api.txt"


def test_search_returns_normalised_scores() -> None:
    """Scores are scaled so the best hit reports 1.0."""
    index = BM25Index()
    index.add_chunks(chunk_text("alpha beta gamma delta", source="a.txt", chunk_size=500))

    hits = index.search("alpha", top_k=3)

    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].raw_score > 0


def test_search_on_empty_index_returns_nothing() -> None:
    """An unpopulated index never raises."""
    assert BM25Index().search("anything", top_k=5) == []


def test_search_with_stopword_only_query_returns_nothing(index: BM25Index) -> None:
    """A query with no content words cannot match anything."""
    assert index.search("the and of", top_k=5) == []


def test_search_respects_top_k(index: BM25Index) -> None:
    """The result list is capped."""
    assert len(index.search("notice period days", top_k=1)) <= 1


def test_add_chunks_is_idempotent(index: BM25Index) -> None:
    """Re-adding the same chunk replaces rather than duplicates it."""
    before = index.count
    records = list(index.iter_records())

    added = index.add_chunks(chunk_text(records[0][1], source=records[0][2]["source"], chunk_size=220))

    assert index.count == before
    assert added == 0


def test_delete_by_source_removes_only_that_source(index: BM25Index) -> None:
    """Deletion is scoped to one document."""
    query = "authentication multi factor"  # vocabulary unique to security.txt

    assert index.search(query, top_k=5), "precondition: the query should match"

    removed = index.delete_by_source("security.txt")

    assert removed > 0
    assert all(record[2]["source"] != "security.txt" for record in index.iter_records())
    assert index.search(query, top_k=5) == []


def test_delete_unknown_source_is_a_noop(index: BM25Index) -> None:
    """Deleting something absent reports zero rather than raising."""
    assert index.delete_by_source("missing.txt") == 0


def test_list_sources_summarises_documents(index: BM25Index) -> None:
    """Source aggregation reports per-document chunk counts."""
    sources = {entry["source"] for entry in index.list_sources()}

    assert sources == {"handbook.txt", "security.txt", "api.txt"}
    assert all(entry["chunks"] > 0 for entry in index.list_sources())


def test_save_and_load_round_trip(index: BM25Index, tmp_path: Path) -> None:
    """A persisted index behaves identically after reloading."""
    path = tmp_path / "bm25.pkl"
    index.save(path)

    restored = BM25Index.load(path)

    assert restored.count == index.count
    assert [hit.chunk_id for hit in restored.search("notice period", top_k=3)] == [
        hit.chunk_id for hit in index.search("notice period", top_k=3)
    ]


def test_load_missing_file_yields_empty_index(tmp_path: Path) -> None:
    """A missing cache file is not an error; re-ingestion repopulates it."""
    assert BM25Index.load(tmp_path / "absent.pkl").count == 0


def test_load_corrupt_file_yields_empty_index(tmp_path: Path) -> None:
    """A corrupt cache must not stop the application from starting."""
    path = tmp_path / "corrupt.pkl"
    path.write_bytes(b"not a pickle at all")

    assert BM25Index.load(path).count == 0


def test_reset_clears_everything(index: BM25Index) -> None:
    """Reset empties the index and its search results."""
    index.reset()

    assert index.count == 0
    assert index.search("notice", top_k=5) == []
