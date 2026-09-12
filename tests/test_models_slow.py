"""Smoke tests for the real embedding and reranking models.

These are the only tests that download weights, so they are marked ``slow`` and
skipped when ``sentence-transformers`` is not installed. Run them once after a
fresh setup to confirm the model paths in configuration actually resolve:

    pytest -m slow

Everything else in the suite uses a deterministic stub embedder so it stays
hermetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.hybrid import RetrievedChunk

#: Excluded from the default run; execute these with `pytest -m slow`.
#:
#: The dependency check lives inside the fixtures rather than at module scope on
#: purpose: importing ``sentence-transformers`` pulls in torch and costs several
#: seconds, which the default run should not pay for tests it has deselected.
pytestmark = pytest.mark.slow

#: Two passages with deliberately different vocabulary.
NOTICE_PASSAGE = "The notice period for senior staff is 90 calendar days."
PASSWORD_PASSAGE = "Passwords must be at least 14 characters and rotate every 180 days."


@pytest.fixture(scope="module")
def embedder():
    """The real embedding model named in configuration."""
    pytest.importorskip("sentence_transformers", reason="requires sentence-transformers")
    from app.ingestion.embedder import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", "cpu", 384)


@pytest.fixture(scope="module")
def reranker():
    """The real cross-encoder named in configuration."""
    pytest.importorskip("sentence_transformers", reason="requires sentence-transformers")
    from app.retrieval.reranker import CrossEncoderReranker

    return CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", "cpu")


def test_embedder_reports_the_configured_dimension(embedder) -> None:
    """The model loads and agrees with the configured vector width."""
    assert embedder.dimension == 384
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"


def test_embeddings_are_l2_normalised(embedder) -> None:
    """Cosine distance in the vector store relies on normalised vectors."""
    vectors = embedder.embed_documents([NOTICE_PASSAGE, PASSWORD_PASSAGE])

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_empty_input_produces_an_empty_matrix(embedder) -> None:
    """An empty batch short-circuits without loading anything."""
    assert embedder.embed_documents([]).shape == (0, 384)


def test_semantic_similarity_matches_topical_relevance(embedder) -> None:
    """A question about notice should sit closer to the notice passage."""
    documents = embedder.embed_documents([NOTICE_PASSAGE, PASSWORD_PASSAGE])
    query = embedder.embed_query("How long is the notice period?")

    similarities = documents @ query

    assert similarities[0] > similarities[1]


def test_semantic_similarity_beats_lexical_overlap(embedder) -> None:
    """Paraphrase is the reason for dense retrieval; check it actually happens."""
    documents = embedder.embed_documents(
        ["Employees must give three months of notice.", "Dogs are not permitted in the office."]
    )
    # Shares almost no vocabulary with the first passage.
    query = embedder.embed_query("How much warning must a leaver provide?")

    similarities = documents @ query

    assert similarities[0] > similarities[1]


def test_reranker_scores_forward_passes(reranker) -> None:
    """The cross-encoder produces one finite score per candidate."""
    chunks = [
        RetrievedChunk(chunk_id="a", text=NOTICE_PASSAGE, metadata={"source": "a.txt"}),
        RetrievedChunk(chunk_id="b", text=PASSWORD_PASSAGE, metadata={"source": "b.txt"}),
    ]

    ordered = reranker.rerank("How long is the notice period?", chunks, top_n=2)

    assert len(ordered) == 2
    assert all(chunk.rerank_score is not None for chunk in ordered)
    assert all(np.isfinite(chunk.rerank_score) for chunk in ordered)


def test_reranker_promotes_the_relevant_passage(reranker) -> None:
    """Reranking reorders candidates by true relevance to the question."""
    chunks = [
        RetrievedChunk(chunk_id="irrelevant", text=PASSWORD_PASSAGE, metadata={"source": "b.txt"}),
        RetrievedChunk(chunk_id="relevant", text=NOTICE_PASSAGE, metadata={"source": "a.txt"}),
    ]

    ordered = reranker.rerank("How long is the notice period?", chunks, top_n=2)

    assert ordered[0].chunk_id == "relevant"
    assert 0.0 <= ordered[0].relevance <= 1.0


def test_reranker_truncates_to_top_n(reranker) -> None:
    """Only the requested number of candidates survive."""
    chunks = [
        RetrievedChunk(chunk_id=f"c{index}", text=text, metadata={"source": "x.txt"})
        for index, text in enumerate([NOTICE_PASSAGE, PASSWORD_PASSAGE, "Unrelated filler text."])
    ]

    assert len(reranker.rerank("notice period", chunks, top_n=2)) == 2
