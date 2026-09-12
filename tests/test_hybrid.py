"""Tests for hybrid fusion and reranker selection."""

from __future__ import annotations

import pytest

from app.retrieval.bm25_index import BM25Hit
from app.retrieval.hybrid import (
    RetrievedChunk,
    fuse_hits,
    reciprocal_rank_fusion,
)
from app.retrieval.reranker import NoOpReranker, sigmoid
from app.retrieval.vectorstore import VectorHit


def _dense(chunk_id: str, score: float) -> VectorHit:
    """Build a dense hit for testing."""
    return VectorHit(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        metadata={"source": f"{chunk_id}.txt"},
        score=score,
    )


def _sparse(chunk_id: str, score: float) -> BM25Hit:
    """Build a lexical hit for testing."""
    return BM25Hit(
        chunk_id=chunk_id,
        text=f"text of {chunk_id}",
        metadata={"source": f"{chunk_id}.txt"},
        score=score,
        raw_score=score * 10,
    )


def test_rrf_scores_match_the_formula() -> None:
    """Fused scores equal the sum of 1/(k + rank) across retrievers."""
    fused = reciprocal_rank_fusion({"dense": ["a", "b"], "sparse": ["b", "c"]}, k=60)

    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_ranks_agreement_highest() -> None:
    """A document both retrievers rank highly beats one only one of them found."""
    fused = reciprocal_rank_fusion({"dense": ["shared", "x"], "sparse": ["shared", "y"]}, k=60)

    assert fused["shared"] > fused["x"]
    assert fused["shared"] > fused["y"]


def test_rrf_rejects_non_positive_k() -> None:
    """A nonsensical damping constant is rejected."""
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": ["a"]}, k=0)


def test_rrf_handles_empty_rankings() -> None:
    """No retrievers means no scores."""
    assert reciprocal_rank_fusion({}, k=60) == {}
    assert reciprocal_rank_fusion({"dense": []}, k=60) == {}


def test_fuse_hits_preserves_both_score_sets() -> None:
    """A chunk found by both retrievers keeps both scores and both ranks."""
    fused = fuse_hits([_dense("a", 0.9), _dense("b", 0.5)], [_sparse("a", 0.8)], k=60)

    top = fused[0]
    assert top.chunk_id == "a"
    assert top.dense_rank == 1
    assert top.sparse_rank == 1
    assert top.dense_score == pytest.approx(0.9)
    assert top.sparse_score == pytest.approx(0.8)
    assert top.source == "a.txt"


def test_fuse_hits_orders_by_fusion_score() -> None:
    """Ordering follows the fused score, highest first."""
    fused = fuse_hits(
        [_dense("a", 0.9), _dense("b", 0.8)],
        [_sparse("b", 0.9)],
        k=60,
    )

    # 'b' appears in both lists, so it fuses above 'a'.
    assert [chunk.chunk_id for chunk in fused] == ["b", "a"]


def test_fuse_hits_respects_the_limit() -> None:
    """The candidate cap is applied after ordering."""
    fused = fuse_hits([_dense(c, 1.0) for c in ("a", "b", "c")], [], k=60, limit=2)

    assert len(fused) == 2


def test_fuse_hits_is_deterministic_on_ties() -> None:
    """Equal fusion scores break ties on dense rank, then id."""
    first = fuse_hits([_dense("a", 1.0), _dense("b", 1.0)], [], k=60)
    second = fuse_hits([_dense("a", 1.0), _dense("b", 1.0)], [], k=60)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second] == ["a", "b"]


def test_fuse_hits_with_no_input_is_empty() -> None:
    """Nothing retrieved means nothing fused."""
    assert fuse_hits([], [], k=60) == []


def test_noop_reranker_truncates_and_normalises() -> None:
    """With reranking disabled the fusion order is kept and scaled to 0..1."""
    candidates = [
        RetrievedChunk(chunk_id="a", text="a", fusion_score=0.5),
        RetrievedChunk(chunk_id="b", text="b", fusion_score=0.25),
        RetrievedChunk(chunk_id="c", text="c", fusion_score=0.1),
    ]

    selected = NoOpReranker().rerank("query", candidates, top_n=2)

    assert [chunk.chunk_id for chunk in selected] == ["a", "b"]
    assert selected[0].relevance == pytest.approx(1.0)
    assert selected[1].relevance == pytest.approx(0.5)


def test_noop_reranker_handles_empty_input() -> None:
    """No candidates means no output."""
    assert NoOpReranker().rerank("query", [], top_n=3) == []
    assert NoOpReranker().model_name == "none"


def test_sigmoid_bounds_and_monotonicity() -> None:
    """The relevance transform stays in (0, 1) and preserves ordering."""
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-30.0) < 0.001
    assert 0.999 < sigmoid(30.0) < 1.0
    assert sigmoid(2.0) > sigmoid(1.0)


def test_retrieved_chunk_to_dict_round_trips_fields() -> None:
    """Serialisation exposes the fields the API and UI rely on."""
    chunk = RetrievedChunk(
        chunk_id="abc",
        text="body",
        metadata={"source": "doc.pdf", "page": 4},
        dense_rank=1,
        dense_score=0.8,
        relevance=0.77,
    )

    payload = chunk.to_dict()

    assert payload["chunk_id"] == "abc"
    assert payload["source"] == "doc.pdf"
    assert payload["page"] == 4
    assert payload["relevance"] == pytest.approx(0.77)
