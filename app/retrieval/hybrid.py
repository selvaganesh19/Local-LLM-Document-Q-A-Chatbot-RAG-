"""Hybrid retrieval: reciprocal rank fusion of dense and lexical results.

Reciprocal Rank Fusion (RRF) combines ranked lists using only rank position,
which makes it robust to the fact that BM25 scores and cosine similarities live
on completely different, incomparable scales.

    score(d) = sum over retrievers of 1 / (k + rank(d))

``k`` (default 60, from the original Cormack et al. paper) damps the influence
of top ranks so a single retriever cannot dominate the fused list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.retrieval.bm25_index import BM25Hit
from app.retrieval.vectorstore import VectorHit

logger = logging.getLogger(__name__)

#: Damping constant for reciprocal rank fusion.
DEFAULT_RRF_K = 60


@dataclass
class RetrievedChunk:
    """A candidate chunk carrying every score assigned to it.

    Attributes:
        chunk_id: Identifier of the chunk.
        text: Chunk text.
        metadata: Stored provenance used for citations.
        dense_rank: 1-based rank in the dense result list, if retrieved.
        sparse_rank: 1-based rank in the BM25 result list, if retrieved.
        dense_score: Cosine similarity from the vector store.
        sparse_score: Normalised BM25 score.
        fusion_score: Reciprocal rank fusion score.
        rerank_score: Raw cross-encoder score, once reranked.
        relevance: Display-friendly relevance in ``[0, 1]``.
    """

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fusion_score: float = 0.0
    rerank_score: Optional[float] = None
    relevance: float = 0.0

    @property
    def source(self) -> str:
        """Source document name recorded in metadata."""
        return str(self.metadata.get("source", "unknown"))

    @property
    def page(self) -> Optional[int]:
        """Page number, when the source was a paginated document."""
        page = self.metadata.get("page")
        return int(page) if isinstance(page, (int, float)) else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for API responses and trace payloads."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "text": self.text,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "relevance": self.relevance,
            "metadata": dict(self.metadata),
        }


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    k: int = DEFAULT_RRF_K,
) -> Dict[str, float]:
    """Fuse several ranked id lists into a single score per id.

    Args:
        rankings: Mapping of retriever name to its ordered chunk ids, best
            first.  Missing retrievers are simply skipped.
        k: RRF damping constant.

    Returns:
        Mapping of chunk id to fused score.  Ids absent from every ranking are
        absent from the result.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")

    scores: Dict[str, float] = {}
    for ids in rankings.values():
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse_hits(
    dense_hits: Sequence[VectorHit],
    sparse_hits: Sequence[BM25Hit],
    k: int = DEFAULT_RRF_K,
    limit: Optional[int] = None,
) -> List[RetrievedChunk]:
    """Merge dense and lexical hits into one ranked candidate list.

    Args:
        dense_hits: Vector-store results, best first.
        sparse_hits: BM25 results, best first.
        k: RRF damping constant.
        limit: Optional cap on the number of fused candidates.

    Returns:
        Candidates sorted by descending fusion score.  Ties break on the better
        dense rank, then on chunk id, so ordering is deterministic.
    """
    candidates: Dict[str, RetrievedChunk] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        candidates[hit.chunk_id] = RetrievedChunk(
            chunk_id=hit.chunk_id,
            text=hit.text,
            metadata=dict(hit.metadata),
            dense_rank=rank,
            dense_score=hit.score,
            relevance=hit.score,
        )

    for rank, hit in enumerate(sparse_hits, start=1):
        existing = candidates.get(hit.chunk_id)
        if existing is None:
            candidates[hit.chunk_id] = RetrievedChunk(
                chunk_id=hit.chunk_id,
                text=hit.text,
                metadata=dict(hit.metadata),
                sparse_rank=rank,
                sparse_score=hit.score,
                relevance=hit.score,
            )
        else:
            existing.sparse_rank = rank
            existing.sparse_score = hit.score

    fused_scores = reciprocal_rank_fusion(
        {
            "dense": [hit.chunk_id for hit in dense_hits],
            "sparse": [hit.chunk_id for hit in sparse_hits],
        },
        k=k,
    )

    for chunk_id, chunk in candidates.items():
        chunk.fusion_score = fused_scores.get(chunk_id, 0.0)

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            -item.fusion_score,
            item.dense_rank if item.dense_rank is not None else float("inf"),
            item.chunk_id,
        ),
    )

    if limit is not None:
        ordered = ordered[:limit]

    logger.debug(
        "Fused retrieval results",
        extra={
            "dense": len(dense_hits),
            "sparse": len(sparse_hits),
            "fused": len(ordered),
            "rrf_k": k,
        },
    )
    return ordered
