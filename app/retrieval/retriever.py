"""Retrieval orchestration.

Wires the four retrieval stages together for a single query:

1. **Dense recall** - ANN search over the ChromaDB HNSW index.
2. **Lexical recall** - BM25 over the same corpus, catching rare literal tokens.
3. **Fusion** - reciprocal rank fusion of the two ranked lists.
4. **Reranking** - cross-encoder rescoring of the fused candidates.

Every stage is timed and reported to RAGObserve, so a slow query can be
attributed to a specific step rather than the pipeline as a whole.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import Settings, get_settings
from app.ingestion.embedder import BaseEmbedder
from app.observability import tracing
from app.retrieval.bm25_index import BM25Hit, BM25Index
from app.retrieval.hybrid import DEFAULT_RRF_K, RetrievedChunk, fuse_hits
from app.retrieval.reranker import BaseReranker
from app.retrieval.vectorstore import VectorHit, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Everything produced by one retrieval pass.

    Attributes:
        query: The query that was run.
        chunks: Final ranked candidates handed to the generator.
        dense_hits: Raw vector-store results, kept for tracing and evaluation.
        sparse_hits: Raw BM25 results, kept for tracing and evaluation.
        timings: Per-stage elapsed milliseconds.
    """

    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    dense_hits: List[VectorHit] = field(default_factory=list)
    sparse_hits: List[BM25Hit] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        """Total retrieval latency in milliseconds."""
        return round(sum(self.timings.values()), 2)

    @property
    def is_empty(self) -> bool:
        """True when no chunk survived retrieval."""
        return not self.chunks


class Retriever:
    """Hybrid retriever with cross-encoder reranking."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        reranker: BaseReranker,
        settings: Optional[Settings] = None,
    ) -> None:
        """Bind the retriever to its collaborators.

        Args:
            embedder: Produces the query vector.
            vector_store: Dense index.
            bm25_index: Lexical index.
            reranker: Reorders fused candidates.
            settings: Configuration override; defaults to process settings.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._reranker = reranker
        self._settings = settings or get_settings()

    @property
    def reranker_model(self) -> str:
        """Name of the active reranker, for health/debug output."""
        return self._reranker.model_name

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> RetrievalResult:
        """Run the full retrieval pipeline for ``query``.

        Args:
            query: Natural-language question.
            top_k: Number of chunks to return; defaults to ``TOP_K_FINAL``.
            source_filter: Restrict results to a single source document.

        Returns:
            A :class:`RetrievalResult`; empty when nothing is indexed or nothing
            matches.
        """
        final_k = top_k or self._settings.top_k_final
        result = RetrievalResult(query=query)

        if not query.strip():
            return result

        where = {"source": {"$eq": source_filter}} if source_filter else None

        # -- Stage 1: dense recall ----------------------------------------
        with tracing.Stopwatch() as embed_timer:
            query_vector = self._embedder.embed_query(query)
        result.timings["embedding_ms"] = round(embed_timer.elapsed_ms, 2)

        with tracing.Stopwatch() as dense_timer:
            dense_hits = self._vector_store.search(
                query_vector, top_k=self._settings.top_k_dense, where=where
            )
        result.dense_hits = dense_hits
        result.timings["dense_ms"] = round(dense_timer.elapsed_ms, 2)

        # -- Stage 2: lexical recall --------------------------------------
        with tracing.Stopwatch() as sparse_timer:
            sparse_hits = self._bm25.search(query, top_k=self._settings.top_k_sparse)
            if source_filter:
                sparse_hits = [
                    hit for hit in sparse_hits if hit.metadata.get("source") == source_filter
                ]
        result.sparse_hits = sparse_hits
        result.timings["sparse_ms"] = round(sparse_timer.elapsed_ms, 2)

        tracing.log_retrieval(
            query, dense_hits, retriever="chroma-hnsw", top_k=self._settings.top_k_dense,
            duration_ms=result.timings["dense_ms"],
        )
        tracing.log_retrieval(
            query, sparse_hits, retriever="bm25", top_k=self._settings.top_k_sparse,
            duration_ms=result.timings["sparse_ms"],
        )

        # -- Stage 3: fusion ----------------------------------------------
        with tracing.Stopwatch() as fusion_timer:
            fused = fuse_hits(
                dense_hits,
                sparse_hits,
                k=self._settings.rrf_k or DEFAULT_RRF_K,
                limit=self._settings.top_k_fused,
            )
        result.timings["fusion_ms"] = round(fusion_timer.elapsed_ms, 2)
        tracing.log_fusion(
            fused,
            inputs={"dense": dense_hits, "sparse": sparse_hits},
            strategy=f"rrf(k={self._settings.rrf_k})",
        )

        # -- Stage 4: rerank ----------------------------------------------
        with tracing.Stopwatch() as rerank_timer:
            reranked = self._reranker.rerank(query, fused, top_n=final_k)
        result.timings["rerank_ms"] = round(rerank_timer.elapsed_ms, 2)

        if reranked is not fused:
            tracing.log_rerank(
                fused, reranked, model=self._reranker.model_name,
                duration_ms=result.timings["rerank_ms"],
            )

        result.chunks = reranked
        logger.info(
            "Retrieval complete",
            extra={
                "query_chars": len(query),
                "dense": len(dense_hits),
                "sparse": len(sparse_hits),
                "fused": len(fused),
                "returned": len(reranked),
                "total_ms": result.total_ms,
            },
        )
        return result

    def index_stats(self) -> Dict[str, Any]:
        """Summarise the state of both indexes for the health endpoint."""
        return {
            "vectors": self._vector_store.count(),
            "lexical_chunks": self._bm25.count,
            "collection": self._vector_store.collection_name,
            "reranker": self._reranker.model_name,
        }
