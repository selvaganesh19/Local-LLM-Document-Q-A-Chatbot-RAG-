"""Ingestion pipeline.

Orchestrates the full document intake path:

    load -> chunk -> embed -> index (dense + lexical) -> persist

Both indexes are kept in lockstep so hybrid retrieval always sees the same
corpus.  Chunk ids are content-addressed, which makes re-ingesting the same
file an idempotent upsert rather than a source of duplicates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from app.config import Settings, get_settings
from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.embedder import BaseEmbedder
from app.ingestion.loaders import (
    Document,
    DocumentLoadError,
    load_directory,
    load_document,
    load_from_text,
)
from app.observability import tracing
from app.retrieval.bm25_index import BM25Index
from app.retrieval.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Outcome of one ingestion run.

    Attributes:
        documents: Number of source documents processed.
        chunks: Total chunks produced.
        new_chunks: Chunks not previously present in the lexical index.
        sources: Distinct source names touched.
        failures: Human-readable messages for files that could not be read.
        duration_ms: Wall-clock duration of the run.
        embedder: Name of the embedding model used.
        dimensions: Vector dimensionality.
    """

    documents: int = 0
    chunks: int = 0
    new_chunks: int = 0
    sources: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    embedder: str = ""
    dimensions: int = 0

    def to_dict(self) -> dict:
        """Serialise for API responses."""
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "new_chunks": self.new_chunks,
            "sources": self.sources,
            "failures": self.failures,
            "duration_ms": round(self.duration_ms, 2),
            "embedder": self.embedder,
            "dimensions": self.dimensions,
        }


class IngestionPipeline:
    """Loads, chunks, embeds and indexes documents."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        settings: Optional[Settings] = None,
    ) -> None:
        """Bind the pipeline to its collaborators.

        Args:
            embedder: Produces chunk vectors.
            vector_store: Dense index receiving the vectors.
            bm25_index: Lexical index receiving the chunk text.
            settings: Configuration override; defaults to process settings.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def ingest_documents(self, documents: Sequence[Document], persist: bool = True) -> IngestionResult:
        """Chunk, embed and index pre-loaded documents.

        Args:
            documents: Documents to ingest.
            persist: Write the BM25 index to disk when finished.

        Returns:
            An :class:`IngestionResult` describing what happened.
        """
        if not documents:
            return IngestionResult(embedder=self._embedder.model_name)

        with tracing.Stopwatch() as total_timer:
            chunks: List[Chunk] = chunk_documents(
                documents,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
            )
            tracing.log_chunks(
                chunks,
                strategy="sentence-aware-overlap",
                chunk_size=self._settings.chunk_size,
                overlap=self._settings.chunk_overlap,
            )

            if not chunks:
                logger.warning("No chunks produced", extra={"documents": len(documents)})
                return IngestionResult(
                    documents=len(documents),
                    sources=sorted({doc.source for doc in documents}),
                    embedder=self._embedder.model_name,
                )

            with tracing.Stopwatch() as embed_timer:
                vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
            tracing.log_embedding(
                model=self._embedder.model_name,
                input_count=len(chunks),
                dimensions=self._embedder.dimension,
                duration_ms=embed_timer.elapsed_ms,
            )

            self._vector_store.upsert_chunks(chunks, vectors)
            new_chunks = self._bm25.add_chunks(chunks, rebuild=False)
            self._bm25.rebuild_index()

            if persist:
                self._bm25.save(self._settings.bm25_index_path)

        sources = sorted({chunk.source for chunk in chunks})
        result = IngestionResult(
            documents=len(documents),
            chunks=len(chunks),
            new_chunks=new_chunks,
            sources=sources,
            duration_ms=total_timer.elapsed_ms,
            embedder=self._embedder.model_name,
            dimensions=self._embedder.dimension,
        )

        tracing.log_ingestion(
            source=", ".join(sources) or "unknown",
            count=len(chunks),
            duration_ms=result.duration_ms,
        )
        logger.info("Ingestion complete", extra=result.to_dict())
        return result

    def ingest_text(self, text: str, source_name: str) -> IngestionResult:
        """Ingest a raw string, e.g. text pasted into the UI."""
        return self.ingest_documents(load_from_text(text, source_name))

    def ingest_files(self, paths: Sequence[str | Path]) -> IngestionResult:
        """Ingest a list of files, collecting per-file failures.

        Args:
            paths: Files to ingest.

        Returns:
            A merged result; unreadable files appear in ``failures`` rather
            than aborting the run.
        """
        documents: List[Document] = []
        failures: List[str] = []
        for path in paths:
            try:
                documents.extend(load_document(path))
            except DocumentLoadError as exc:
                logger.warning("Skipping document", extra={"error": str(exc)})
                failures.append(str(exc))

        result = self.ingest_documents(documents)
        result.failures.extend(failures)
        return result

    def ingest_directory(self, directory: str | Path, recursive: bool = True) -> IngestionResult:
        """Ingest every supported file under ``directory``."""
        documents, failures = load_directory(directory, recursive=recursive)
        result = self.ingest_documents(documents)
        result.failures.extend(failures)
        return result

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def delete_source(self, source: str) -> dict:
        """Remove a source document from both indexes.

        Args:
            source: Exact source name recorded at ingestion time.

        Returns:
            Counts of what was removed.
        """
        removed_vectors = self._vector_store.delete_by_source(source)
        removed_lexical = self._bm25.delete_by_source(source)
        self._bm25.save(self._settings.bm25_index_path)
        logger.info(
            "Deleted source", extra={"source": source, "vectors": removed_vectors, "lexical": removed_lexical}
        )
        return {"source": source, "vectors": removed_vectors, "lexical": removed_lexical}

    def reset(self) -> None:
        """Drop every indexed chunk from both indexes."""
        self._vector_store.reset()
        self._bm25.reset()
        self._bm25.save(self._settings.bm25_index_path)
        logger.warning("Ingestion indexes reset")

    def index_stats(self) -> dict:
        """Summarise the size of both indexes."""
        return {
            "vectors": self._vector_store.count(),
            "lexical_chunks": self._bm25.count,
            "sources": len(self._vector_store.list_sources()),
        }
