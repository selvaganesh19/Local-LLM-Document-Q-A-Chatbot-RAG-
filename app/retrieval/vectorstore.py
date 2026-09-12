"""ChromaDB-backed vector store with an explicitly configured HNSW index.

The collection is created with cosine space and tunable HNSW parameters
(``M``, ``construction_ef``, ``search_ef``) so recall/latency can be traded off
without re-embedding anything.

The embedding function is intentionally disabled: vectors are always supplied
by :mod:`app.ingestion.embedder`, which keeps Chroma from pulling down its own
default ONNX model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import Settings, get_settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

#: Maximum number of records pushed to Chroma in a single call.
_WRITE_BATCH_SIZE = 500

_METADATA_PRIMITIVES = (str, int, float, bool)


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a metadata mapping into Chroma's supported value types.

    ``None`` values are dropped and everything else is stringified unless it is
    already a primitive, since Chroma rejects nested objects.
    """
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, _METADATA_PRIMITIVES) else str(value)
    return clean


@dataclass
class VectorHit:
    """A single dense-retrieval result.

    Attributes:
        chunk_id: Identifier of the matched chunk.
        text: Chunk text.
        metadata: Stored provenance.
        score: Cosine similarity in ``[-1, 1]`` (higher is better).
        source: Source document name, hoisted out of ``metadata`` for convenience.
    """

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def source(self) -> str:
        """Source document name recorded in metadata."""
        return str(self.metadata.get("source", "unknown"))


class VectorStore:
    """Persistent ChromaDB collection tuned for semantic search."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Open (or create) the persistent collection.

        Args:
            settings: Configuration override; defaults to the process settings.
        """
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._settings = settings or get_settings()
        self._path = self._settings.chroma_path
        self._path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": self._settings.hnsw_m,
                "hnsw:construction_ef": self._settings.hnsw_construction_ef,
                "hnsw:search_ef": self._settings.hnsw_search_ef,
            },
        )
        logger.info(
            "Vector store ready",
            extra={
                "path": str(self._path),
                "collection": self._settings.chroma_collection,
                "hnsw_m": self._settings.hnsw_m,
                "count": self.count(),
            },
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def collection_name(self) -> str:
        """Name of the underlying Chroma collection."""
        return self._settings.chroma_collection

    def count(self) -> int:
        """Number of stored chunks."""
        return int(self._collection.count())

    def list_sources(self) -> List[Dict[str, Any]]:
        """Aggregate stored chunks by source document.

        Returns:
            One entry per distinct source with its chunk count and file type,
            sorted by source name.
        """
        if self.count() == 0:
            return []

        records = self._collection.get(include=["metadatas"])
        summaries: Dict[str, Dict[str, Any]] = {}
        for metadata in records.get("metadatas") or []:
            if not metadata:
                continue
            source = str(metadata.get("source", "unknown"))
            entry = summaries.setdefault(
                source,
                {"source": source, "chunks": 0, "type": metadata.get("type", "unknown")},
            )
            entry["chunks"] += 1

        return sorted(summaries.values(), key=lambda item: item["source"])

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        """Insert or replace chunks together with their pre-computed vectors.

        Args:
            chunks: Chunks to store.
            embeddings: One vector per chunk, in the same order.

        Returns:
            Number of records written.

        Raises:
            ValueError: If the two sequences differ in length.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must align"
            )
        if not chunks:
            return 0

        written = 0
        for start in range(0, len(chunks), _WRITE_BATCH_SIZE):
            batch_chunks = chunks[start : start + _WRITE_BATCH_SIZE]
            batch_vectors = embeddings[start : start + _WRITE_BATCH_SIZE]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                embeddings=[list(map(float, vector)) for vector in batch_vectors],
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[sanitize_metadata(chunk.to_record()) for chunk in batch_chunks],
            )
            written += len(batch_chunks)

        logger.info("Stored chunks in vector store", extra={"chunks": written})
        return written

    def delete_by_source(self, source: str) -> int:
        """Delete every chunk belonging to ``source``.

        Args:
            source: Exact source name recorded at ingestion time.

        Returns:
            Number of chunks removed.
        """
        matching = self._collection.get(where={"source": {"$eq": source}}, include=[])
        ids = matching.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted source", extra={"source": source, "chunks": len(ids)})
        return len(ids)

    def reset(self) -> None:
        """Drop and recreate the collection, discarding all vectors."""
        self._client.delete_collection(self._settings.chroma_collection)
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": self._settings.hnsw_m,
                "hnsw:construction_ef": self._settings.hnsw_construction_ef,
                "hnsw:search_ef": self._settings.hnsw_search_ef,
            },
        )
        logger.warning("Vector store reset", extra={"collection": self.collection_name})

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        """Run an approximate nearest-neighbour search over the HNSW index.

        Args:
            query_embedding: Query vector, same dimensionality as the index.
            top_k: Number of neighbours to return.
            where: Optional Chroma metadata filter.

        Returns:
            Matching hits ordered by descending cosine similarity.
        """
        if top_k <= 0 or self.count() == 0:
            return []

        response = self._collection.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=min(top_k, self.count()),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        ids = (response.get("ids") or [[]])[0]

        hits: List[VectorHit] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = 1.0 - float(distance)
            hits.append(
                VectorHit(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    metadata=dict(metadata or {}),
                    score=max(-1.0, min(1.0, similarity)),
                )
            )
        return hits

    def get_chunk(self, chunk_id: str) -> Optional[VectorHit]:
        """Fetch a single chunk by id, or ``None`` when it is absent."""
        response = self._collection.get(
            ids=[chunk_id], include=["documents", "metadatas"]
        )
        ids = response.get("ids") or []
        if not ids:
            return None
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        return VectorHit(
            chunk_id=str(ids[0]),
            text=str(documents[0]) if documents else "",
            metadata=dict(metadatas[0]) if metadatas else {},
            score=1.0,
        )
