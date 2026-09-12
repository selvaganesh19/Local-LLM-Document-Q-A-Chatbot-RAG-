"""Lexical retrieval with BM25.

Deliberately kept separate from the vector store: dense embeddings handle
paraphrase and synonymy well but miss rare literal tokens (identifiers, error
codes, names), which is exactly where BM25 excels.  The two result sets are
fused downstream by reciprocal rank fusion.

The index is persisted to a pickle next to the vector store so a restart does
not require re-tokenising the corpus.

The ``BM25Plus`` variant is used rather than the classic ``BM25``.  Its inverse
document frequency term is ``log((N + 1) / df)``, which stays strictly positive,
whereas classic BM25 assigns an IDF of exactly zero to any term appearing in
half a small corpus - which would silently drop the single most distinctive
query term on a two-document index.  The additive ``delta`` term is set to zero
so that a document containing none of the query terms still scores zero, rather
than being returned as a weak match.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rank_bm25 import BM25Plus

from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

#: Bumped when the persisted payload shape changes, so stale pickles are ignored.
_INDEX_VERSION = 1

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

#: Minimal English stop-word list; keeps the index small without a dependency.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been but by for from had has have he her his i if in
    into is it its of on or our she that the their them then there these they
    this to was were what when where which who will with would you your
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lower-case tokenise ``text`` and drop stop-words and singletons.

    Args:
        text: Raw text.

    Returns:
        The surviving tokens, in order.
    """
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


@dataclass
class BM25Hit:
    """A single lexical-retrieval result.

    Attributes:
        chunk_id: Identifier of the matched chunk.
        text: Chunk text.
        metadata: Stored provenance.
        score: Normalised score in ``[0, 1]``; ``1.0`` marks the top result.
        raw_score: Un-normalised BM25 score, kept for diagnostics.
    """

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    raw_score: float = 0.0

    @property
    def source(self) -> str:
        """Source document name recorded in metadata."""
        return str(self.metadata.get("source", "unknown"))


class BM25Index:
    """In-memory BM25 index with optional disk persistence."""

    def __init__(self) -> None:
        """Create an empty index."""
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._tokenized: List[List[str]] = []
        self._bm25: Optional[BM25Plus] = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of indexed chunks."""
        return len(self._ids)

    @property
    def count(self) -> int:
        """Number of indexed chunks."""
        return len(self._ids)

    def list_sources(self) -> List[Dict[str, Any]]:
        """Aggregate indexed chunks by source document."""
        summaries: Dict[str, Dict[str, Any]] = {}
        for metadata in self._metadatas:
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
    def _rebuild(self) -> None:
        """Recompute the BM25 model from the current corpus."""
        if not self._tokenized:
            self._bm25 = None
            return
        # An all-empty corpus would divide by zero inside the model.
        if not any(self._tokenized):
            self._bm25 = None
            return
        # delta=0 removes BM25Plus's additive floor.  With the default delta of
        # 1 every document scores above zero for every query term, so an
        # unrelated document would still be returned as a "match" and the
        # sparse leg of the fusion would never come back empty.
        self._bm25 = BM25Plus(self._tokenized, delta=0.0)

    def add_chunks(self, chunks: Sequence[Chunk], rebuild: bool = True) -> int:
        """Insert or replace chunks, keyed by ``chunk_id``.

        Args:
            chunks: Chunks to index.
            rebuild: Recompute the BM25 model immediately.  Pass ``False`` to
                batch several calls followed by one :meth:`rebuild_index`.

        Returns:
            Number of newly added chunks (replacements are not counted).
        """
        if not chunks:
            return 0

        positions = {chunk_id: position for position, chunk_id in enumerate(self._ids)}
        added = 0
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            existing = positions.get(chunk.chunk_id)
            if existing is None:
                positions[chunk.chunk_id] = len(self._ids)
                self._ids.append(chunk.chunk_id)
                self._texts.append(chunk.text)
                self._metadatas.append(chunk.to_record())
                self._tokenized.append(tokens)
                added += 1
            else:
                self._texts[existing] = chunk.text
                self._metadatas[existing] = chunk.to_record()
                self._tokenized[existing] = tokens

        if rebuild:
            self._rebuild()
        return added

    def rebuild_index(self) -> None:
        """Recompute the BM25 model after a batched set of additions."""
        self._rebuild()

    def delete_by_source(self, source: str) -> int:
        """Remove every chunk belonging to ``source``.

        Args:
            source: Exact source name recorded at ingestion time.

        Returns:
            Number of chunks removed.
        """
        keep = [
            position
            for position, metadata in enumerate(self._metadatas)
            if str(metadata.get("source", "unknown")) != source
        ]
        removed = len(self._ids) - len(keep)
        if removed:
            self._ids = [self._ids[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
            self._metadatas = [self._metadatas[i] for i in keep]
            self._tokenized = [self._tokenized[i] for i in keep]
            self._rebuild()
            logger.info("Removed source from BM25 index", extra={"source": source, "chunks": removed})
        return removed

    def reset(self) -> None:
        """Discard the entire index."""
        self._ids.clear()
        self._texts.clear()
        self._metadatas.clear()
        self._tokenized.clear()
        self._bm25 = None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> List[BM25Hit]:
        """Score the corpus against ``query`` and return the best ``top_k``.

        Args:
            query: Natural-language query.
            top_k: Maximum number of hits.

        Returns:
            Hits ordered by descending BM25 score; empty when nothing matches.
        """
        if top_k <= 0 or self._bm25 is None or not self._ids:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # Guard against any numerical noise so normalisation stays well-defined.
        scores = [max(0.0, float(score)) for score in scores]

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked = [i for i in ranked[:top_k] if scores[i] > 0]
        if not ranked:
            return []

        best = max(scores[i] for i in ranked) or 1.0
        return [
            BM25Hit(
                chunk_id=self._ids[i],
                text=self._texts[i],
                metadata=dict(self._metadatas[i]),
                score=scores[i] / best,
                raw_score=scores[i],
            )
            for i in ranked
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        """Persist the index to ``path`` as a pickle.

        Args:
            path: Destination file; parent directories are created as needed.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _INDEX_VERSION,
            "ids": self._ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
            "tokenized": self._tokenized,
        }
        with destination.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Persisted BM25 index", extra={"path": str(destination), "chunks": len(self._ids)})

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        """Load an index from ``path``.

        A missing, corrupt, or version-mismatched file yields an empty index
        rather than an exception, so the application still starts; re-ingesting
        the corpus repopulates it.

        Args:
            path: Pickle written by :meth:`save`.

        Returns:
            The restored index.
        """
        index = cls()
        source = Path(path)
        if not source.exists():
            return index

        try:
            with source.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001 - corrupt cache must not be fatal
            logger.warning("Ignoring unreadable BM25 index", extra={"path": str(source), "error": str(exc)})
            return index

        if not isinstance(payload, dict) or payload.get("version") != _INDEX_VERSION:
            logger.warning("Ignoring BM25 index with incompatible format", extra={"path": str(source)})
            return index

        index._ids = list(payload.get("ids", []))
        index._texts = list(payload.get("texts", []))
        index._metadatas = [dict(item) for item in payload.get("metadatas", [])]
        index._tokenized = [list(item) for item in payload.get("tokenized", [])]
        index._rebuild()
        logger.info("Loaded BM25 index", extra={"path": str(source), "chunks": index.count})
        return index

    def iter_records(self) -> Iterable[tuple[str, str, Dict[str, Any]]]:
        """Yield ``(chunk_id, text, metadata)`` for every indexed chunk."""
        for chunk_id, text, metadata in zip(self._ids, self._texts, self._metadatas):
            yield chunk_id, text, dict(metadata)
