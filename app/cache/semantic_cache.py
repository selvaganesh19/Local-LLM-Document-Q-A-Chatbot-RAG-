"""Semantic response cache.

Repeated questions are common in document Q&A ("what is the notice period?" is
asked in many phrasings), and a local LLM answer costs seconds.  This cache
matches on *meaning* rather than string equality: the incoming query is
embedded and compared against cached queries by cosine similarity, so a
paraphrase can reuse a previous answer.

Guards that keep it honest:

* **High threshold.**  Default 0.95 similarity - deliberately conservative,
  because serving a subtly wrong cached answer is worse than recomputing.
* **TTL.**  Entries expire so the cache cannot serve answers forever after the
  underlying documents change.
* **Bounded size with LRU eviction.**  Memory stays flat under load.
* **Thread-safe.**  FastAPI runs sync endpoints in a worker pool, so reads and
  writes are guarded by a lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import Settings, get_settings
from app.ingestion.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """One cached query/response pair.

    Attributes:
        query: The original query text.
        embedding: L2-normalised query vector.
        payload: The cached response body.
        created_at: Unix timestamp of insertion.
        last_access: Unix timestamp of the most recent hit.
        hits: Number of times this entry has been served.
    """

    query: str
    embedding: np.ndarray
    payload: Dict[str, Any]
    created_at: float
    last_access: float
    hits: int = 0

    def is_expired(self, ttl_seconds: int, now: float) -> bool:
        """Whether the entry has outlived ``ttl_seconds``."""
        return ttl_seconds > 0 and (now - self.created_at) > ttl_seconds


@dataclass
class CacheHit:
    """Result of a successful cache lookup.

    Attributes:
        payload: The cached response body.
        similarity: Cosine similarity between the incoming and cached query.
        matched_query: The query text that produced the cached entry.
        age_seconds: How long the entry has been cached.
        original_hits: How many times the entry had been served before now.
    """

    payload: Dict[str, Any]
    similarity: float
    matched_query: str
    age_seconds: float
    original_hits: int


class SemanticCache:
    """Similarity-matched, TTL-bounded, LRU-evicting response cache."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        settings: Optional[Settings] = None,
    ) -> None:
        """Create a cache.

        Args:
            embedder: Model used to vectorise queries.
            settings: Configuration override; defaults to process settings.
        """
        self._embedder = embedder
        self._settings = settings or get_settings()
        self._entries: List[CacheEntry] = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Whether caching is turned on."""
        return self._settings.cache_enabled

    def __len__(self) -> int:
        """Number of live entries, after purging expired ones."""
        self._purge_expired()
        return len(self._entries)

    def stats(self) -> Dict[str, Any]:
        """Return cache counters and configuration for the health endpoint."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": self._settings.cache_enabled,
                "entries": len(self._entries),
                "max_entries": self._settings.cache_max_entries,
                "similarity_threshold": self._settings.cache_similarity_threshold,
                "ttl_seconds": self._settings.cache_ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _purge_expired(self) -> None:
        """Drop entries past their TTL.  Caller must hold the lock."""
        if self._settings.cache_ttl_seconds <= 0:
            return
        now = time.time()
        ttl = self._settings.cache_ttl_seconds
        before = len(self._entries)
        self._entries = [entry for entry in self._entries if not entry.is_expired(ttl, now)]
        dropped = before - len(self._entries)
        if dropped:
            logger.debug("Expired cache entries", extra={"dropped": dropped})

    def _evict_lru(self) -> None:
        """Trim to ``cache_max_entries``, dropping least-recently-used first."""
        limit = self._settings.cache_max_entries
        if limit <= 0 or len(self._entries) <= limit:
            return
        self._entries.sort(key=lambda entry: entry.last_access)
        evicted = len(self._entries) - limit
        self._entries = self._entries[-limit:]
        logger.debug("Evicted cache entries", extra={"evicted": evicted})

    def _similarities(self, vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of ``vector`` against every cached query."""
        if not self._entries:
            return np.zeros((0,), dtype=np.float32)
        matrix = np.vstack([entry.embedding for entry in self._entries])
        return matrix @ vector.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, query: str) -> Optional[CacheHit]:
        """Look up a semantically similar cached response.

        Args:
            query: The incoming user question.

        Returns:
            A :class:`CacheHit` when a live entry matches at or above the
            configured threshold, otherwise ``None``.
        """
        if not self._settings.cache_enabled or not query.strip():
            return None

        try:
            vector = self._embedder.embed_query(query)
        except Exception as exc:  # noqa: BLE001 - a broken cache must not fail requests
            logger.warning("Cache lookup failed to embed query", extra={"error": str(exc)})
            return None

        with self._lock:
            self._purge_expired()
            if not self._entries:
                self._misses += 1
                return None

            scores = self._similarities(vector)
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
            entry = self._entries[best_index]

            if best_score < self._settings.cache_similarity_threshold:
                self._misses += 1
                return None

            now = time.time()
            entry.last_access = now
            original_hits = entry.hits
            entry.hits += 1
            self._hits += 1

            logger.info(
                "Semantic cache hit",
                extra={
                    "similarity": round(best_score, 4),
                    "matched_query": entry.query[:80],
                    "age_seconds": round(now - entry.created_at, 1),
                },
            )
            return CacheHit(
                payload=dict(entry.payload),
                similarity=best_score,
                matched_query=entry.query,
                age_seconds=round(now - entry.created_at, 1),
                original_hits=original_hits,
            )

    def set(self, query: str, payload: Dict[str, Any]) -> None:
        """Store a response for ``query``.

        An existing entry for a near-identical query is replaced rather than
        duplicated, so the cache does not fill with paraphrases of one question.

        Args:
            query: The query the response answers.
            payload: The response body to cache.
        """
        if not self._settings.cache_enabled or not query.strip():
            return

        try:
            vector = self._embedder.embed_query(query)
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            logger.warning("Cache store failed to embed query", extra={"error": str(exc)})
            return

        now = time.time()
        with self._lock:
            self._purge_expired()

            if self._entries:
                scores = self._similarities(vector)
                best_index = int(np.argmax(scores))
                if float(scores[best_index]) >= self._settings.cache_similarity_threshold:
                    self._entries[best_index] = CacheEntry(
                        query=query,
                        embedding=vector,
                        payload=dict(payload),
                        created_at=now,
                        last_access=now,
                    )
                    return

            self._entries.append(
                CacheEntry(
                    query=query,
                    embedding=vector,
                    payload=dict(payload),
                    created_at=now,
                    last_access=now,
                )
            )
            self._evict_lru()

    def clear(self) -> int:
        """Discard every entry.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self._hits = 0
            self._misses = 0
        logger.warning("Semantic cache cleared", extra={"entries": removed})
        return removed

    def invalidate_all(self) -> None:
        """Alias for :meth:`clear`, used after the corpus changes."""
        self.clear()
