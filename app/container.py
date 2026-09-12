"""Dependency container.

Every collaborator is constructed lazily and cached on the container, so the
application pays for the embedding model, the cross-encoder and the Chroma
client only when a request actually needs them.  Construction is also
overridable, which is what lets the test suite swap in a deterministic embedder
and an offline LLM client without monkeypatching module globals.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request

from app.cache.semantic_cache import SemanticCache
from app.config import Settings, get_settings
from app.generation.generator import AnswerGenerator
from app.generation.ollama_client import FakeOllamaClient, OllamaClient
from app.ingestion.embedder import BaseEmbedder, get_embedder
from app.ingestion.pipeline import IngestionPipeline
from app.observability import tracing
from app.retrieval.bm25_index import BM25Index
from app.retrieval.reranker import BaseReranker, get_reranker
from app.retrieval.retriever import Retriever
from app.retrieval.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Holds the application's long-lived services."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[BaseReranker] = None,
        client: Optional[OllamaClient] = None,
        cache: Optional[SemanticCache] = None,
    ) -> None:
        """Create a container.

        Args:
            settings: Configuration override; defaults to process settings.
            embedder: Pre-built embedder, bypassing the model cache.
            vector_store: Pre-built vector store.
            bm25_index: Pre-built lexical index.
            reranker: Pre-built reranker.
            client: Pre-built LLM client.
            cache: Pre-built semantic cache.
        """
        self.settings = settings or get_settings()
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._reranker = reranker
        self._client = client
        self._cache = cache
        self._retriever: Optional[Retriever] = None
        self._pipeline: Optional[IngestionPipeline] = None
        self._generator: Optional[AnswerGenerator] = None
        self._started = False

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------
    @property
    def embedder(self) -> BaseEmbedder:
        """Shared embedding model."""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        """Persistent ChromaDB collection."""
        if self._vector_store is None:
            self._vector_store = VectorStore(settings=self.settings)
        return self._vector_store

    @property
    def bm25_index(self) -> BM25Index:
        """Lexical index; loaded from disk on startup when available."""
        if self._bm25_index is None:
            self._bm25_index = BM25Index()
        return self._bm25_index

    @property
    def reranker(self) -> BaseReranker:
        """Candidate reranker."""
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    @property
    def client(self) -> OllamaClient:
        """LLM client, real or fake depending on configuration."""
        if self._client is None:
            if self.settings.dev_fake_llm:
                logger.warning("DEV_FAKE_LLM enabled - using FakeOllamaClient")
                self._client = FakeOllamaClient()
            else:
                self._client = OllamaClient(settings=self.settings)
        return self._client

    @property
    def retriever(self) -> Retriever:
        """Hybrid retriever wired to the shared indexes."""
        if self._retriever is None:
            self._retriever = Retriever(
                embedder=self.embedder,
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                reranker=self.reranker,
                settings=self.settings,
            )
        return self._retriever

    @property
    def pipeline(self) -> IngestionPipeline:
        """Ingestion pipeline wired to the same indexes."""
        if self._pipeline is None:
            self._pipeline = IngestionPipeline(
                embedder=self.embedder,
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                settings=self.settings,
            )
        return self._pipeline

    @property
    def generator(self) -> AnswerGenerator:
        """Grounded answer generator."""
        if self._generator is None:
            self._generator = AnswerGenerator(client=self.client, settings=self.settings)
        return self._generator

    @property
    def cache(self) -> SemanticCache:
        """Semantic response cache."""
        if self._cache is None:
            self._cache = SemanticCache(embedder=self.embedder, settings=self.settings)
        return self._cache

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def startup(self) -> None:
        """Prepare long-lived state: tracing and the persisted BM25 index."""
        if self._started:
            return

        tracing.init_tracing(self.settings)

        if self._bm25_index is None:
            # Restore the lexical index from disk; a missing or corrupt pickle
            # yields an empty index and re-ingestion repopulates it.
            restored = BM25Index.load(self.settings.bm25_index_path)
            self._bm25_index = restored
            if restored.count:
                logger.info("Restored BM25 index from disk", extra={"chunks": restored.count})

        self._started = True

    async def shutdown(self) -> None:
        """Release resources held by the container."""
        tracing.flush()
        if self._client is not None:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def is_llm_configured(self) -> bool:
        """Whether generation can be attempted with the current configuration."""
        return self.client.is_configured

    def configuration_errors(self) -> list[str]:
        """Human-readable list of missing mandatory settings."""
        problems = self.settings.llm_configuration_errors()
        if self.settings.dev_fake_llm:
            return []
        return problems


def get_container(request: Request) -> ServiceContainer:
    """FastAPI dependency returning the container attached to the app.

    Args:
        request: The incoming request.

    Returns:
        The process-wide :class:`ServiceContainer`.
    """
    return request.app.state.container
