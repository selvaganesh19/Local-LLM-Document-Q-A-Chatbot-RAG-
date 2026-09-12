"""Shared pytest fixtures.

The suite is deliberately hermetic: no test downloads a model or contacts an
Ollama server.  Swapping the embedding model for :class:`HashingEmbedder`, the
reranker for :class:`NoOpReranker` and the LLM for :class:`FakeOllamaClient`
exercises the real pipeline - chunking, indexing, fusion, ranking, citation
parsing, caching, HTTP - while staying fast and offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import ServiceContainer
from app.generation.ollama_client import FakeOllamaClient
from app.ingestion.chunker import Chunk, chunk_text
from app.ingestion.embedder import HashingEmbedder
from app.ingestion.loaders import Document
from app.main import create_app
from app.middleware.rate_limit import limiter
from app.retrieval.reranker import NoOpReranker

#: Fixed embedding width for the test embedder; small keeps the maths obvious.
TEST_EMBEDDING_DIM = 64

#: Corpus shared by the retrieval and API tests.
SAMPLE_DOCUMENTS = [
    Document(
        text=(
            "The notice period for senior staff is 90 calendar days. "
            "Standard employees must give 30 calendar days of notice. "
            "Notice must be given in writing to the line manager."
        ),
        source="handbook.txt",
        metadata={"type": "txt", "page": 1},
    ),
    Document(
        text=(
            "Passwords must be at least 14 characters and rotated every 180 days. "
            "Multi-factor authentication is mandatory for all remote access. "
            "Security incidents must be reported within 1 hour."
        ),
        source="security.txt",
        metadata={"type": "txt", "page": 2},
    ),
    Document(
        text=(
            "The API rate limit is 1000 requests per hour and 50 requests per minute. "
            "Exceeding the limit returns HTTP 429 with a Retry-After header. "
            "Access tokens expire after 3600 seconds."
        ),
        source="api.txt",
        metadata={"type": "txt", "page": 3},
    ),
]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway directory, with fakes-friendly values."""
    return Settings(
        _env_file=None,
        ollama_base_url="http://test-ollama:11434",
        ollama_model="test-model",
        chroma_dir=str(tmp_path / "chroma"),
        chroma_collection="test_documents",
        embedding_model="test-embedder",
        embedding_dim=TEST_EMBEDDING_DIM,
        reranker_model="test-reranker",
        chunk_size=220,
        chunk_overlap=40,
        top_k_dense=10,
        top_k_sparse=10,
        top_k_fused=10,
        top_k_final=3,
        rerank_enabled=False,
        cache_enabled=True,
        cache_similarity_threshold=0.9,
        cache_ttl_seconds=3600,
        cache_max_entries=8,
        rate_limit_enabled=False,
        ragobserve_enabled=False,
        dev_fake_llm=False,
    )


@pytest.fixture
def unconfigured_settings(settings: Settings) -> Settings:
    """Settings with the mandatory Ollama values left blank."""
    return settings.model_copy(update={"ollama_base_url": "", "ollama_model": ""})


@pytest.fixture
def embedder() -> HashingEmbedder:
    """Deterministic, dependency-free embedder."""
    return HashingEmbedder(dimension=TEST_EMBEDDING_DIM)


@pytest.fixture
def fake_client() -> FakeOllamaClient:
    """Offline LLM stand-in with a call log."""
    return FakeOllamaClient()


@pytest.fixture
def container(
    settings: Settings,
    embedder: HashingEmbedder,
    fake_client: FakeOllamaClient,
) -> ServiceContainer:
    """Container wired entirely to test doubles."""
    return ServiceContainer(
        settings=settings,
        embedder=embedder,
        reranker=NoOpReranker(),
        client=fake_client,
    )


@pytest.fixture
def ingested_container(container: ServiceContainer) -> ServiceContainer:
    """Container with the sample corpus already indexed."""
    container.pipeline.ingest_documents(SAMPLE_DOCUMENTS)
    return container


@pytest.fixture
def client(container: ServiceContainer) -> Iterator[TestClient]:
    """HTTP client bound to a test-configured application."""
    application = create_app(settings=container.settings, container=container)
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
def ingested_client(ingested_container: ServiceContainer) -> Iterator[TestClient]:
    """HTTP client whose application already has the sample corpus indexed."""
    application = create_app(settings=ingested_container.settings, container=ingested_container)
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> Iterator[None]:
    """Keep the process-wide limiter disabled between tests.

    ``slowapi``'s limiter is a module-level singleton, so a test that enables
    limiting would otherwise leak into every test that follows.
    """
    limiter.enabled = False
    yield
    limiter.enabled = False
    limiter.reset()


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """A handful of chunks spanning the sample documents."""
    chunks: List[Chunk] = []
    for document in SAMPLE_DOCUMENTS:
        chunks.extend(
            chunk_text(
                document.text,
                source=document.source,
                metadata=document.metadata,
                chunk_size=220,
                chunk_overlap=40,
            )
        )
    return chunks
