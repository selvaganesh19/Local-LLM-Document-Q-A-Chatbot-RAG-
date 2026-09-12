"""Pydantic request and response models for the HTTP API.

Response models use ``extra="allow"`` so that adding a field to an internal
payload never causes a 500 at serialisation time - the extra key simply passes
through.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _LooseModel(BaseModel):
    """Base model that tolerates and forwards unexpected fields."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
class ChatRequest(_LooseModel):
    """A question to answer from the indexed corpus."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question.",
        examples=["What is the notice period for termination?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of chunks to retrieve before generation.",
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to a single source document name.",
    )
    use_cache: bool = Field(
        default=True,
        description="Allow serving a semantically similar cached answer.",
    )


class CitationModel(_LooseModel):
    """A passage the generated answer explicitly cited."""

    index: int = Field(description="Citation number as it appears in the answer.")
    chunk_id: str
    source: str
    page: Optional[int] = None
    label: str = Field(description="Human-readable provenance, e.g. 'report.pdf, page 3'.")
    snippet: str
    relevance: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedModel(_LooseModel):
    """A passage that was retrieved, whether or not it was cited."""

    index: int
    chunk_id: str
    source: str
    page: Optional[int] = None
    label: str
    snippet: str
    relevance: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fusion_score: float = 0.0
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(_LooseModel):
    """A grounded answer with its sources and diagnostics."""

    question: str
    answer: str
    citations: List[CitationModel] = Field(default_factory=list)
    retrieved: List[RetrievedModel] = Field(default_factory=list)
    insufficient_context: bool = False
    uncited: bool = False
    grounded: bool = True
    cached: bool = False
    cache_similarity: Optional[float] = None
    matched_query: Optional[str] = None
    model: str = ""
    trace_id: Optional[str] = None
    timings: Dict[str, float] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class IngestTextRequest(_LooseModel):
    """Ingest a block of text pasted by the user."""

    text: str = Field(..., min_length=1, max_length=2_000_000)
    source_name: str = Field(
        default="pasted-text.txt",
        max_length=255,
        description="Label used for citations of this content.",
    )


class IngestPathRequest(_LooseModel):
    """Ingest files from a path on the server's filesystem."""

    path: str = Field(..., min_length=1, description="File or directory path.")
    recursive: bool = Field(default=True, description="Descend into sub-directories.")


class IngestResponse(_LooseModel):
    """Outcome of an ingestion run."""

    documents: int = 0
    chunks: int = 0
    new_chunks: int = 0
    sources: List[str] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)
    duration_ms: float = 0.0
    embedder: str = ""
    dimensions: int = 0


class DocumentInfo(_LooseModel):
    """One indexed source document."""

    source: str
    chunks: int
    type: str = "unknown"


class DocumentList(_LooseModel):
    """All indexed source documents."""

    documents: List[DocumentInfo] = Field(default_factory=list)
    total_chunks: int = 0


class DeleteResponse(_LooseModel):
    """Result of deleting a source document."""

    source: str
    vectors: int = 0
    lexical: int = 0


# ---------------------------------------------------------------------------
# Health and configuration
# ---------------------------------------------------------------------------
class OllamaStatus(_LooseModel):
    """Reachability and model availability of the local LLM server."""

    reachable: bool = False
    configured: bool = False
    base_url: str = "<unset>"
    model: str = "<unset>"
    flavor: str = "ollama"
    model_available: bool = False
    models: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class HealthResponse(_LooseModel):
    """Liveness and readiness report."""

    status: str = Field(description="'ok', 'degraded', or 'unconfigured'.")
    version: str
    ollama: OllamaStatus
    configuration_errors: List[str] = Field(default_factory=list)
    index: Dict[str, Any] = Field(default_factory=dict)


class ConfigResponse(_LooseModel):
    """Non-secret configuration surfaced to the web UI."""

    model: str
    llm_configured: bool
    embedding_model: str
    reranker_model: str
    top_k_final: int
    rerank_enabled: bool
    cache_enabled: bool
    rate_limit_enabled: bool
    ragobserve_enabled: bool
    dev_fake_llm: bool


class StatsResponse(_LooseModel):
    """Index and cache statistics."""

    vectors: int = 0
    lexical_chunks: int = 0
    collection: str = ""
    reranker: str = ""
    cache: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(_LooseModel):
    """Structured error body."""

    detail: str
    request_id: Optional[str] = None
    hint: Optional[str] = None
