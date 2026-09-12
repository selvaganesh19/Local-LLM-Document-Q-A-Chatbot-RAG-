"""Application configuration.

All tunables are read from environment variables (optionally via a ``.env``
file) using ``pydantic-settings``.  The two mandatory values - the Ollama base
URL and the Ollama model tag - intentionally ship blank so that they are never
hard-coded into the repository; supply them in your own ``.env``.

Import the process-wide singleton via :func:`get_settings`.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Repository root (the directory containing the ``app`` package).
BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration for the RAG application."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Ollama (mandatory, intentionally blank) ---------------------------
    ollama_base_url: str = Field(
        default="",
        description="Base URL of the local Ollama server, e.g. http://localhost:11434",
    )
    ollama_model: str = Field(
        default="",
        description="Ollama model tag used for generation, e.g. qwen3:8b",
    )
    ollama_timeout_seconds: float = 120.0
    ollama_temperature: float = 0.1
    ollama_num_ctx: int = 8192

    # Which wire protocol the endpoint above speaks:
    #   "auto"   - infer from the URL (a path ending in /v1 means OpenAI).
    #   "ollama" - Ollama's native REST API (POST /api/chat).
    #   "openai" - any OpenAI-compatible server (POST /v1/chat/completions),
    #              e.g. vLLM, llama.cpp's server, LM Studio, Ollama's own
    #              compatibility shim, or a remote GPU deployment.
    llm_api_flavor: Literal["auto", "ollama", "openai"] = "auto"
    llm_api_key: str = Field(
        default="",
        description="Bearer token for the OpenAI-compatible flavor; usually blank for a local server",
    )
    llm_disable_thinking: bool = Field(
        default=True,
        description=(
            "Ask a reasoning model to skip its chain of thought. Sent as "
            "chat_template_kwargs on the OpenAI flavor, and dropped automatically "
            "for servers that reject the field. Set false to keep reasoning enabled."
        ),
    )

    # -- Embedding / reranking --------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_device: str = "cpu"

    # -- Chunking ----------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # -- Retrieval ---------------------------------------------------------
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_fused: int = 20
    top_k_final: int = 5
    rrf_k: int = 60
    rerank_enabled: bool = True

    # -- Vector store (ChromaDB) ------------------------------------------
    chroma_dir: str = "./data/vectorstore"
    chroma_collection: str = "rag_documents"
    hnsw_m: int = 32
    hnsw_construction_ef: int = 200
    hnsw_search_ef: int = 100

    # -- Semantic cache ----------------------------------------------------
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 512

    # -- Rate limiting -----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_chat: str = "20/minute"
    rate_limit_ingest: str = "10/minute"
    rate_limit_default: str = "120/minute"

    # -- Observability -----------------------------------------------------
    ragobserve_enabled: bool = True
    ragobserve_project: str = "local-llm-rag"
    ragobserve_db_path: str = "./.ragobserve/ragobserve.db"

    # -- Application -------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False
    dev_fake_llm: bool = False
    max_upload_mb: int = 25
    allow_ingest_any_path: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Accept ``"a,b"`` or ``"*"`` in addition to a real list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("embedding_device")
    @classmethod
    def _normalise_device(cls, value: str) -> str:
        return value.strip().lower() or "cpu"

    @field_validator("chroma_collection")
    @classmethod
    def _validate_collection_name(cls, value: str) -> str:
        """ChromaDB requires 3-63 chars, alphanumeric with ``_``/``-``/``.``."""
        name = value.strip()
        if not 3 <= len(name) <= 63:
            raise ValueError("CHROMA_COLLECTION must be 3-63 characters long")
        if not name[0].isalnum() or not name[-1].isalnum():
            raise ValueError(
                "CHROMA_COLLECTION must start and end with an alphanumeric character"
            )
        return name

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    def _resolve(self, raw: str) -> Path:
        """Resolve a possibly-relative path against the repository root."""
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    @property
    def chroma_path(self) -> Path:
        """Absolute path of the ChromaDB persistence directory."""
        return self._resolve(self.chroma_dir)

    @property
    def bm25_index_path(self) -> Path:
        """Absolute path of the persisted BM25 index pickle."""
        return self.chroma_path / "bm25_index.pkl"

    @property
    def ragobserve_db(self) -> Path:
        """Absolute path of the RAGObserve SQLite trace database."""
        return self._resolve(self.ragobserve_db_path)

    @property
    def documents_dir(self) -> Path:
        """Absolute path of the default document drop folder."""
        return BASE_DIR / "data" / "documents"

    @property
    def static_dir(self) -> Path:
        """Absolute path of the static web UI directory."""
        return BASE_DIR / "static"

    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024

    # ------------------------------------------------------------------
    # LLM configuration helpers
    # ------------------------------------------------------------------
    @property
    def resolved_api_flavor(self) -> Literal["ollama", "openai"]:
        """The concrete wire protocol to use, resolving ``"auto"``.

        An explicit ``LLM_API_FLAVOR`` always wins.  Otherwise the URL decides:
        OpenAI-compatible servers conventionally expose their API under a
        ``/v1`` prefix, which Ollama's native routes never use.
        """
        if self.llm_api_flavor != "auto":
            return self.llm_api_flavor
        return "openai" if self.ollama_base_url.rstrip("/").endswith("/v1") else "ollama"

    @property
    def llm_configured(self) -> bool:
        """True when both mandatory LLM settings are present."""
        return bool(self.ollama_base_url.strip() and self.ollama_model.strip())

    def llm_configuration_errors(self) -> List[str]:
        """Return a human-readable list of missing mandatory settings."""
        problems: List[str] = []
        if not self.ollama_base_url.strip():
            problems.append(
                "OLLAMA_BASE_URL is empty - set it to your LLM server, "
                "e.g. OLLAMA_BASE_URL=http://localhost:11434 (Ollama) or "
                "OLLAMA_BASE_URL=http://localhost:8000/v1 (OpenAI-compatible)"
            )
        if not self.ollama_model.strip():
            problems.append(
                "OLLAMA_MODEL is empty - set it to the model tag or id, "
                "e.g. OLLAMA_MODEL=qwen3:8b (list tags with `ollama list`) or "
                "OLLAMA_MODEL=qwen3.8-27b for an OpenAI-compatible server"
            )
        return problems

    def log_summary(self) -> None:
        """Emit the effective configuration at startup (secrets excluded)."""
        logger.info(
            "Configuration loaded",
            extra={
                "ollama_base_url": self.ollama_base_url or "<unset>",
                "llm_api_flavor": self.resolved_api_flavor,
                "ollama_model": self.ollama_model or "<unset>",
                "embedding_model": self.embedding_model,
                "reranker_model": self.reranker_model if self.rerank_enabled else "<disabled>",
                "chroma_dir": str(self.chroma_path),
                "top_k_final": self.top_k_final,
                "cache_enabled": self.cache_enabled,
                "rate_limit_enabled": self.rate_limit_enabled,
                "ragobserve_enabled": self.ragobserve_enabled,
            },
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton."""
    return Settings()
