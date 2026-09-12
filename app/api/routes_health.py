"""Health, statistics and configuration endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response

from app import __version__
from app.api.schemas import ConfigResponse, HealthResponse, OllamaStatus, StatsResponse
from app.container import ServiceContainer, get_container
from app.middleware.rate_limit import read_limit
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and readiness")
@read_limit
async def health(
    request: Request,
    response: Response,
    container: ServiceContainer = Depends(get_container),
) -> HealthResponse:
    """Report service status, including whether the local model is usable.

    ``status`` is one of:

    * ``ok`` - Ollama is reachable and the configured model is installed.
    * ``degraded`` - configured, but the server or model is unavailable.
    * ``unconfigured`` - ``OLLAMA_BASE_URL`` or ``OLLAMA_MODEL`` is missing.
    """
    errors = container.configuration_errors()
    ollama_report = await container.client.health()

    if errors:
        status_value = "unconfigured"
    elif ollama_report.get("reachable") and ollama_report.get("model_available"):
        status_value = "ok"
    else:
        status_value = "degraded"

    return HealthResponse(
        status=status_value,
        version=__version__,
        ollama=OllamaStatus(**ollama_report),
        configuration_errors=errors,
        index=container.retriever.index_stats(),
    )


@router.get("/stats", response_model=StatsResponse, summary="Index and cache statistics")
@read_limit
async def stats(
    request: Request,
    response: Response,
    container: ServiceContainer = Depends(get_container),
) -> StatsResponse:
    """Report index sizes, reranker in use and cache counters."""
    service = RagService(container)
    return StatsResponse(**service.index_stats())


@router.get("/config", response_model=ConfigResponse, summary="Non-secret configuration")
@read_limit
async def config(
    request: Request,
    response: Response,
    container: ServiceContainer = Depends(get_container),
) -> ConfigResponse:
    """Expose the configuration the web UI needs to render itself."""
    settings = container.settings
    return ConfigResponse(
        model=settings.ollama_model or "<unset>",
        llm_configured=container.is_llm_configured(),
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model if settings.rerank_enabled else "disabled",
        top_k_final=settings.top_k_final,
        rerank_enabled=settings.rerank_enabled,
        cache_enabled=settings.cache_enabled,
        rate_limit_enabled=settings.rate_limit_enabled,
        ragobserve_enabled=settings.ragobserve_enabled,
        dev_fake_llm=settings.dev_fake_llm,
    )
