"""FastAPI application factory.

Composition happens here and nowhere else: the container is built, middleware is
ordered, routers are mounted, and the web UI is served from ``/``.

Ordering matters.  ``CORSMiddleware`` is added last so that it is the outermost
layer and its headers are attached even to error responses; the request-context
middleware sits just inside it so the correlation id is present for everything
downstream.

Run with::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

from app import __version__
from app.api import routes_chat, routes_health, routes_ingest
from app.config import Settings, get_settings
from app.container import ServiceContainer
from app.logging_config import request_id_var, setup_logging
from app.middleware.rate_limit import configure_limiter, limiter, rate_limit_handler
from app.middleware.request_context import RequestContextMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown.

    Startup restores persisted state and reports - loudly, but without
    crashing - when the mandatory Ollama settings are missing, so the web UI
    still loads and can explain what to configure.
    """
    container: ServiceContainer = app.state.container
    container.startup()
    container.settings.log_summary()

    missing = container.configuration_errors()
    if missing:
        logger.error(
            "LLM is not configured - /api/chat will return 503 until this is fixed",
            extra={"problems": missing},
        )
    else:
        logger.info("LLM configured", extra={"model": container.settings.ollama_model})

    logger.info("Application ready", extra={"version": __version__})
    try:
        yield
    finally:
        await container.shutdown()
        logger.info("Application shut down")


def create_app(
    settings: Optional[Settings] = None,
    container: Optional[ServiceContainer] = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Configuration override; defaults to process settings.  Used
            by the test suite to point storage at a temporary directory.
        container: Pre-built container; usually only supplied by tests.

    Returns:
        The configured application.
    """
    config = settings or get_settings()
    setup_logging(config.log_level, config.log_json)

    app = FastAPI(
        title="Local LLM Document Q&A (RAG)",
        description=(
            "Retrieval-augmented question answering over your own documents, "
            "powered entirely by a locally hosted Ollama model. Hybrid BM25 + "
            "semantic retrieval, cross-encoder reranking, and inline source "
            "citations."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    app.state.settings = config
    app.state.container = container or ServiceContainer(settings=config)
    app.state.limiter = limiter

    configure_limiter(config)

    # Middleware: added inner-first, so CORS ends up outermost.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # Error handlers.
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return a structured 500 instead of a bare traceback."""
        request_id = request_id_var.get()
        logger.exception("Unhandled exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error.",
                "request_id": request_id,
                "type": exc.__class__.__name__,
            },
        )

    # API routes.
    app.include_router(routes_chat.router)
    app.include_router(routes_ingest.router)
    app.include_router(routes_health.router)

    # Static web UI. Mounted last so /api routes win; `html=True` serves
    # index.html for GET /.
    static_dir = config.static_dir
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")
    else:  # pragma: no cover - only when the front-end is absent
        logger.warning("Static UI directory missing", extra={"path": str(static_dir)})

    return app


app = create_app()
