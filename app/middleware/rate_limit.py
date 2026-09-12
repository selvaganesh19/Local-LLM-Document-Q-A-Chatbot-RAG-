"""Rate limiting.

Built on `slowapi <https://pypi.org/project/slowapi/>`_, which wraps the
``limits`` library and integrates with FastAPI's request lifecycle.

Limits are configured per route group (chat, ingestion, everything else) so a
chatty UI cannot starve ingestion, and a bulk re-index cannot exhaust the
generation budget.  Limits are keyed on the client address, with
``X-Forwarded-For`` respected when the app runs behind a reverse proxy.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import Settings, get_settings
from app.logging_config import request_id_var

logger = logging.getLogger(__name__)


def client_key(request: Request) -> str:
    """Derive the rate-limit bucket key for a request.

    Honours ``X-Forwarded-For`` so that limiting still works when the service
    sits behind a proxy; falls back to the peer address.

    Args:
        request: The incoming request.

    Returns:
        A stable per-client identifier.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


#: Configuration used by the dynamic limit providers.  Set by
#: :func:`configure_limiter` so limits follow the running application's settings
#: rather than whatever the process-wide singleton happened to be at import.
_active_settings: Optional[Settings] = None


def _current() -> Settings:
    """Return the active settings, falling back to the process singleton."""
    return _active_settings or get_settings()


def _chat_limit() -> str:
    """Chat endpoint limit, resolved per request so it stays configurable."""
    return _current().rate_limit_chat


def _ingest_limit() -> str:
    """Ingestion endpoint limit, resolved per request."""
    return _current().rate_limit_ingest


def _default_limit() -> str:
    """Fallback limit for read-only endpoints."""
    return _current().rate_limit_default


#: Process-wide limiter.  ``enabled`` is synchronised with configuration by
#: :func:`configure_limiter` during application startup.
limiter = Limiter(
    key_func=client_key,
    enabled=get_settings().rate_limit_enabled,
    headers_enabled=True,
    default_limits=[],
)


def configure_limiter(settings: Optional[Settings] = None) -> None:
    """Apply configuration to the shared limiter.

    Also records the settings object used by the dynamic limit providers, so a
    test or embedding application that builds its own :class:`Settings` gets
    the limits it asked for rather than the process defaults.

    Args:
        settings: Configuration override; defaults to process settings.
    """
    global _active_settings

    config = settings or get_settings()
    _active_settings = config
    limiter.enabled = config.rate_limit_enabled
    logger.info(
        "Rate limiting configured",
        extra={
            "enabled": config.rate_limit_enabled,
            "chat": config.rate_limit_chat,
            "ingest": config.rate_limit_ingest,
        },
    )


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Return a structured 429 instead of slowapi's plain-text default.

    Args:
        request: The rejected request.
        exc: The raised limit exception.

    Returns:
        A JSON response carrying the limit that was hit and a ``Retry-After``
        header.
    """
    detail = str(getattr(exc, "detail", "Rate limit exceeded"))
    logger.warning(
        "Rate limit exceeded",
        extra={"path": request.url.path, "client": client_key(request), "limit": detail},
    )

    request_id = request_id_var.get()
    payload = {
        "detail": "Rate limit exceeded. Slow down and retry shortly.",
        "limit": detail,
        "path": request.url.path,
    }
    if request_id:
        payload["request_id"] = request_id

    return Response(
        content=json.dumps(payload),
        status_code=429,
        media_type="application/json",
        headers={"Retry-After": "60"},
    )


#: Convenience decorators, kept here so route modules stay declarative.
chat_limit = limiter.limit(_chat_limit)
ingest_limit = limiter.limit(_ingest_limit)
read_limit = limiter.limit(_default_limit)
