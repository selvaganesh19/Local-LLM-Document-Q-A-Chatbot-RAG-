"""Request context middleware.

Implemented as a pure ASGI middleware rather than
``starlette.middleware.base.BaseHTTPMiddleware`` so that streaming responses
(``text/event-stream``) are passed through chunk by chunk instead of being
buffered until the generator finishes.

Responsibilities:

* assign or propagate an ``X-Request-ID``,
* expose it to the logging context so every line emitted while handling the
  request is correlated,
* emit one structured access-log line with status and duration.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Awaitable, Callable, MutableMapping

from starlette.datastructures import MutableHeaders

from app.logging_config import request_id_var

logger = logging.getLogger("app.access")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

#: Header used to carry the correlation id in and out.
REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware:
    """Attach a correlation id to every request and log its outcome."""

    def __init__(self, app: Any) -> None:
        """Wrap the downstream ASGI application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI call."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        request_id = headers.get(REQUEST_ID_HEADER.lower()) or uuid.uuid4().hex[:16]

        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            """Record the status code and echo the request id back."""
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        path = scope.get("path", "")
        method = scope.get("method", "")
        client = scope.get("client")
        client_host = client[0] if client else "unknown"

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "Request failed",
                extra={
                    "method": method,
                    "path": path,
                    "client": client_host,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise
        else:
            duration_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "Request handled",
                extra={
                    "method": method,
                    "path": path,
                    "status": status_code,
                    "client": client_host,
                    "duration_ms": round(duration_ms, 2),
                    "request_id": request_id,
                },
            )
        finally:
            request_id_var.reset(token)
