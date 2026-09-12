"""Chat endpoints: grounded question answering with citations."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.container import ServiceContainer, get_container
from app.generation.ollama_client import OllamaError, OllamaNotConfiguredError
from app.middleware.rate_limit import chat_limit
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

#: Server-sent-events headers; ``X-Accel-Buffering`` disables proxy buffering.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _require_llm(container: ServiceContainer) -> None:
    """Reject the request when Ollama has not been configured.

    Args:
        container: The service container to inspect.

    Raises:
        HTTPException: 503 with actionable setup instructions.
    """
    if container.is_llm_configured():
        return

    problems = container.configuration_errors()
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "detail": "The local LLM is not configured, so answers cannot be generated.",
            "configuration_errors": problems,
            "hint": "Set OLLAMA_BASE_URL and OLLAMA_MODEL in your .env file, then restart the app.",
        },
    )


def _sse(event: Dict[str, Any]) -> str:
    """Encode one event as an SSE ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about the indexed documents",
    responses={
        503: {"model": ErrorResponse, "description": "Ollama is not configured"},
        502: {"model": ErrorResponse, "description": "Ollama is unreachable"},
    },
)
@chat_limit
async def chat(
    request: Request,
    response: Response,
    payload: ChatRequest,
    container: ServiceContainer = Depends(get_container),
) -> ChatResponse:
    """Retrieve relevant passages and generate a cited answer.

    Returns the answer together with the passages it cited, every passage that
    was retrieved, per-stage timings and the trace id of the run.
    """
    _require_llm(container)
    service = RagService(container)

    try:
        result = await service.answer(
            question=payload.question,
            top_k=payload.top_k,
            source_filter=payload.source_filter,
            use_cache=payload.use_cache,
        )
    except OllamaNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except OllamaError as exc:
        logger.error("Generation failed", extra={"error": str(exc)})
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return ChatResponse(**result)


@router.post(
    "/chat/stream",
    summary="Ask a question and stream the answer as server-sent events",
    response_class=StreamingResponse,
)
@chat_limit
async def chat_stream(
    request: Request,
    response: Response,
    payload: ChatRequest,
    container: ServiceContainer = Depends(get_container),
) -> StreamingResponse:
    """Stream a cited answer token by token.

    The event stream carries JSON frames of the form
    ``{"type": "sources" | "timings" | "trace" | "token" | "done" | "error", ...}``
    emitted in a fixed order:

    1. ``timings`` - retrieval timings, sent as soon as retrieval finishes.
    2. ``trace`` - the RAGObserve trace id, when tracing is enabled.
    3. ``sources`` - every passage that will be offered to the model.
    4. ``token`` - streamed answer fragments, repeated.
    5. ``done`` - the full response body, including parsed citations.

    Clients may ignore every intermediate frame and render only ``done``. A
    terminal ``data: [DONE]`` sentinel closes the stream on every path,
    including errors, so a client can tell a clean end from a dropped socket.
    """
    _require_llm(container)
    service = RagService(container)

    async def event_stream() -> AsyncIterator[str]:
        """Yield SSE frames, converting failures into an error frame."""
        try:
            async for event in service.stream_answer(
                question=payload.question,
                top_k=payload.top_k,
                source_filter=payload.source_filter,
            ):
                yield _sse(event)
        except OllamaNotConfiguredError as exc:
            yield _sse({"type": "error", "detail": str(exc), "status": 503})
        except OllamaError as exc:
            logger.error("Streamed generation failed", extra={"error": str(exc)})
            yield _sse({"type": "error", "detail": str(exc), "status": 502})
        except Exception as exc:  # noqa: BLE001 - the stream must terminate cleanly
            logger.exception("Unexpected streaming failure")
            yield _sse({"type": "error", "detail": f"Unexpected error: {exc}", "status": 500})

        # Sentinel so clients can distinguish a clean end from a dropped socket.
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)
