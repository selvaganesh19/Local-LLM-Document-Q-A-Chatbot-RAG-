"""Observability helpers.

Thin, failure-tolerant wrappers around `RAGObserve <https://pypi.org/project/ragobserve/>`_,
the local-first RAG tracing/evaluation toolkit.

Two rules govern this module:

1. **Observability must never break the application.**  Every call is guarded;
   import errors, disabled tracing, or a misbehaving backend all degrade to a
   no-op.  RAGObserve already swallows logging failures internally, but the
   import itself and the initialisation step are ours to guard.
2. **Call sites stay readable.**  A single ``if`` on the module flag means
   hot paths pay almost nothing when tracing is off.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Whether :func:`init_tracing` successfully enabled RAGObserve.
_ENABLED = False
logger_disabled_reason: Optional[str] = None


def _to_payload(items: Sequence[Any]) -> List[Dict[str, Any]]:
    """Convert chunks/hits into the dict shape RAGObserve renders.

    RAGObserve accepts bare strings, but passing dicts is what makes scores,
    ranks and sources show up in the dashboard, so everything is normalised to
    ``{"id", "text", "score", "rank", "source", "metadata"}``.
    """
    payload: List[Dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        if isinstance(item, str):
            payload.append({"id": f"chunk-{rank}", "text": item, "rank": rank})
            continue

        metadata = dict(getattr(item, "metadata", {}) or {})
        score = getattr(item, "relevance", None)
        if score is None:
            score = getattr(item, "score", None)
        if score is None:
            score = getattr(item, "fusion_score", None)

        payload.append(
            {
                "id": str(getattr(item, "chunk_id", metadata.get("chunk_id", rank))),
                "text": str(getattr(item, "text", "")),
                "score": float(score) if score is not None else None,
                "rank": rank,
                "source": str(metadata.get("source", getattr(item, "source", "unknown"))),
                "metadata": metadata,
            }
        )
    return payload


def init_tracing(settings: Optional[Settings] = None) -> bool:
    """Initialise RAGObserve.

    Args:
        settings: Configuration override; defaults to the process settings.

    Returns:
        ``True`` when tracing is active, ``False`` when it is disabled or the
        backend could not be initialised.
    """
    global _ENABLED, logger_disabled_reason

    config = settings or get_settings()
    if not config.ragobserve_enabled:
        _ENABLED = False
        logger_disabled_reason = "disabled by configuration"
        logger.info("RAGObserve tracing disabled")
        return False

    try:
        import ragobserve

        db_path: Path = config.ragobserve_db
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ragobserve.init(project=config.ragobserve_project, db_path=str(db_path))
        _ENABLED = True
        logger_disabled_reason = None
        logger.info(
            "RAGObserve tracing enabled",
            extra={"project": config.ragobserve_project, "db": str(db_path)},
        )
        return True
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        _ENABLED = False
        logger_disabled_reason = str(exc)
        logger.warning("RAGObserve tracing unavailable", extra={"error": str(exc)})
        return False


def is_enabled() -> bool:
    """Whether tracing is currently active."""
    return _ENABLED


@contextmanager
def trace_query(query: str, **kwargs: Any) -> Generator[None, None, None]:
    """Open a trace span for one RAG query.

    Degrades to a no-op context manager when tracing is disabled.

    Args:
        query: The user's question, recorded on the trace.
        **kwargs: Extra trace attributes passed through to RAGObserve.
    """
    if not _ENABLED:
        with nullcontext():
            yield
        return

    try:
        import ragobserve

        with ragobserve.trace("query", query=query, **kwargs):
            yield
    except Exception as exc:  # noqa: BLE001 - never fail the request for tracing
        logger.debug("Trace span failed", extra={"error": str(exc)})
        yield


def _call(name: str, *args: Any, **kwargs: Any) -> None:
    """Invoke a RAGObserve ``log_*`` function, swallowing any failure."""
    if not _ENABLED:
        return
    try:
        import ragobserve

        getattr(ragobserve, name)(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.debug("Trace call failed", extra={"call": name, "error": str(exc)})


# ---------------------------------------------------------------------------
# Stage-specific loggers
# ---------------------------------------------------------------------------
def log_ingestion(source: str, count: int, duration_ms: float) -> None:
    """Record a completed ingestion run."""
    _call("log_ingestion", source=source, count=count, sources=[source], duration_ms=duration_ms)


def log_chunks(chunks: Sequence[Any], strategy: str, chunk_size: int, overlap: int) -> None:
    """Record the chunking stage."""
    _call(
        "log_chunks",
        _to_payload(chunks),
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def log_embedding(model: str, input_count: int, dimensions: int, duration_ms: float) -> None:
    """Record the embedding stage."""
    _call(
        "log_embedding",
        model=model,
        input_count=input_count,
        dimensions=dimensions,
        duration_ms=duration_ms,
    )


def log_retrieval(
    query: str,
    results: Sequence[Any],
    retriever: str,
    top_k: int,
    duration_ms: float,
) -> None:
    """Record one retrieval leg (dense or lexical)."""
    _call(
        "log_retrieval",
        query,
        _to_payload(results),
        retriever=retriever,
        top_k=top_k,
        duration_ms=duration_ms,
    )


def log_fusion(results: Sequence[Any], inputs: Dict[str, Sequence[Any]], strategy: str) -> None:
    """Record the rank-fusion stage."""
    _call(
        "log_fusion",
        _to_payload(results),
        inputs={name: _to_payload(items) for name, items in inputs.items()},
        strategy=strategy,
    )


def log_rerank(before: Sequence[Any], after: Sequence[Any], model: str, duration_ms: float) -> None:
    """Record the reranking stage, including the order it replaced."""
    _call(
        "log_rerank",
        _to_payload(before),
        _to_payload(after),
        model=model,
        top_n=len(after),
        duration_ms=duration_ms,
    )


def log_context(final_prompt: str, query: str, system_prompt: str, chunks: Sequence[Any], context_window: int) -> None:
    """Record the assembled context handed to the model."""
    _call(
        "log_context",
        final_prompt,
        query=query,
        system_prompt=system_prompt,
        chunks=_to_payload(chunks),
        context_window=context_window,
    )


def log_generation(
    model: str,
    prompt: str,
    response: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: float,
) -> None:
    """Record the generation stage with token accounting."""
    _call(
        "log_generation",
        model=model,
        prompt=prompt,
        response=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
    )


def log_ground_truth(relevant_chunk_ids: Sequence[str]) -> None:
    """Attach ground-truth chunk ids to the active trace, for evaluation."""
    _call("log_ground_truth", relevant_chunk_ids=list(relevant_chunk_ids))


def current_trace_id() -> Optional[str]:
    """Return the active trace id, or ``None`` when tracing is off."""
    if not _ENABLED:
        return None
    try:
        import ragobserve

        return ragobserve.current_trace_id()
    except Exception:  # noqa: BLE001 - tracing is best-effort
        return None


def flush() -> None:
    """Flush pending trace events to the backing store."""
    if not _ENABLED:
        return
    try:
        import ragobserve

        ragobserve.flush()
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.debug("Trace flush failed", extra={"error": str(exc)})


class Stopwatch:
    """Context manager measuring elapsed milliseconds.

    Example:
        >>> with Stopwatch() as sw:
        ...     do_work()
        >>> sw.elapsed_ms
    """

    def __init__(self) -> None:
        """Create a stopped stopwatch."""
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "Stopwatch":
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop timing and store the elapsed milliseconds."""
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
