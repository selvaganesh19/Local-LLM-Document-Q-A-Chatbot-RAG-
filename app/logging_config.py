"""Logging configuration.

Provides a human-readable console formatter and an optional JSON formatter for
machine ingestion.  A :data:`contextvars.ContextVar` carries the current request
id so every log line emitted while handling a request can be correlated.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

#: Correlation id of the request currently being handled (if any).
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Attributes present on every ``logging.LogRecord``; anything else that a caller
# passes via ``extra=`` is treated as structured context.
_RESERVED_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


def _extract_context(record: logging.LogRecord) -> Dict[str, Any]:
    """Return the caller-supplied ``extra`` fields of ``record``."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED_ATTRS and not key.startswith("_")
    }


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None) or request_id_var.get()
        if request_id:
            payload["request_id"] = request_id

        context = _extract_context(record)
        context.pop("request_id", None)
        if context:
            payload["context"] = context

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """Compact, aligned single-line console output with optional context."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)

        request_id = getattr(record, "request_id", None) or request_id_var.get()
        if request_id:
            base = f"{base} | req={request_id}"

        context = _extract_context(record)
        context.pop("request_id", None)
        if context:
            rendered = " ".join(f"{key}={value}" for key, value in context.items())
            base = f"{base} | {rendered}"

        return base


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the root logger.

    Args:
        level: Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        json_logs: Emit JSON lines instead of human-readable text.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Replace any pre-existing handlers so repeated calls stay idempotent.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else ConsoleFormatter())
    root.addHandler(handler)

    # Third-party loggers that are noisy at INFO.
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Chroma's telemetry is off by default but the logger still chatters.
    logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; thin wrapper that keeps call sites terse."""
    return logging.getLogger(name)
