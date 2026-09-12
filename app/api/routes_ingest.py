"""Ingestion and document-management endpoints."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status

from app.api.schemas import (
    DeleteResponse,
    DocumentList,
    IngestPathRequest,
    IngestResponse,
    IngestTextRequest,
)
from app.container import ServiceContainer, get_container
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, DocumentLoadError, load_document
from app.middleware.rate_limit import ingest_limit, read_limit
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


def _resolve_allowed_path(raw: str, container: ServiceContainer) -> Path:
    """Validate a server-side path against the configured ingestion root.

    Accepting arbitrary filesystem paths over HTTP would let any caller read
    any file the service account can reach.  Unless
    ``ALLOW_INGEST_ANY_PATH=true`` is set explicitly, paths are confined to
    ``data/documents``.

    Args:
        raw: The client-supplied path.
        container: Container holding the active settings.

    Returns:
        The resolved, permitted path.

    Raises:
        HTTPException: 403 when the path escapes the permitted root, 404 when
            it does not exist.
    """
    candidate = Path(raw).expanduser().resolve()
    if not container.settings.allow_ingest_any_path:
        root = container.settings.documents_dir.resolve()
        if not candidate.is_relative_to(root):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "detail": "Path is outside the permitted ingestion directory.",
                    "permitted_root": str(root),
                    "hint": "Place the file under data/documents, upload it instead, "
                    "or set ALLOW_INGEST_ANY_PATH=true to lift this restriction.",
                },
            )

    if not candidate.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path does not exist: {candidate}",
        )
    return candidate


@router.post("/ingest/text", response_model=IngestResponse, summary="Ingest pasted text")
@ingest_limit
async def ingest_text(
    request: Request,
    response: Response,
    payload: IngestTextRequest,
    container: ServiceContainer = Depends(get_container),
) -> IngestResponse:
    """Chunk, embed and index a block of text."""
    service = RagService(container)
    result = service.ingest_text(payload.text, payload.source_name)
    return IngestResponse(**result)


@router.post("/ingest/upload", response_model=IngestResponse, summary="Upload and ingest a file")
@ingest_limit
async def ingest_upload(
    request: Request,
    response: Response,
    file: UploadFile = File(..., description="PDF, DOCX, Markdown, text or HTML file."),
    container: ServiceContainer = Depends(get_container),
) -> IngestResponse:
    """Ingest an uploaded document.

    The upload is streamed to a temporary file with the original extension so
    the existing loaders can be reused, then removed once ingestion finishes.
    """
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "detail": f"Unsupported file type '{suffix or 'unknown'}'.",
                "supported": sorted(SUPPORTED_EXTENSIONS),
            },
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
    if len(data) > container.settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File is {len(data) / 1_048_576:.1f} MB; the limit is "
                f"{container.settings.max_upload_mb} MB."
            ),
        )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(data)
            temp_path = Path(handle.name)

        # Reuse the normal loaders, then rewrite the recorded source name to the
        # user's original filename so citations read sensibly rather than
        # referencing a temporary file.
        documents = load_document(temp_path)
        for document in documents:
            document.source = filename

        service = RagService(container)
        result = service.ingest_documents(documents)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return IngestResponse(**result)


@router.post("/ingest/path", response_model=IngestResponse, summary="Ingest a server-side path")
@ingest_limit
async def ingest_path(
    request: Request,
    response: Response,
    payload: IngestPathRequest,
    container: ServiceContainer = Depends(get_container),
) -> IngestResponse:
    """Ingest a file or directory that already exists on the server."""
    target = _resolve_allowed_path(payload.path, container)
    service = RagService(container)

    try:
        if target.is_dir():
            result = service.ingest_directory(str(target), recursive=payload.recursive)
        else:
            result = service.ingest_files([target])
    except DocumentLoadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return IngestResponse(**result)


@router.get("/documents", response_model=DocumentList, summary="List indexed documents")
@read_limit
async def list_documents(
    request: Request,
    response: Response,
    container: ServiceContainer = Depends(get_container),
) -> DocumentList:
    """List every source document currently in the index."""
    service = RagService(container)
    documents = service.list_documents()
    return DocumentList(
        documents=documents,
        total_chunks=sum(int(item.get("chunks", 0)) for item in documents),
    )


@router.delete("/documents/{source}", response_model=DeleteResponse, summary="Delete a source document")
@ingest_limit
async def delete_document(
    request: Request,
    response: Response,
    source: str,
    container: ServiceContainer = Depends(get_container),
) -> DeleteResponse:
    """Remove every chunk belonging to ``source`` from both indexes."""
    service = RagService(container)
    result = service.delete_source(source)
    if not result.get("vectors") and not result.get("lexical"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No indexed chunks found for source '{source}'.",
        )
    return DeleteResponse(**result)


@router.post("/documents/reset", summary="Delete the entire index")
@ingest_limit
async def reset_index(
    request: Request,
    response: Response,
    container: ServiceContainer = Depends(get_container),
) -> dict:
    """Drop every chunk from both indexes and clear the cache.

    Destructive and irreversible: the documents must be re-ingested.
    """
    service = RagService(container)
    return service.reset_index()
