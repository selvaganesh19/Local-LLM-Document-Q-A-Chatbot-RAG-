"""Sentence-aware text chunking.

Splits documents into overlapping windows that respect sentence boundaries,
which keeps individual chunks semantically coherent for retrieval while the
overlap preserves context that would otherwise be cut in half.

Chunk identifiers are content-addressed (SHA-1 over source, position and text),
so re-ingesting an unchanged document is idempotent.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from app.ingestion.loaders import Document

logger = logging.getLogger(__name__)

#: Sentence terminators followed by whitespace, or one or more blank lines.
_SEGMENT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n\s*\n+")

_CHUNK_ID_LENGTH = 16


@dataclass
class Chunk:
    """A retrievable unit of text with full provenance.

    Attributes:
        chunk_id: Stable content-addressed identifier.
        text: The chunk text.
        source: Source document name used in citations.
        metadata: Provenance copied from the parent document plus
            ``chunk_index`` and ``char_start``.
        index: Zero-based position of this chunk within its source.
    """

    chunk_id: str
    text: str
    source: str
    metadata: Dict[str, object] = field(default_factory=dict)
    index: int = 0

    @property
    def char_count(self) -> int:
        """Length of the chunk text in characters."""
        return len(self.text)

    def to_record(self) -> Dict[str, object]:
        """Flatten to the plain mapping stored in the vector database."""
        merged = dict(self.metadata)
        merged.update(
            {
                "chunk_id": self.chunk_id,
                "source": self.source,
                "chunk_index": self.index,
            }
        )
        return merged


def _split_segments(text: str) -> List[str]:
    """Split ``text`` into non-empty sentence/paragraph segments."""
    return [segment.strip() for segment in _SEGMENT_PATTERN.split(text) if segment.strip()]


def _hard_split(segment: str, chunk_size: int) -> List[str]:
    """Break a single over-long segment into ``chunk_size`` slices."""
    return [segment[i : i + chunk_size] for i in range(0, len(segment), chunk_size)]


def _make_chunk_id(source: str, index: int, text: str) -> str:
    """Derive a stable identifier from the chunk's identity and content."""
    digest = hashlib.sha1(f"{source}|{index}|{text}".encode("utf-8")).hexdigest()
    return digest[:_CHUNK_ID_LENGTH]


def _overlap_tail(segments: Sequence[str], overlap: int) -> List[str]:
    """Return the trailing segments that fit within ``overlap`` characters."""
    if overlap <= 0:
        return []

    tail: List[str] = []
    length = 0
    for segment in reversed(segments):
        added = len(segment) + 1
        if tail and length + added > overlap:
            break
        if not tail and added > overlap:
            # The newest segment alone exceeds the overlap budget; taking it
            # would reproduce the whole chunk on the next window.
            break
        tail.insert(0, segment)
        length += added
    return tail


def chunk_text(
    text: str,
    source: str,
    metadata: Dict[str, object] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: Optional[int] = None,
    start_index: int = 0,
) -> List[Chunk]:
    """Split a single text into overlapping, sentence-aligned chunks.

    Args:
        text: The text to split.
        source: Source name recorded on every produced chunk.
        metadata: Provenance copied onto every produced chunk.
        chunk_size: Target maximum chunk length in characters.
        chunk_overlap: Characters of trailing context repeated at the start of
            the next chunk.  Defaults to roughly a seventh of ``chunk_size``,
            which keeps the ratio sensible at any window size.  Must be
            non-negative and smaller than ``chunk_size``.
        start_index: Index to assign to the first produced chunk.

    Returns:
        The produced chunks, in document order.

    Raises:
        ValueError: If ``chunk_size`` is not positive, or ``chunk_overlap`` is
            negative or not smaller than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    overlap = chunk_size // 7 if chunk_overlap is None else chunk_overlap
    if overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    base_metadata = dict(metadata or {})
    if not text.strip():
        return []

    segments: List[str] = []
    for segment in _split_segments(text):
        if len(segment) > chunk_size:
            segments.extend(_hard_split(segment, chunk_size))
        else:
            segments.append(segment)

    chunks: List[Chunk] = []
    current: List[str] = []
    current_length = 0
    cursor = 0

    def flush() -> None:
        """Materialise ``current`` as a chunk."""
        nonlocal cursor
        if not current:
            return
        body = " ".join(current).strip()
        if not body:
            return
        index = start_index + len(chunks)
        chunk_metadata = dict(base_metadata)
        chunk_metadata["char_start"] = cursor
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(source, index, body),
                text=body,
                source=source,
                metadata=chunk_metadata,
                index=index,
            )
        )
        cursor += len(body)

    for segment in segments:
        if current and current_length + len(segment) + 1 > chunk_size:
            flush()
            tail = _overlap_tail(current, overlap)
            # Never let the overlap reproduce the full previous chunk, which
            # would stall forward progress.
            if len(tail) >= len(current):
                tail = tail[1:]
            current = list(tail)
            current_length = sum(len(item) + 1 for item in current)

        current.append(segment)
        current_length += len(segment) + 1

    flush()
    return chunks


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 1000,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """Chunk a stream of documents, numbering chunks per source document.

    Args:
        documents: Documents produced by :mod:`app.ingestion.loaders`.
        chunk_size: Target maximum chunk length in characters.
        chunk_overlap: Characters of overlap between adjacent chunks; defaults
            to roughly a seventh of ``chunk_size``.

    Returns:
        All chunks across all documents, in input order.
    """
    materialised: List[Document] = list(documents)
    chunks: List[Chunk] = []
    for document in materialised:
        produced = chunk_text(
            text=document.text,
            source=document.source,
            metadata=document.metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(produced)

    logger.info(
        "Chunking complete",
        extra={
            "documents": len(materialised),
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap if chunk_overlap is not None else chunk_size // 7,
        },
    )
    return chunks
