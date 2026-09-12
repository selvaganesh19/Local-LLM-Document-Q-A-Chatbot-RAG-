"""Document ingestion: loading, chunking, embedding and indexing."""

from app.ingestion.chunker import Chunk, chunk_documents, chunk_text
from app.ingestion.embedder import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from app.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    Document,
    load_directory,
    load_document,
    load_from_text,
)
from app.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "Chunk",
    "chunk_documents",
    "chunk_text",
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "get_embedder",
    "Document",
    "load_document",
    "load_directory",
    "load_from_text",
    "SUPPORTED_EXTENSIONS",
    "IngestionPipeline",
    "IngestionResult",
]
