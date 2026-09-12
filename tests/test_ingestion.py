"""Tests for document loading and the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.container import ServiceContainer
from app.ingestion.loaders import (
    Document,
    DocumentLoadError,
    load_directory,
    load_document,
    load_from_text,
)


def test_load_from_text_wraps_a_string() -> None:
    """Raw text becomes a single labelled document."""
    documents = load_from_text("Some content.", "notes.txt")

    assert len(documents) == 1
    assert documents[0].text == "Some content."
    assert documents[0].source == "notes.txt"


def test_document_flags_empty_content() -> None:
    """Whitespace-only documents are detectable so they can be skipped."""
    assert Document(text="   ", source="x.txt").is_empty() is True
    assert Document(text="real", source="x.txt").is_empty() is False
    assert Document(text="abc", source="x.txt").char_count == 3


def test_load_text_file(tmp_path: Path) -> None:
    """A plain text file loads with its file name as the source."""
    path = tmp_path / "notes.txt"
    path.write_text("Line one.\nLine two.", encoding="utf-8")

    documents = load_document(path)

    assert len(documents) == 1
    assert documents[0].source == "notes.txt"
    assert "Line two." in documents[0].text
    assert documents[0].metadata["type"] == "txt"


def test_load_markdown_file(tmp_path: Path) -> None:
    """Markdown is treated as text."""
    path = tmp_path / "guide.md"
    path.write_text("# Heading\n\nBody text.", encoding="utf-8")

    documents = load_document(path)

    assert documents[0].metadata["type"] == "md"
    assert "Body text." in documents[0].text


def test_load_empty_file_yields_nothing(tmp_path: Path) -> None:
    """A file with no usable text produces no documents."""
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  ", encoding="utf-8")

    assert load_document(path) == []


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """A missing file reports the path rather than a bare OSError."""
    with pytest.raises(DocumentLoadError, match="File not found"):
        load_document(tmp_path / "absent.txt")


def test_load_unsupported_extension_raises(tmp_path: Path) -> None:
    """Unsupported types are rejected with the supported list."""
    path = tmp_path / "data.xyz"
    path.write_text("content", encoding="utf-8")

    with pytest.raises(DocumentLoadError) as excinfo:
        load_document(path)

    assert ".pdf" in str(excinfo.value)


def test_load_html_strips_markup(tmp_path: Path) -> None:
    """HTML loads as visible text with scripts and styles removed."""
    pytest.importorskip("bs4")
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><style>p{color:red}</style></head>"
        "<body><script>alert(1)</script><p>Visible text.</p></body></html>",
        encoding="utf-8",
    )

    documents = load_document(path)

    assert "Visible text." in documents[0].text
    assert "alert" not in documents[0].text
    assert "color:red" not in documents[0].text


def test_load_directory_skips_unsupported_files(tmp_path: Path) -> None:
    """Only supported extensions are read; others are ignored silently."""
    (tmp_path / "good.txt").write_text("Content here.", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01")

    documents, failures = load_directory(tmp_path)

    assert len(documents) == 1
    assert failures == []


def test_load_directory_reports_failures_without_aborting(tmp_path: Path) -> None:
    """One unreadable file does not stop the scan."""
    (tmp_path / "good.txt").write_text("Content here.", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"definitely not a pdf")

    documents, failures = load_directory(tmp_path)

    assert len(documents) == 1
    assert len(failures) == 1
    assert "broken.pdf" in failures[0]


def test_load_directory_rejects_a_file_path(tmp_path: Path) -> None:
    """Passing a file instead of a directory is an error, not an empty result."""
    path = tmp_path / "file.txt"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Not a directory"):
        load_directory(path)


def test_load_directory_can_skip_subdirectories(tmp_path: Path) -> None:
    """Recursion is optional."""
    (tmp_path / "top.txt").write_text("Top level.", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.txt").write_text("Deep level.", encoding="utf-8")

    shallow, _ = load_directory(tmp_path, recursive=False)
    deep, _ = load_directory(tmp_path, recursive=True)

    assert {document.source for document in shallow} == {"top.txt"}
    assert {document.source for document in deep} == {"top.txt", "deep.txt"}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def test_pipeline_indexes_both_stores(container: ServiceContainer) -> None:
    """Ingestion populates the dense and lexical indexes together."""
    from tests.conftest import SAMPLE_DOCUMENTS

    result = container.pipeline.ingest_documents(SAMPLE_DOCUMENTS)

    assert result.chunks > 0
    assert result.sources == ["api.txt", "handbook.txt", "security.txt"]
    assert result.embedder
    assert container.vector_store.count() == result.chunks
    assert container.bm25_index.count == result.chunks


def test_pipeline_persists_the_lexical_index(container: ServiceContainer) -> None:
    """The BM25 index is written to disk so a restart does not re-tokenise."""
    from tests.conftest import SAMPLE_DOCUMENTS

    container.pipeline.ingest_documents(SAMPLE_DOCUMENTS)

    assert container.settings.bm25_index_path.exists()


def test_pipeline_ingest_text(container: ServiceContainer) -> None:
    """Pasted text is indexed under the supplied source name."""
    result = container.pipeline.ingest_text("A short pasted note about widgets.", "note.txt")

    assert result.sources == ["note.txt"]
    assert container.vector_store.count() > 0


def test_pipeline_ingest_files_collects_failures(container: ServiceContainer, tmp_path: Path) -> None:
    """A bad file is reported while the good ones still index."""
    good = tmp_path / "good.txt"
    good.write_text("Valid content for indexing.", encoding="utf-8")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")

    result = container.pipeline.ingest_files([good, bad])

    assert result.chunks > 0
    assert len(result.failures) == 1


def test_pipeline_ingest_empty_input_is_a_noop(container: ServiceContainer) -> None:
    """Nothing in means nothing indexed."""
    result = container.pipeline.ingest_documents([])

    assert result.chunks == 0
    assert container.vector_store.count() == 0


def test_pipeline_delete_source_removes_from_both_stores(ingested_container: ServiceContainer) -> None:
    """Deleting a document cleans up vectors and lexical entries alike."""
    removed = ingested_container.pipeline.delete_source("security.txt")

    assert removed["vectors"] > 0
    assert removed["lexical"] > 0
    assert {entry["source"] for entry in ingested_container.vector_store.list_sources()} == {
        "handbook.txt",
        "api.txt",
    }


def test_pipeline_reset_clears_everything(ingested_container: ServiceContainer) -> None:
    """Reset empties both indexes."""
    ingested_container.pipeline.reset()

    assert ingested_container.vector_store.count() == 0
    assert ingested_container.bm25_index.count == 0


def test_pipeline_index_stats(ingested_container: ServiceContainer) -> None:
    """Stats report matching counts across both indexes."""
    stats = ingested_container.pipeline.index_stats()

    assert stats["vectors"] == stats["lexical_chunks"]
    assert stats["sources"] == 3
