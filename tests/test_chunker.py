"""Tests for sentence-aware chunking."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import chunk_documents, chunk_text
from app.ingestion.loaders import Document


def test_empty_text_produces_no_chunks() -> None:
    """Whitespace-only input yields nothing rather than an empty chunk."""
    assert chunk_text("   \n\n  ", source="empty.txt") == []


def test_short_text_produces_one_chunk() -> None:
    """Text under the size limit stays intact."""
    chunks = chunk_text("A short sentence.", source="short.txt", chunk_size=100)

    assert len(chunks) == 1
    assert chunks[0].text == "A short sentence."
    assert chunks[0].source == "short.txt"
    assert chunks[0].index == 0


def test_long_text_produces_overlapping_chunks() -> None:
    """Adjacent chunks share trailing context, up to the overlap budget."""
    sentences = [f"Sentence number {index} carries some words." for index in range(40)]
    text = " ".join(sentences)

    chunks = chunk_text(text, source="long.txt", chunk_size=200, chunk_overlap=60)

    assert len(chunks) > 1
    assert all(chunk.char_count <= 200 for chunk in chunks)

    # Every chunk but the first should begin with content from its predecessor.
    for previous, current in zip(chunks, chunks[1:]):
        first_sentence = current.text.split(".")[0]
        assert first_sentence in previous.text


def test_chunk_ids_are_deterministic() -> None:
    """Identical input produces identical ids, so re-ingestion is idempotent."""
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."

    first = chunk_text(text, source="doc.txt", chunk_size=80, chunk_overlap=10)
    second = chunk_text(text, source="doc.txt", chunk_size=80, chunk_overlap=10)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_chunk_ids_differ_across_sources() -> None:
    """The same text in two documents must not collide."""
    text = "Identical content in two files. Repeated verbatim."

    one = chunk_text(text, source="a.txt", chunk_size=100)
    two = chunk_text(text, source="b.txt", chunk_size=100)

    assert one[0].chunk_id != two[0].chunk_id


def test_metadata_is_carried_and_extended() -> None:
    """Provenance fields survive and char_start is added."""
    chunks = chunk_text(
        "First sentence here. Second sentence follows.",
        source="meta.txt",
        metadata={"page": 7, "type": "pdf"},
        chunk_size=100,
    )

    assert chunks[0].metadata["page"] == 7
    assert chunks[0].metadata["type"] == "pdf"
    assert chunks[0].metadata["char_start"] == 0
    assert chunks[0].to_record()["source"] == "meta.txt"
    assert chunks[0].to_record()["chunk_index"] == 0


def test_oversized_sentence_is_hard_split() -> None:
    """A single sentence longer than the window is sliced rather than dropped."""
    text = "word " * 300  # 1500 characters, no sentence boundaries

    chunks = chunk_text(text, source="mono.txt", chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.char_count <= 200 for chunk in chunks)


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap",
    [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -5)],
)
def test_invalid_window_parameters_are_rejected(chunk_size: int, chunk_overlap: int) -> None:
    """Nonsensical window sizes raise instead of looping forever."""
    with pytest.raises(ValueError):
        chunk_text("Some text.", source="x.txt", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunk_documents_spans_multiple_documents() -> None:
    """Every document contributes chunks, numbered independently."""
    documents = [
        Document(text="First document text. More words here.", source="one.txt"),
        Document(text="Second document text. Different words.", source="two.txt"),
    ]

    chunks = chunk_documents(documents, chunk_size=200, chunk_overlap=20)

    assert {chunk.source for chunk in chunks} == {"one.txt", "two.txt"}
    assert [chunk.index for chunk in chunks if chunk.source == "one.txt"] == [0]


def test_chunk_documents_accepts_a_generator() -> None:
    """A generator input is materialised once and fully consumed."""
    documents = (Document(text=f"Document {index} text.", source=f"{index}.txt") for index in range(3))

    chunks = chunk_documents(documents, chunk_size=200, chunk_overlap=10)

    assert len(chunks) == 3
