"""Tests for the RAGObserve tracing wrapper.

These exercise the real ``ragobserve`` package - it is a hard dependency - to
confirm the integration actually initialises, writes a trace database, and
degrades cleanly when disabled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.observability import tracing
from app.retrieval.hybrid import RetrievedChunk


@pytest.fixture
def tracing_settings(settings: Settings, tmp_path: Path) -> Settings:
    """Settings with tracing enabled and a throwaway database."""
    return settings.model_copy(
        update={
            "ragobserve_enabled": True,
            "ragobserve_db_path": str(tmp_path / "traces" / "ragobserve.db"),
        }
    )


@pytest.fixture(autouse=True)
def _restore_tracing(settings: Settings):
    """Leave global tracing state disabled after each test."""
    yield
    tracing.init_tracing(settings.model_copy(update={"ragobserve_enabled": False}))


def _chunks() -> list[RetrievedChunk]:
    """Two candidate chunks for trace payloads."""
    return [
        RetrievedChunk(
            chunk_id="c1",
            text="The notice period is 90 days.",
            metadata={"source": "handbook.txt", "page": 1},
            fusion_score=0.03,
            relevance=0.9,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Passwords rotate every 180 days.",
            metadata={"source": "security.txt"},
            fusion_score=0.02,
            relevance=0.5,
        ),
    ]


def test_disabled_tracing_is_a_noop(settings: Settings) -> None:
    """With tracing off, every helper is safe to call and reports no trace id."""
    disabled = settings.model_copy(update={"ragobserve_enabled": False})

    assert tracing.init_tracing(disabled) is False
    assert tracing.is_enabled() is False

    with tracing.trace_query("question"):
        tracing.log_retrieval("question", _chunks(), retriever="test", top_k=2, duration_ms=1.0)
        tracing.log_generation(
            model="m", prompt="p", response="r", input_tokens=1, output_tokens=1, duration_ms=1.0
        )

    assert tracing.current_trace_id() is None
    tracing.flush()  # must not raise


def test_tracing_initialises_and_writes_a_database(tracing_settings: Settings) -> None:
    """Enabling tracing creates the SQLite store on disk."""
    assert tracing.init_tracing(tracing_settings) is True
    assert tracing.is_enabled() is True

    with tracing.trace_query("What is the notice period?"):
        tracing.log_retrieval(
            "What is the notice period?", _chunks(), retriever="chroma-hnsw", top_k=2, duration_ms=4.2
        )
        tracing.log_fusion(_chunks(), inputs={"dense": _chunks()}, strategy="rrf(k=60)")
        tracing.log_rerank(_chunks(), _chunks()[:1], model="cross-encoder", duration_ms=8.0)
        tracing.log_context(
            "prompt body", query="q", system_prompt="sys", chunks=_chunks(), context_window=8192
        )
        tracing.log_generation(
            model="qwen3:8b", prompt="prompt body", response="Answer [1].",
            input_tokens=120, output_tokens=8, duration_ms=900.0,
        )
        tracing.log_ingestion(source="handbook.txt", count=3, duration_ms=12.0)
        tracing.log_chunks(_chunks(), strategy="sentence-aware-overlap", chunk_size=1000, overlap=150)
        tracing.log_embedding(model="bge-small", input_count=2, dimensions=384, duration_ms=5.0)
        tracing.log_ground_truth(["c1"])

    tracing.flush()

    assert Path(tracing_settings.ragobserve_db_path).exists()


def test_trace_id_is_available_inside_a_span(tracing_settings: Settings) -> None:
    """A span exposes its id, which the API returns to the caller."""
    tracing.init_tracing(tracing_settings)

    with tracing.trace_query("question"):
        trace_id = tracing.current_trace_id()

    assert trace_id


def test_payload_conversion_handles_chunks_and_strings() -> None:
    """Scores, ranks and sources survive conversion for the dashboard."""
    payload = tracing._to_payload(_chunks() + ["a bare string"])

    assert payload[0]["id"] == "c1"
    assert payload[0]["source"] == "handbook.txt"
    assert payload[0]["rank"] == 1
    assert payload[0]["score"] == pytest.approx(0.9)
    assert payload[0]["metadata"]["page"] == 1

    assert payload[2]["id"] == "chunk-3"
    assert payload[2]["text"] == "a bare string"


def test_payload_conversion_of_nothing_is_empty() -> None:
    """An empty payload list is fine."""
    assert tracing._to_payload([]) == []


def test_stopwatch_measures_elapsed_time() -> None:
    """The stopwatch reports a positive duration."""
    with tracing.Stopwatch() as stopwatch:
        sum(range(1000))

    assert stopwatch.elapsed_ms >= 0.0


def test_init_tracing_with_an_unwritable_path_degrades(settings: Settings) -> None:
    """A broken trace path disables tracing instead of failing startup."""
    # A path under a *file* cannot be created as a directory.
    blocker = Path(settings.chroma_path) / "blocker"
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a directory", encoding="utf-8")

    broken = settings.model_copy(
        update={"ragobserve_enabled": True, "ragobserve_db_path": str(blocker / "traces.db")}
    )

    assert tracing.init_tracing(broken) is False
    assert tracing.is_enabled() is False
