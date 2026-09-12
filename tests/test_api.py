"""End-to-end tests for the HTTP API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import ServiceContainer
from app.generation.ollama_client import OllamaClient
from app.ingestion.embedder import HashingEmbedder
from app.ingestion.loaders import Document
from app.main import create_app
from app.middleware.rate_limit import configure_limiter, limiter
from app.retrieval.reranker import NoOpReranker

QUESTION = "What is the notice period for senior staff?"


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------
def test_health_reports_ok_when_the_model_is_available(client: TestClient) -> None:
    """A configured, reachable model yields an 'ok' status."""
    payload = client.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["ollama"]["reachable"] is True
    assert payload["ollama"]["model_available"] is True
    assert payload["configuration_errors"] == []


def test_health_reports_unconfigured_when_settings_are_blank(
    unconfigured_settings: Settings,
) -> None:
    """Missing mandatory settings are surfaced with actionable text."""
    container = ServiceContainer(
        settings=unconfigured_settings,
        embedder=HashingEmbedder(dimension=64),
        reranker=NoOpReranker(),
        client=OllamaClient(settings=unconfigured_settings),
    )
    application = create_app(settings=unconfigured_settings, container=container)

    with TestClient(application) as test_client:
        payload = test_client.get("/api/health").json()

    assert payload["status"] == "unconfigured"
    assert any("OLLAMA_BASE_URL" in problem for problem in payload["configuration_errors"])
    assert any("OLLAMA_MODEL" in problem for problem in payload["configuration_errors"])


def test_config_exposes_non_secret_settings(client: TestClient) -> None:
    """The UI can read what it needs to render itself."""
    payload = client.get("/api/config").json()

    assert payload["llm_configured"] is True
    assert payload["model"] == "test-model"
    assert payload["top_k_final"] == 3
    assert payload["dev_fake_llm"] is False


def test_stats_reflects_the_index(client: TestClient) -> None:
    """Stats report index sizes and cache counters."""
    payload = client.get("/api/stats").json()

    assert "vectors" in payload
    assert "lexical_chunks" in payload
    assert payload["cache"]["enabled"] is True


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    """The correlation id is echoed so a client can quote it in a bug report."""
    response = client.get("/api/health")

    assert response.headers.get("X-Request-ID")


def test_supplied_request_id_is_preserved(client: TestClient) -> None:
    """An inbound request id is propagated rather than replaced."""
    response = client.get("/api/health", headers={"X-Request-ID": "abc123"})

    assert response.headers["X-Request-ID"] == "abc123"


def test_openapi_schema_is_served(client: TestClient) -> None:
    """The generated API documentation is available."""
    assert client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------------------
# Static web UI
# ---------------------------------------------------------------------------
def test_web_ui_is_served_at_the_root(client: TestClient) -> None:
    """The chat interface is mounted at / without shadowing the API."""
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Document Q&amp;A" in response.text


def test_web_ui_assets_are_served(client: TestClient) -> None:
    """The stylesheet and script are reachable."""
    stylesheet = client.get("/styles.css")
    script = client.get("/app.js")

    assert stylesheet.status_code == 200
    assert "text/css" in stylesheet.headers["content-type"]
    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]


def test_api_routes_take_precedence_over_the_static_mount(client: TestClient) -> None:
    """A static mount at / must not swallow /api paths."""
    assert client.get("/api/health").json()["version"]
    assert client.get("/api/nonexistent").status_code != 200


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------
def test_ingest_text_then_list_documents(client: TestClient) -> None:
    """Text can be indexed and then appears in the document list."""
    ingest = client.post(
        "/api/ingest/text",
        json={"text": "The office opens at 08:00 and closes at 18:00.", "source_name": "office.txt"},
    )

    assert ingest.status_code == 200
    assert ingest.json()["sources"] == ["office.txt"]

    documents = client.get("/api/documents").json()
    assert [entry["source"] for entry in documents["documents"]] == ["office.txt"]
    assert documents["total_chunks"] > 0


def test_ingest_text_rejects_an_empty_body(client: TestClient) -> None:
    """A blank question body fails validation rather than indexing nothing."""
    assert client.post("/api/ingest/text", json={"text": ""}).status_code == 422


def test_ingest_text_without_a_name_uses_the_default(client: TestClient) -> None:
    """Omitting the source name still produces a labelled document."""
    payload = client.post("/api/ingest/text", json={"text": "Some indexable content."}).json()

    assert payload["sources"] == ["pasted-text.txt"]


def test_upload_rejects_an_unsupported_type(client: TestClient) -> None:
    """Only known document types are accepted."""
    response = client.post(
        "/api/ingest/upload",
        files={"file": ("payload.exe", b"MZ\x90\x00", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert ".pdf" in str(response.json()["detail"])


def test_upload_rejects_an_empty_file(client: TestClient) -> None:
    """An empty upload is a client error."""
    response = client.post(
        "/api/ingest/upload",
        files={"file": ("blank.txt", b"", "text/plain")},
    )

    assert response.status_code == 400


def test_upload_indexes_a_text_file(client: TestClient) -> None:
    """An uploaded document is indexed under its original file name."""
    content = b"Support hours are 09:00 to 17:00 on weekdays."

    response = client.post(
        "/api/ingest/upload",
        files={"file": ("support.txt", content, "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["sources"] == ["support.txt"]


def test_ingest_path_outside_the_permitted_root_is_forbidden(
    client: TestClient, tmp_path: Path
) -> None:
    """Server-side paths are confined to the documents directory by default."""
    outside = tmp_path / "secret.txt"
    outside.write_text("Should not be readable through the API.", encoding="utf-8")

    response = client.post("/api/ingest/path", json={"path": str(outside)})

    assert response.status_code == 403
    assert "permitted" in str(response.json()["detail"]).lower()


def test_ingest_path_missing_file_is_not_found(client: TestClient) -> None:
    """A permitted but absent path reports 404."""
    missing = str(Path("./data/documents").resolve() / "does-not-exist.txt")

    response = client.post("/api/ingest/path", json={"path": missing})

    assert response.status_code == 404


def test_delete_unknown_source_is_not_found(client: TestClient) -> None:
    """Deleting something that was never indexed reports 404."""
    assert client.delete("/api/documents/nothing.txt").status_code == 404


def test_delete_known_source(ingested_client: TestClient) -> None:
    """Deleting an indexed document removes it."""
    response = ingested_client.delete("/api/documents/security.txt")

    assert response.status_code == 200
    assert response.json()["vectors"] > 0

    remaining = {entry["source"] for entry in ingested_client.get("/api/documents").json()["documents"]}
    assert "security.txt" not in remaining


def test_reset_clears_the_index(ingested_client: TestClient) -> None:
    """Reset empties the document list."""
    assert ingested_client.post("/api/documents/reset").status_code == 200
    assert ingested_client.get("/api/documents").json()["documents"] == []


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------
def test_chat_returns_a_cited_answer(ingested_client: TestClient) -> None:
    """A question over an indexed corpus returns an answer with citations."""
    response = ingested_client.post("/api/chat", json={"question": QUESTION})

    assert response.status_code == 200
    payload = response.json()

    assert payload["answer"]
    assert payload["citations"], "expected at least one citation"
    assert payload["grounded"] is True
    assert payload["cached"] is False
    assert payload["model"] == "fake-model"

    citation = payload["citations"][0]
    assert citation["source"]
    assert citation["snippet"]
    assert citation["index"] >= 1
    assert payload["retrieved"]


def test_chat_returns_retrieved_sources_even_when_uncited(ingested_client: TestClient) -> None:
    """Every passage considered is reported, so the UI can show what was seen."""
    payload = ingested_client.post("/api/chat", json={"question": "How do I rotate an API token?"}).json()

    assert payload["retrieved"]
    assert all("source" in entry for entry in payload["retrieved"])


def test_chat_without_documents_short_circuits(client: TestClient) -> None:
    """With nothing indexed the model is not called and a refusal is returned."""
    payload = client.post("/api/chat", json={"question": QUESTION}).json()

    assert payload["insufficient_context"] is True
    assert payload["answer"].startswith("I don't know")
    assert payload["citations"] == []
    assert payload["reason"] == "no_documents_retrieved"


def test_chat_uses_the_semantic_cache_on_a_repeat(ingested_client: TestClient) -> None:
    """A repeated question is served from cache and says so."""
    first = ingested_client.post("/api/chat", json={"question": QUESTION}).json()
    second = ingested_client.post("/api/chat", json={"question": QUESTION}).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["cache_similarity"] == pytest.approx(1.0)
    assert second["answer"] == first["answer"]


def test_chat_can_bypass_the_cache(ingested_client: TestClient) -> None:
    """A client can opt out of cached answers."""
    ingested_client.post("/api/chat", json={"question": QUESTION})

    payload = ingested_client.post(
        "/api/chat", json={"question": QUESTION, "use_cache": False}
    ).json()

    assert payload["cached"] is False


def test_chat_reports_timings_and_token_statistics(ingested_client: TestClient) -> None:
    """Per-stage timings and token counts are returned for diagnostics."""
    payload = ingested_client.post("/api/chat", json={"question": QUESTION}).json()

    assert "total_ms" in payload["timings"]
    assert "generation_ms" in payload["timings"]
    assert payload["stats"]["completion_tokens"] > 0


def test_chat_respects_top_k(ingested_client: TestClient) -> None:
    """The retrieved passage count honours the request."""
    payload = ingested_client.post(
        "/api/chat", json={"question": QUESTION, "top_k": 1, "use_cache": False}
    ).json()

    assert len(payload["retrieved"]) == 1


def test_chat_rejects_an_empty_question(ingested_client: TestClient) -> None:
    """A blank question fails validation."""
    assert ingested_client.post("/api/chat", json={"question": ""}).status_code == 422


def test_chat_rejects_an_excessive_top_k(ingested_client: TestClient) -> None:
    """The retrieval depth is bounded so one caller cannot request everything."""
    response = ingested_client.post("/api/chat", json={"question": QUESTION, "top_k": 500})

    assert response.status_code == 422


def test_chat_returns_503_when_ollama_is_not_configured(unconfigured_settings: Settings) -> None:
    """An unconfigured deployment explains the problem instead of failing oddly."""
    container = ServiceContainer(
        settings=unconfigured_settings,
        embedder=HashingEmbedder(dimension=64),
        reranker=NoOpReranker(),
        client=OllamaClient(settings=unconfigured_settings),
    )
    application = create_app(settings=unconfigured_settings, container=container)

    with TestClient(application) as test_client:
        response = test_client.post("/api/chat", json={"question": QUESTION})

    assert response.status_code == 503
    assert "OLLAMA_BASE_URL" in json.dumps(response.json())


def test_chat_stream_emits_sse_frames(ingested_client: TestClient) -> None:
    """The streaming endpoint yields frames in the documented order."""
    response = ingested_client.post("/api/chat/stream", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = [
        line[6:]
        for line in response.text.splitlines()
        if line.startswith("data: ") and line.strip() != "data: [DONE]"
    ]
    events = [json.loads(frame) for frame in frames]
    kinds = [event["type"] for event in events]

    # Frames are: timings, trace, sources, tokens..., done.
    assert kinds[0] == "timings"
    assert kinds[1] == "trace"
    assert kinds[2] == "sources"
    assert "token" in kinds
    assert kinds[-1] == "done"

    # Sources are announced before any token, so the UI can render them early.
    assert kinds.index("sources") < kinds.index("token")

    assert events[-1]["answer"]
    assert events[-1]["citations"]
    assert response.text.rstrip().endswith("[DONE]")


def test_chat_stream_without_documents_reports_insufficient_context(client: TestClient) -> None:
    """The stream terminates cleanly even with an empty index."""
    response = client.post("/api/chat/stream", json={"question": QUESTION})

    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line.strip() != "data: [DONE]"
    ]

    assert frames[-1]["type"] == "done"
    assert frames[-1]["insufficient_context"] is True


def test_streaming_does_not_populate_the_cache(ingested_client: TestClient) -> None:
    """Streaming bypasses the cache entirely, so a later blocking call is a miss."""
    ingested_client.post("/api/chat/stream", json={"question": QUESTION})

    payload = ingested_client.post("/api/chat", json={"question": QUESTION}).json()

    assert payload["cached"] is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
def test_rate_limit_returns_429_once_exceeded(container: ServiceContainer) -> None:
    """Exceeding the chat limit produces a structured 429."""
    limited = container.settings.model_copy(
        update={"rate_limit_enabled": True, "rate_limit_chat": "2/minute"}
    )
    limited_container = ServiceContainer(
        settings=limited,
        embedder=container.embedder,
        reranker=NoOpReranker(),
        client=container.client,
    )
    limited_container.pipeline.ingest_documents(
        [
            Document(
                text="Rate limiting test content about notices and limits.",
                source="limits.txt",
            )
        ]
    )

    application = create_app(settings=limited, container=limited_container)
    try:
        with TestClient(application) as test_client:
            statuses = [
                test_client.post("/api/chat", json={"question": QUESTION}).status_code
                for _ in range(3)
            ]
    finally:
        configure_limiter(container.settings)
        limiter.reset()

    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429


def test_rate_limiting_can_be_disabled(container: ServiceContainer) -> None:
    """With limiting off, repeated requests are never rejected."""
    assert container.settings.rate_limit_enabled is False

    application = create_app(settings=container.settings, container=container)
    with TestClient(application) as test_client:
        statuses = {test_client.get("/api/health").status_code for _ in range(5)}

    assert statuses == {200}
