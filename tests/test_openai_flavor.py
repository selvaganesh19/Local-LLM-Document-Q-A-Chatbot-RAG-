"""Tests for the OpenAI-compatible wire protocol.

The client speaks either Ollama's native API or the OpenAI-compatible one used
by vLLM, llama.cpp's server and similar backends.  These tests pin the
selection rule and the request/response translation for the second flavor,
using a mock transport so nothing touches the network.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from app.config import Settings
from app.generation.ollama_client import (
    ChatMessage,
    OllamaClient,
    _ReasoningStreamFilter,
    _stats_from_usage,
    strip_reasoning,
)

OPENAI_URL = "http://localhost:8000/v1"
OLLAMA_URL = "http://localhost:11434"


def _settings(**overrides: Any) -> Settings:
    """Build a hermetic settings object for client tests."""
    values: Dict[str, Any] = {
        "ollama_base_url": OPENAI_URL,
        "ollama_model": "qwen3.8-27b",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _handler(routes: Dict[str, Any]) -> httpx.MockTransport:
    """Route mock requests to canned JSON or SSE bodies by path."""

    def respond(request: httpx.Request) -> httpx.Response:
        payload = routes.get(request.url.path)
        if payload is None:
            return httpx.Response(404, json={"error": "not found"})
        if isinstance(payload, str):
            return httpx.Response(200, text=payload, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(respond)


def _wire(client: OllamaClient, transport: httpx.MockTransport) -> OllamaClient:
    """Attach a mock transport to a client, replacing its HTTP pool.

    The replacement mirrors what :meth:`OllamaClient._http` builds, so the
    default headers a real request would carry are still exercised.
    """
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=transport,
        headers=client._default_headers(),
    )
    return client


def _sse(*chunks: Dict[str, Any]) -> str:
    """Render OpenAI-style server-sent events, terminated by ``[DONE]``."""
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Flavor selection
# ---------------------------------------------------------------------------
def test_a_v1_suffix_selects_the_openai_flavor() -> None:
    """The /v1 convention is what distinguishes the two protocols."""
    assert OllamaClient(settings=_settings()).flavor == "openai"


def test_a_bare_host_selects_the_ollama_flavor() -> None:
    """A URL without /v1 keeps the native Ollama routes."""
    assert OllamaClient(settings=_settings(ollama_base_url=OLLAMA_URL)).flavor == "ollama"


def test_an_explicit_flavor_overrides_detection() -> None:
    """LLM_API_FLAVOR wins when auto-detection would guess wrong."""
    settings = _settings(llm_api_flavor="ollama")

    assert settings.resolved_api_flavor == "ollama"
    assert OllamaClient(settings=settings).flavor == "ollama"


def test_an_explicit_openai_flavor_works_without_the_v1_suffix() -> None:
    """A proxy that hides /v1 can still be driven explicitly."""
    settings = _settings(ollama_base_url="http://localhost:8000", llm_api_flavor="openai")
    client = OllamaClient(settings=settings)

    assert client.flavor == "openai"
    assert client._build_openai_body([ChatMessage("user", "hi")], None, stream=False)


def test_trailing_slashes_do_not_break_detection() -> None:
    """A copy-pasted URL with a trailing slash still resolves correctly."""
    settings = _settings(ollama_base_url="http://localhost:8000/v1/")

    assert settings.resolved_api_flavor == "openai"
    assert OllamaClient(settings=settings).base_url == "http://localhost:8000/v1"


def test_a_base_url_override_redetects_the_flavor() -> None:
    """An ad-hoc client pointed at another server is not stuck on the config."""
    client = OllamaClient(base_url=OLLAMA_URL, model="llama3.2", settings=_settings())

    assert client.flavor == "ollama"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
def test_openai_body_puts_sampling_options_at_the_top_level() -> None:
    """OpenAI-compatible servers reject Ollama's nested `options` object."""
    settings = _settings(ollama_temperature=0.2)
    body = OllamaClient(settings=settings)._build_openai_body(
        [ChatMessage("system", "be brief"), ChatMessage("user", "hi")],
        temperature=None,
        stream=False,
    )

    assert body["model"] == "qwen3.8-27b"
    assert body["temperature"] == pytest.approx(0.2)
    assert "options" not in body
    assert body["messages"][0] == {"role": "system", "content": "be brief"}


def test_openai_streaming_requests_usage_in_the_final_chunk() -> None:
    """Without stream_options the server reports no token counts at all."""
    body = OllamaClient(settings=_settings())._build_openai_body(
        [ChatMessage("user", "hi")], temperature=None, stream=True
    )

    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Stream parsing
# ---------------------------------------------------------------------------
def test_openai_delta_is_translated_into_a_token_event() -> None:
    """`choices[0].delta.content` is the streamed answer fragment."""
    line = json.dumps({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]})

    events, reason = OllamaClient._parse_openai_line(f"data: {line}", started=0.0)

    assert events == [{"type": "token", "content": "Hello"}]
    assert reason == ""


def test_reasoning_traces_are_not_shown_as_answers() -> None:
    """Reasoning models emit chain-of-thought deltas that must be withheld."""
    line = json.dumps(
        {"choices": [{"delta": {"reasoning_content": "Let me think about [1]..."}, "finish_reason": None}]}
    )

    events, _ = OllamaClient._parse_openai_line(f"data: {line}", started=0.0)

    assert events == []


def test_the_finish_reason_is_reported_for_the_caller_to_remember() -> None:
    """vLLM sends the stop reason on the last content chunk, not the usage one."""
    line = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})

    events, reason = OllamaClient._parse_openai_line(f"data: {line}", started=0.0)

    assert events == []
    assert reason == "stop"


def test_the_usage_chunk_closes_the_stream() -> None:
    """The final chunk carries usage and is turned into a `done` event."""
    line = json.dumps(
        {
            "model": "qwen3.8-27b",
            "choices": [],
            "usage": {"prompt_tokens": 40, "completion_tokens": 7},
        }
    )

    events, _ = OllamaClient._parse_openai_line(f"data: {line}", started=0.0)

    assert len(events) == 1
    assert events[0]["type"] == "done"
    assert events[0]["stats"]["prompt_tokens"] == 40
    assert events[0]["stats"]["completion_tokens"] == 7


def test_the_done_sentinel_and_noise_are_ignored() -> None:
    """`[DONE]`, keep-alive comments and malformed JSON yield no events."""
    assert OllamaClient._parse_openai_line("data: [DONE]", started=0.0) == ([], "")
    assert OllamaClient._parse_openai_line(": keep-alive", started=0.0) == ([], "")
    assert OllamaClient._parse_openai_line("data: {not json", started=0.0) == ([], "")


def test_the_ollama_parser_is_unchanged() -> None:
    """The native protocol still parses its own frame shape."""
    line = json.dumps({"message": {"content": "Hi"}, "done": True, "done_reason": "stop"})

    events, reason = OllamaClient._parse_ollama_line(line)

    assert [event["type"] for event in events] == ["token", "done"]
    assert events[-1]["finish_reason"] == "stop"
    assert reason == "stop"


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------
def test_usage_is_mapped_onto_the_stats_object() -> None:
    """OpenAI reports usage but no timings, so the caller supplies elapsed time."""
    stats = _stats_from_usage(
        {"model": "qwen3.8-27b", "usage": {"prompt_tokens": 10, "completion_tokens": 200}},
        fallback_model="fallback",
        elapsed_ms=2000.0,
    )

    assert stats.model == "qwen3.8-27b"
    assert stats.tokens_per_second == pytest.approx(100.0)


def test_usage_mapping_tolerates_a_missing_block() -> None:
    """Some servers omit `usage`; the stats must still be well-formed."""
    stats = _stats_from_usage({}, fallback_model="fallback", elapsed_ms=0.0)

    assert stats.model == "fallback"
    assert stats.tokens_per_second == 0.0


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
async def test_chat_reads_the_openai_response_shape() -> None:
    """A completion is read from `choices[0].message.content`."""
    transport = _handler(
        {
            "/v1/chat/completions": {
                "model": "qwen3.8-27b",
                "choices": [
                    {"message": {"role": "assistant", "content": "Ninety days [1]."}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            }
        }
    )
    client = _wire(OllamaClient(settings=_settings()), transport)

    response = await client.chat([ChatMessage("user", "How long?")])

    assert response.content == "Ninety days [1]."
    assert response.finish_reason == "stop"
    assert response.stats.completion_tokens == 5
    await client.aclose()


async def test_streaming_yields_tokens_then_done() -> None:
    """The SSE body is translated into the same event shape as Ollama's."""
    body = _sse(
        {"model": "qwen3.8-27b", "choices": [{"delta": {"content": "Ninety "}, "finish_reason": None}]},
        {"model": "qwen3.8-27b", "choices": [{"delta": {"content": "days [1]."}, "finish_reason": None}]},
        {
            "model": "qwen3.8-27b",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        },
    )
    client = _wire(OllamaClient(settings=_settings()), _handler({"/v1/chat/completions": body}))

    events = [event async for event in client.chat_stream([ChatMessage("user", "How long?")])]

    assert [event["type"] for event in events] == ["token", "token", "done"]
    assert "".join(event["content"] for event in events if event["type"] == "token") == "Ninety days [1]."
    # The reason arrived on the previous chunk; the done frame still carries it.
    assert events[-1]["finish_reason"] == "stop"
    await client.aclose()


async def test_health_lists_models_from_the_openai_endpoint() -> None:
    """Health checks `/models` and matches on the `data[].id` field."""
    transport = _handler(
        {
            "/v1/models": {
                "object": "list",
                "data": [{"id": "qwen3.8-27b", "object": "model"}],
            }
        }
    )
    client = _wire(OllamaClient(settings=_settings()), transport)

    report = await client.health()

    assert report["reachable"] is True
    assert report["flavor"] == "openai"
    assert report["model_available"] is True
    await client.aclose()


async def test_health_explains_an_unknown_model_id() -> None:
    """A model the server does not serve is reported with a fixable message."""
    transport = _handler({"/v1/models": {"object": "list", "data": [{"id": "other-model"}]}})
    client = _wire(OllamaClient(settings=_settings()), transport)

    report = await client.health()

    assert report["reachable"] is True
    assert report["model_available"] is False
    assert "qwen3.8-27b" in report["error"]
    await client.aclose()


async def test_an_api_key_becomes_a_bearer_header() -> None:
    """A hosted endpoint that checks credentials gets the configured token."""
    seen: List[httpx.Headers] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json={"object": "list", "data": []})

    client = OllamaClient(settings=_settings(llm_api_key="secret-token"))
    _wire(client, httpx.MockTransport(respond))

    await client.health()
    await client.aclose()

    assert seen[0]["authorization"] == "Bearer secret-token"


async def test_no_authorization_header_without_a_key() -> None:
    """Local servers are not sent a spurious empty bearer token."""
    seen: List[httpx.Headers] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(200, json={"object": "list", "data": []})

    client = _wire(OllamaClient(settings=_settings()), httpx.MockTransport(respond))

    await client.health()
    await client.aclose()

    assert "authorization" not in seen[0]


async def test_a_server_error_is_surfaced_with_its_status() -> None:
    """A 5xx from the model server becomes an actionable OllamaError."""
    from app.generation.ollama_client import OllamaError

    transport = httpx.MockTransport(lambda request: httpx.Response(503, text="model loading"))
    client = _wire(OllamaClient(settings=_settings()), transport)

    with pytest.raises(OllamaError) as excinfo:
        await client.chat([ChatMessage("user", "hi")])

    assert "503" in str(excinfo.value)
    await client.aclose()


# ---------------------------------------------------------------------------
# Reasoning models
# ---------------------------------------------------------------------------
def test_thinking_is_disabled_on_the_openai_flavor_by_default() -> None:
    """A reasoning model is asked to answer directly instead of deliberating."""
    body = OllamaClient(settings=_settings())._build_openai_body(
        [ChatMessage("user", "hi")], temperature=None, stream=False
    )

    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_can_be_left_enabled() -> None:
    """LLM_DISABLE_THINKING=false keeps the model deliberating."""
    body = OllamaClient(settings=_settings(llm_disable_thinking=False))._build_openai_body(
        [ChatMessage("user", "hi")], temperature=None, stream=False
    )

    assert "chat_template_kwargs" not in body


def test_the_ollama_flavor_never_sends_the_thinking_option() -> None:
    """The option is vLLM-shaped; Ollama's native API would reject it."""
    client = OllamaClient(settings=_settings(ollama_base_url=OLLAMA_URL))

    assert client._thinking_off_requested is False


def test_chain_of_thought_is_stripped_from_an_answer() -> None:
    """Only the answer survives a leaked `<think>` block."""
    raw = "We need answer the question.\nThe passage says 90 days.\n</think>\n\nThe notice period is 90 days [1]."

    assert strip_reasoning(raw) == "The notice period is 90 days [1]."


def test_stripping_keeps_an_answer_that_has_no_reasoning() -> None:
    """A model that does not deliberate is passed through unchanged."""
    assert strip_reasoning("Ninety days [1].") == "Ninety days [1]."


def test_only_the_final_reasoning_tag_is_honoured() -> None:
    """A model that quotes the tag in its reasoning must not be truncated early."""
    raw = "The user asked about </think>. Answering now.\n</think>\n\nNinety days [1]."

    assert strip_reasoning(raw) == "Ninety days [1]."


def test_the_stream_filter_withholds_reasoning_tokens() -> None:
    """Nothing is shown until the closing tag proves the answer has started."""
    filter_ = _ReasoningStreamFilter()

    assert filter_.feed("We need answer ") == ""
    assert filter_.feed("the question.</think>\n\nNinety ") == "Ninety "
    assert filter_.feed("days [1].") == "days [1]."


def test_the_stream_filter_releases_when_no_tag_ever_arrives() -> None:
    """A model that never emits a tag must not have its output swallowed."""
    filter_ = _ReasoningStreamFilter()

    assert filter_.feed("x" * 5000) == "x" * 5000
    assert filter_.feed(" more") == " more"


def test_the_stream_filter_flushes_whatever_is_buffered() -> None:
    """At end of stream the buffer is released rather than dropped."""
    filter_ = _ReasoningStreamFilter()

    assert filter_.feed("partial answer") == ""
    assert filter_.flush() == "partial answer"


async def test_a_rejected_thinking_option_is_dropped_and_retried() -> None:
    """Strict OpenAI servers 400 on unknown fields; the client must recover."""
    bodies: List[Dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "chat_template_kwargs" in body:
            return httpx.Response(400, json={"error": "unknown field chat_template_kwargs"})
        return httpx.Response(
            200,
            json={
                "model": "qwen3.8-27b",
                "choices": [{"message": {"content": "Ninety days [1]."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4},
            },
        )

    client = _wire(OllamaClient(settings=_settings()), httpx.MockTransport(respond))

    response = await client.chat([ChatMessage("user", "hi")])

    assert response.content == "Ninety days [1]."
    assert len(bodies) == 2
    assert client._thinking_off_requested is False
    await client.aclose()


async def test_an_openai_answer_is_stripped_of_reasoning() -> None:
    """Servers that strip the opening tag still leave a closing one behind."""
    transport = _handler(
        {
            "/v1/chat/completions": {
                "model": "qwen3.8-27b",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Thinking...\n</think>\n\nNinety days [1]."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            }
        }
    )
    client = _wire(OllamaClient(settings=_settings(llm_disable_thinking=False)), transport)

    response = await client.chat([ChatMessage("user", "How long?")])

    assert response.content == "Ninety days [1]."
    await client.aclose()
