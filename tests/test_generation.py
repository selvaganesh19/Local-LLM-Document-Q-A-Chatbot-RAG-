"""Tests for prompt assembly, citation parsing and grounded generation."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.generation.generator import (
    AnswerGenerator,
    Citation,
    looks_like_refusal,
    make_snippet,
)
from app.generation.ollama_client import (
    ChatMessage,
    ChatResponse,
    FakeOllamaClient,
    GenerationStats,
    OllamaClient,
    OllamaNotConfiguredError,
)
from app.generation.prompts import (
    INSUFFICIENT_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_context_block,
    build_messages,
    build_user_prompt,
    context_budget_chars,
    format_source_label,
)
from app.retrieval.hybrid import RetrievedChunk


def _chunk(chunk_id: str, text: str, source: str = "doc.txt", page: int | None = None) -> RetrievedChunk:
    """Build a retrieved chunk for testing."""
    metadata: dict = {"source": source}
    if page is not None:
        metadata["page"] = page
    return RetrievedChunk(chunk_id=chunk_id, text=text, metadata=metadata, relevance=0.8)


class EchoClient(OllamaClient):
    """Client that returns a canned answer, for exercising the parser."""

    def __init__(self, answer: str) -> None:
        """Store the answer this client will return."""
        super().__init__(base_url="http://fake", model="echo-model")
        self._answer = answer
        self.prompts: list[str] = []

    @property
    def is_configured(self) -> bool:
        """Always ready, like the fake client."""
        return True

    def _ensure_configured(self) -> None:
        """No-op override."""

    async def chat(self, messages, temperature=None, num_ctx=None) -> ChatResponse:  # type: ignore[override]
        """Record the prompt and return the canned answer."""
        self.prompts.append(messages[-1].content)
        return ChatResponse(
            content=self._answer,
            stats=GenerationStats(model=self.model, prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def test_context_block_numbers_passages() -> None:
    """Passages are numbered in rank order so citations are unambiguous."""
    block = build_context_block([_chunk("a", "First body."), _chunk("b", "Second body.")])

    assert block.startswith("[1] doc.txt")
    assert "[2] doc.txt" in block
    assert "First body." in block and "Second body." in block


def test_context_block_includes_page_numbers() -> None:
    """Paginated sources carry their page into the label."""
    block = build_context_block([_chunk("a", "Body.", source="report.pdf", page=7)])

    assert "report.pdf, page 7" in block


def test_context_block_respects_the_budget() -> None:
    """Passages beyond the character budget are dropped."""
    chunks = [_chunk(f"c{index}", "x" * 400) for index in range(5)]

    block = build_context_block(chunks, max_chars=450)

    assert "[1]" in block
    assert "[5]" not in block


def test_context_block_always_includes_the_first_passage() -> None:
    """An oversized first passage is still supplied - something beats nothing."""
    block = build_context_block([_chunk("a", "y" * 5000)], max_chars=100)

    assert "[1]" in block


def test_context_block_is_empty_without_chunks() -> None:
    """No chunks means no context, which the caller handles explicitly."""
    assert build_context_block([]) == ""


def test_user_prompt_states_when_no_context_was_found() -> None:
    """The model is told plainly that retrieval returned nothing."""
    prompt = build_user_prompt("What is the notice period?", "")

    assert "No context passages" in prompt
    assert "What is the notice period?" in prompt


def test_build_messages_includes_the_system_prompt() -> None:
    """Every request carries the grounding rules and the citation instruction."""
    messages, prompt, block = build_messages("Q?", [_chunk("a", "Body.")])

    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content == SYSTEM_PROMPT
    assert "Body." in block
    assert "[1]" in prompt


def test_system_prompt_contains_the_refusal_sentence() -> None:
    """The refusal instruction is part of the contract, not an afterthought."""
    assert INSUFFICIENT_CONTEXT_ANSWER in SYSTEM_PROMPT


def test_context_budget_scales_with_the_window() -> None:
    """A larger context window buys a larger passage budget."""
    assert context_budget_chars(8192) > context_budget_chars(2048)
    assert context_budget_chars(0) == 0


def test_format_source_label_falls_back_to_unknown() -> None:
    """A chunk without provenance still produces a usable label."""
    assert format_source_label(object()) == "unknown"


# ---------------------------------------------------------------------------
# Snippets and refusal detection
# ---------------------------------------------------------------------------
def test_make_snippet_collapses_whitespace() -> None:
    """Newlines and runs of spaces are normalised for display."""
    assert make_snippet("a\n\n  b\t c") == "a b c"


def test_make_snippet_truncates_long_text() -> None:
    """Long passages are cut with an ellipsis."""
    snippet = make_snippet("word " * 200, length=50)

    assert len(snippet) <= 53
    assert snippet.endswith("...")


def test_looks_like_refusal_matches_the_fixed_sentence() -> None:
    """The canonical refusal is recognised."""
    assert looks_like_refusal(INSUFFICIENT_CONTEXT_ANSWER) is True
    assert looks_like_refusal(f"  {INSUFFICIENT_CONTEXT_ANSWER}  ") is True


def test_looks_like_refusal_rejects_real_answers() -> None:
    """A genuine answer is not mistaken for a refusal."""
    assert looks_like_refusal("The notice period is 90 days [1].") is False
    assert looks_like_refusal("") is False


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------
def test_parse_citations_maps_markers_to_chunks(settings: Settings) -> None:
    """Citation numbers resolve to the passage they refer to."""
    generator = AnswerGenerator(client=FakeOllamaClient(), settings=settings)
    chunks = [_chunk("a", "Alpha body.", source="alpha.txt"), _chunk("b", "Beta body.", source="beta.txt")]

    citations = generator.parse_citations("Alpha is true [1] and so is beta [2].", chunks)

    assert [citation.index for citation in citations] == [1, 2]
    assert citations[0].chunk_id == "a"
    assert citations[0].source == "alpha.txt"
    assert isinstance(citations[0], Citation)


def test_parse_citations_drops_out_of_range_markers(settings: Settings) -> None:
    """A citation the model invented is discarded, never trusted."""
    generator = AnswerGenerator(client=FakeOllamaClient(), settings=settings)

    citations = generator.parse_citations("See [1] and also [7].", [_chunk("a", "Alpha body.")])

    assert [citation.index for citation in citations] == [1]


def test_parse_citations_deduplicates_and_keeps_first_order(settings: Settings) -> None:
    """Repeated markers collapse to one citation, in order of first use."""
    generator = AnswerGenerator(client=FakeOllamaClient(), settings=settings)
    chunks = [_chunk("a", "Alpha."), _chunk("b", "Beta."), _chunk("c", "Gamma.")]

    citations = generator.parse_citations("Beta [2] then alpha [1] then beta again [2].", chunks)

    assert [citation.index for citation in citations] == [2, 1]


def test_parse_citations_returns_nothing_without_markers(settings: Settings) -> None:
    """An uncited answer produces no citations."""
    generator = AnswerGenerator(client=FakeOllamaClient(), settings=settings)

    assert generator.parse_citations("An answer with no markers.", [_chunk("a", "Alpha.")]) == []


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_returns_a_cited_answer(settings: Settings) -> None:
    """The fake model cites the passages it was given."""
    generator = AnswerGenerator(client=FakeOllamaClient(), settings=settings)
    chunks = [_chunk("a", "Alpha body."), _chunk("b", "Beta body.")]

    result = await generator.generate("What is alpha?", chunks)

    assert result.citations
    assert result.uncited is False
    assert result.grounded is True
    assert result.retrieved[0]["index"] == 1
    assert result.stats.completion_tokens > 0


@pytest.mark.asyncio
async def test_generate_flags_an_uncited_answer(settings: Settings) -> None:
    """An answer with no markers is reported as ungrounded."""
    generator = AnswerGenerator(client=EchoClient("Alpha is definitely true."), settings=settings)

    result = await generator.generate("What is alpha?", [_chunk("a", "Alpha body.")])

    assert result.citations == []
    assert result.uncited is True
    assert result.grounded is False


@pytest.mark.asyncio
async def test_generate_detects_a_refusal(settings: Settings) -> None:
    """A refusal is recognised and not treated as a citation failure."""
    generator = AnswerGenerator(client=EchoClient(INSUFFICIENT_CONTEXT_ANSWER), settings=settings)

    result = await generator.generate("What is the capital of Portugal?", [_chunk("a", "Alpha body.")])

    assert result.insufficient_context is True
    assert result.uncited is False
    assert result.grounded is True


@pytest.mark.asyncio
async def test_generate_records_the_prompt(settings: Settings) -> None:
    """The exact prompt is retained, which is what makes a trace auditable."""
    client = EchoClient("Alpha is true [1].")
    generator = AnswerGenerator(client=client, settings=settings)

    result = await generator.generate("What is alpha?", [_chunk("a", "Alpha body.")])

    assert "Alpha body." in result.prompt
    assert "What is alpha?" in result.prompt
    assert result.context_block.startswith("[1]")


@pytest.mark.asyncio
async def test_stream_emits_sources_tokens_and_a_terminal_frame(settings: Settings) -> None:
    """The stream starts with sources, streams tokens, and ends with citations."""
    generator = AnswerGenerator(client=FakeOllamaClient(), settings=settings)
    chunks = [_chunk("a", "Alpha body."), _chunk("b", "Beta body.")]

    events = [event async for event in generator.stream("What is alpha?", chunks)]
    kinds = [event["type"] for event in events]

    assert kinds[0] == "sources"
    assert "token" in kinds
    assert kinds[-1] == "done"

    done = events[-1]
    assert done["answer"]
    assert done["citations"]
    assert done["retrieved"][0]["chunk_id"] == "a"


# ---------------------------------------------------------------------------
# Client behaviour
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_real_client_without_configuration_raises() -> None:
    """A blank base URL produces an actionable error, not a connection attempt."""
    settings = Settings(_env_file=None, ollama_base_url="", ollama_model="")
    client = OllamaClient(settings=settings)

    assert client.is_configured is False
    with pytest.raises(OllamaNotConfiguredError):
        await client.chat([ChatMessage(role="user", content="hello")])


@pytest.mark.asyncio
async def test_health_without_configuration_reports_the_reason() -> None:
    """Health output explains which setting is missing."""
    settings = Settings(_env_file=None, ollama_base_url="", ollama_model="")
    report = await OllamaClient(settings=settings).health()

    assert report["reachable"] is False
    assert report["configured"] is False
    assert "OLLAMA_BASE_URL" in str(report["error"])


@pytest.mark.asyncio
async def test_fake_client_reports_healthy() -> None:
    """The fake client never touches the network."""
    report = await FakeOllamaClient().health()

    assert report["reachable"] is True
    assert report["model_available"] is True


def test_build_body_carries_model_and_options() -> None:
    """The request body includes the model tag and sampling options."""
    settings = Settings(_env_file=None, ollama_model="qwen3:8b", ollama_temperature=0.2, ollama_num_ctx=4096)
    body = OllamaClient(settings=settings)._build_body(
        [ChatMessage(role="user", content="hi")], temperature=None, num_ctx=None, stream=False
    )

    assert body["model"] == "qwen3:8b"
    assert body["options"]["temperature"] == pytest.approx(0.2)
    assert body["options"]["num_ctx"] == 4096
    assert body["stream"] is False
