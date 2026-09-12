"""Grounded answer generation with source citations.

The generator is the only place that knows how retrieved chunks become an
answer.  It assembles the numbered context, calls the local model, and then
parses the inline ``[n]`` markers back into structured citations that the API
and UI can render.

A citation the model invented (``[7]`` when only five passages were supplied)
is discarded rather than trusted, and an answer that cites nothing is flagged
``uncited`` so callers can surface an ungrounded response instead of pretending
it is verified.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from app.config import Settings, get_settings
from app.generation.ollama_client import (
    ChatMessage,
    ChatResponse,
    GenerationStats,
    OllamaClient,
)
from app.generation.prompts import (
    INSUFFICIENT_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_messages,
    context_budget_chars,
    format_source_label,
)
from app.observability import tracing

logger = logging.getLogger(__name__)

#: Inline citation marker produced by the model, e.g. ``[3]``.
_CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")

#: Length of the excerpt shown in the UI's source list.
SNIPPET_LENGTH = 320

#: Whitespace runs collapsed when building snippets.
_WHITESPACE = re.compile(r"\s+")


@dataclass
class Citation:
    """A passage the answer explicitly referenced.

    Attributes:
        index: The ``[n]`` number used in the answer.
        chunk_id: Identifier of the underlying chunk.
        source: Source document name.
        page: Page number, when the source is paginated.
        label: Human-readable provenance label.
        snippet: Short excerpt for display.
        relevance: Retrieval/rerank relevance in ``[0, 1]``.
        text: Full chunk text.
        metadata: Full provenance mapping.
    """

    index: int
    chunk_id: str
    source: str
    page: Optional[int]
    label: str
    snippet: str
    relevance: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for API responses."""
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "label": self.label,
            "snippet": self.snippet,
            "relevance": round(self.relevance, 4),
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass
class GeneratedAnswer:
    """A generated answer plus everything needed to audit it.

    Attributes:
        answer: The model's answer text, verbatim.
        citations: Only those passages actually cited, in citation order.
        retrieved: Every passage that was offered to the model.
        stats: Token accounting and timings from Ollama.
        prompt: The exact user turn sent to the model.
        system_prompt: The system prompt in force.
        context_block: The numbered passages as rendered.
        insufficient_context: The model declined for lack of context.
        uncited: The answer contains no valid citation markers.
        latency_ms: End-to-end generation latency.
    """

    answer: str = ""
    citations: List[Citation] = field(default_factory=list)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    stats: GenerationStats = field(default_factory=GenerationStats)
    prompt: str = ""
    system_prompt: str = SYSTEM_PROMPT
    context_block: str = ""
    insufficient_context: bool = False
    uncited: bool = False
    latency_ms: float = 0.0

    @property
    def cited_indices(self) -> List[int]:
        """Citation numbers referenced by the answer."""
        return [citation.index for citation in self.citations]

    @property
    def grounded(self) -> bool:
        """True when the answer is either cited or an explicit refusal."""
        return self.insufficient_context or not self.uncited

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for API responses."""
        return {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "retrieved": list(self.retrieved),
            "stats": self.stats.to_dict(),
            "insufficient_context": self.insufficient_context,
            "uncited": self.uncited,
            "grounded": self.grounded,
            "latency_ms": round(self.latency_ms, 2),
        }


def make_snippet(text: str, length: int = SNIPPET_LENGTH) -> str:
    """Collapse whitespace and truncate ``text`` for display."""
    collapsed = _WHITESPACE.sub(" ", text).strip()
    if len(collapsed) <= length:
        return collapsed
    return collapsed[:length].rstrip() + "..."


def looks_like_refusal(answer: str) -> bool:
    """Detect the fixed refusal sentence, tolerating minor formatting drift."""
    normalised = _WHITESPACE.sub(" ", answer).strip().lower()
    if not normalised:
        return False
    return normalised.startswith(INSUFFICIENT_CONTEXT_ANSWER.lower()[:24])


class AnswerGenerator:
    """Turns retrieved chunks into a cited answer using the local model."""

    def __init__(self, client: OllamaClient, settings: Optional[Settings] = None) -> None:
        """Bind the generator to an Ollama client.

        Args:
            client: Chat client (real or fake).
            settings: Configuration override; defaults to process settings.
        """
        self._client = client
        self._settings = settings or get_settings()

    @property
    def model_name(self) -> str:
        """Model tag used for generation."""
        return self._client.model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _retrieved_payload(self, chunks: Sequence[Any]) -> List[Dict[str, Any]]:
        """Describe every passage offered to the model, in rank order."""
        payload: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            page = getattr(chunk, "page", None)
            payload.append(
                {
                    "index": index,
                    "chunk_id": str(getattr(chunk, "chunk_id", f"chunk-{index}")),
                    "source": str(getattr(chunk, "source", metadata.get("source", "unknown"))),
                    "page": page,
                    "label": format_source_label(chunk),
                    "snippet": make_snippet(str(getattr(chunk, "text", ""))),
                    "relevance": round(float(getattr(chunk, "relevance", 0.0) or 0.0), 4),
                    "dense_score": getattr(chunk, "dense_score", None),
                    "sparse_score": getattr(chunk, "sparse_score", None),
                    "fusion_score": round(float(getattr(chunk, "fusion_score", 0.0) or 0.0), 6),
                    "rerank_score": getattr(chunk, "rerank_score", None),
                    "metadata": metadata,
                }
            )
        return payload

    def parse_citations(self, answer: str, chunks: Sequence[Any]) -> List[Citation]:
        """Map the answer's ``[n]`` markers back onto the supplied chunks.

        Args:
            answer: The model's answer text.
            chunks: Passages in the same order they were numbered.

        Returns:
            Citations in order of first appearance, de-duplicated.  Markers
            outside ``1..len(chunks)`` are dropped.
        """
        found: List[Citation] = []
        seen: set[int] = set()

        for match in _CITATION_PATTERN.finditer(answer):
            index = int(match.group(1))
            if index in seen or not 1 <= index <= len(chunks):
                continue
            seen.add(index)

            chunk = chunks[index - 1]
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            page = getattr(chunk, "page", None)
            text = str(getattr(chunk, "text", ""))
            found.append(
                Citation(
                    index=index,
                    chunk_id=str(getattr(chunk, "chunk_id", f"chunk-{index}")),
                    source=str(getattr(chunk, "source", metadata.get("source", "unknown"))),
                    page=page,
                    label=format_source_label(chunk),
                    snippet=make_snippet(text),
                    relevance=float(getattr(chunk, "relevance", 0.0) or 0.0),
                    text=text,
                    metadata=metadata,
                )
            )

        return found

    def _finalise(
        self,
        response: ChatResponse,
        chunks: Sequence[Any],
        prompt: str,
        context_block: str,
        latency_ms: float,
    ) -> GeneratedAnswer:
        """Assemble a :class:`GeneratedAnswer` from a raw model response."""
        answer = response.content.strip()
        refusal = looks_like_refusal(answer)
        citations = [] if refusal else self.parse_citations(answer, chunks)

        result = GeneratedAnswer(
            answer=answer,
            citations=citations,
            retrieved=self._retrieved_payload(chunks),
            stats=response.stats,
            prompt=prompt,
            context_block=context_block,
            insufficient_context=refusal,
            uncited=not refusal and not citations,
            latency_ms=latency_ms,
        )

        if result.uncited:
            logger.warning(
                "Answer contained no valid citations",
                extra={"answer_chars": len(answer), "chunks": len(chunks)},
            )
        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    async def generate(self, question: str, chunks: Sequence[Any]) -> GeneratedAnswer:
        """Produce a grounded answer for ``question`` from ``chunks``.

        Args:
            question: The user's question.
            chunks: Retrieved passages in rank order.

        Returns:
            The generated answer with parsed citations.

        Raises:
            OllamaError: If the model is unreachable or misconfigured.
        """
        budget = context_budget_chars(self._settings.ollama_num_ctx)
        messages, prompt, context_block = build_messages(question, chunks, budget)

        tracing.log_context(
            prompt,
            query=question,
            system_prompt=SYSTEM_PROMPT,
            chunks=list(chunks),
            context_window=self._settings.ollama_num_ctx,
        )

        started = time.perf_counter()
        response = await self._client.chat(messages)
        latency_ms = (time.perf_counter() - started) * 1000.0

        tracing.log_generation(
            model=self._client.model,
            prompt=prompt,
            response=response.content,
            input_tokens=response.stats.prompt_tokens,
            output_tokens=response.stats.completion_tokens,
            duration_ms=latency_ms,
        )

        result = self._finalise(response, chunks, prompt, context_block, latency_ms)
        logger.info(
            "Generated answer",
            extra={
                "chunks": len(chunks),
                "citations": len(result.citations),
                "insufficient_context": result.insufficient_context,
                "model": self._client.model,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return result

    async def stream(
        self, question: str, chunks: Sequence[Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a grounded answer as server-sent events.

        Yields:
            ``{"type": "sources", ...}`` once up front, then ``{"type": "token",
            "content": str}`` per delta, then a terminal ``{"type": "done",
            ...}`` carrying the parsed citations and statistics.

        Raises:
            OllamaError: If the model is unreachable or misconfigured.
        """
        budget = context_budget_chars(self._settings.ollama_num_ctx)
        messages, prompt, context_block = build_messages(question, chunks, budget)

        tracing.log_context(
            prompt,
            query=question,
            system_prompt=SYSTEM_PROMPT,
            chunks=list(chunks),
            context_window=self._settings.ollama_num_ctx,
        )

        yield {"type": "sources", "retrieved": self._retrieved_payload(chunks)}

        started = time.perf_counter()
        buffer: List[str] = []
        stats: Dict[str, Any] = {}
        finish_reason = ""

        async for event in self._client.chat_stream(messages):
            if event["type"] == "token":
                buffer.append(event["content"])
                yield event
            elif event["type"] == "done":
                stats = dict(event.get("stats") or {})
                finish_reason = str(event.get("finish_reason", ""))

        latency_ms = (time.perf_counter() - started) * 1000.0
        answer = "".join(buffer).strip()
        response = ChatResponse(
            content=answer,
            stats=GenerationStats(**{key: stats[key] for key in stats if key in GenerationStats.__dataclass_fields__}),
            finish_reason=finish_reason,
        )
        result = self._finalise(response, chunks, prompt, context_block, latency_ms)

        tracing.log_generation(
            model=self._client.model,
            prompt=prompt,
            response=answer,
            input_tokens=response.stats.prompt_tokens,
            output_tokens=response.stats.completion_tokens,
            duration_ms=latency_ms,
        )

        yield {"type": "done", **result.to_dict()}
