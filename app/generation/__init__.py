"""Grounded generation via a locally hosted Ollama model."""

from app.generation.generator import (
    SNIPPET_LENGTH,
    AnswerGenerator,
    Citation,
    GeneratedAnswer,
    looks_like_refusal,
    make_snippet,
)
from app.generation.ollama_client import (
    ChatMessage,
    ChatResponse,
    FakeOllamaClient,
    GenerationStats,
    OllamaClient,
    OllamaError,
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

__all__ = [
    "AnswerGenerator",
    "Citation",
    "GeneratedAnswer",
    "make_snippet",
    "looks_like_refusal",
    "SNIPPET_LENGTH",
    "ChatMessage",
    "ChatResponse",
    "FakeOllamaClient",
    "GenerationStats",
    "OllamaClient",
    "OllamaError",
    "OllamaNotConfiguredError",
    "INSUFFICIENT_CONTEXT_ANSWER",
    "SYSTEM_PROMPT",
    "build_context_block",
    "build_messages",
    "build_user_prompt",
    "context_budget_chars",
    "format_source_label",
]
