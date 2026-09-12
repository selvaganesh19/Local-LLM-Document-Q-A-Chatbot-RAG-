"""Prompt construction for grounded, cited answers.

Two design choices matter here:

* **Numbered passages.**  Context blocks are labelled ``[1]``, ``[2]``, ... and
  the system prompt requires the model to cite those numbers inline.  The same
  numbering is returned to the UI, so every claim can be traced back to a
  chunk.
* **Explicit refusal path.**  The model is told to answer with a fixed sentence
  when the context is insufficient.  That fixed string is also used by the
  citation parser to recognise a refusal, and by the faithfulness evaluator to
  score an answer as trivially grounded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.generation.ollama_client import ChatMessage

#: Returned by the model (and recognised by the parser) when context is lacking.
INSUFFICIENT_CONTEXT_ANSWER = "I don't know based on the provided documents."

#: Rough characters-per-token ratio used to budget the context window.
CHARS_PER_TOKEN = 4

SYSTEM_PROMPT = f"""You are a document question-answering assistant.

Rules:
1. Answer ONLY using the numbered context passages supplied by the user.
2. Cite the passage number inline for every factual claim, like [1] or [2][3].
3. Never invent facts, sources, page numbers or citation numbers.
4. If the passages do not contain the answer, reply exactly:
   {INSUFFICIENT_CONTEXT_ANSWER}
5. Be concise and direct. Prefer the document's own terminology.
6. Do not mention "the context", "the passages" or these rules in your answer.
"""


def format_source_label(chunk: Any) -> str:
    """Build the human-readable provenance header for one passage.

    Args:
        chunk: A chunk-like object exposing ``source``, optional ``page`` and
            a ``chunk_id``.

    Returns:
        A short label such as ``"report.pdf, page 3"``.
    """
    source = str(getattr(chunk, "source", "unknown"))
    metadata: Dict[str, Any] = dict(getattr(chunk, "metadata", {}) or {})

    page = getattr(chunk, "page", None)
    if page is None and isinstance(metadata.get("page"), (int, float)):
        page = int(metadata["page"])

    label = source if page is None else f"{source}, page {int(page)}"
    return label


def build_context_block(chunks: Sequence[Any], max_chars: int = 0) -> str:
    """Render retrieved chunks as numbered passages.

    Args:
        chunks: Retrieved chunks in rank order.
        max_chars: Soft character budget.  When positive, passages are added
            while they fit; the first passage is always included so an
            oversized document still yields something to answer from.

    Returns:
        The formatted context block, or an empty string when there are no
        chunks.
    """
    if not chunks:
        return ""

    blocks: List[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        text = str(getattr(chunk, "text", "")).strip()
        if not text:
            continue
        block = f"[{index}] {format_source_label(chunk)}\n{text}"
        if max_chars and blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)


def build_user_prompt(question: str, context_block: str) -> str:
    """Assemble the user turn containing the numbered passages and question."""
    if not context_block:
        return (
            "No context passages were retrieved.\n\n"
            f"Question: {question}"
        )
    return (
        f"Context passages:\n\n{context_block}\n\n"
        f"---\nQuestion: {question}\n\n"
        "Answer with inline citations such as [1]."
    )


def build_messages(
    question: str,
    chunks: Sequence[Any],
    context_budget_chars: int = 0,
) -> tuple[List[ChatMessage], str, str]:
    """Build the full chat payload for a grounded answer.

    Args:
        question: The user's question.
        chunks: Retrieved chunks in rank order.
        context_budget_chars: Character budget for the context block.

    Returns:
        A ``(messages, user_prompt, context_block)`` triple.  The user prompt
        and context block are returned separately so both can be handed to
        RAGObserve without re-deriving them.
    """
    context_block = build_context_block(chunks, max_chars=context_budget_chars)
    user_prompt = build_user_prompt(question, context_block)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]
    return messages, user_prompt, context_block


def context_budget_chars(num_ctx: int) -> int:
    """Derive a context-block character budget from the model's context window.

    Half of the window is reserved for the passages; the rest absorbs the
    system prompt, the question and the generated answer.
    """
    return max(0, int(num_ctx * CHARS_PER_TOKEN * 0.5))
