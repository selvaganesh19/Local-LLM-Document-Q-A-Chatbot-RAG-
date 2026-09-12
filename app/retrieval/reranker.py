"""Cross-encoder reranking.

Fusion gets the right chunks *into* the candidate set; a cross-encoder decides
their *order*.  Unlike the bi-encoder used for embedding, a cross-encoder sees
the query and passage together, so it captures term interactions that cosine
similarity cannot - at the cost of one forward pass per candidate, which is why
it only ever runs over the fused top-N rather than the whole corpus.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Optional, Sequence

from app.config import Settings, get_settings
from app.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)


def sigmoid(value: float) -> float:
    """Map an unbounded score to ``(0, 1)``.

    Cross-encoder models emit raw logits; the logistic curve turns them into a
    display-friendly relevance figure without changing their ordering.
    """
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


class BaseReranker(ABC):
    """Interface for candidate rerankers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model."""

    @abstractmethod
    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> List[RetrievedChunk]:
        """Reorder ``chunks`` by relevance to ``query`` and keep ``top_n``."""


class NoOpReranker(BaseReranker):
    """Pass-through reranker used when reranking is disabled.

    Keeps the fusion ordering and derives a display relevance from the fusion
    score, normalised against the best candidate.
    """

    @property
    def model_name(self) -> str:
        """Fixed identifier for the disabled reranker."""
        return "none"

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> List[RetrievedChunk]:
        """Return the first ``top_n`` chunks with normalised relevance."""
        selected = list(chunks[:top_n])
        if selected:
            best = max((chunk.fusion_score for chunk in selected), default=0.0) or 1.0
            for chunk in selected:
                chunk.relevance = min(1.0, chunk.fusion_score / best)
        return selected


class CrossEncoderReranker(BaseReranker):
    """Reranker backed by a local ``sentence-transformers`` cross-encoder.

    The model is loaded lazily so importing the application (and running the
    test suite) never triggers a download.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu") -> None:
        """Store configuration; the model loads on first use.

        Args:
            model_name: Hugging Face cross-encoder identifier.
            device: Torch device string.
        """
        self._model_name = model_name
        self._device = device
        self._model = None

    @property
    def model_name(self) -> str:
        """Hugging Face model identifier."""
        return self._model_name

    def _load(self):
        """Load and cache the cross-encoder."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model", extra={"model": self._model_name, "device": self._device})
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> List[RetrievedChunk]:
        """Score each candidate against the query and keep the best ``top_n``."""
        candidates = list(chunks)
        if not candidates or top_n <= 0:
            return []

        model = self._load()
        pairs = [(query, chunk.text) for chunk in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = float(score)
            chunk.relevance = sigmoid(float(score))

        ordered = sorted(
            candidates,
            key=lambda item: (-(item.rerank_score or 0.0), item.chunk_id),
        )
        logger.debug(
            "Reranked candidates",
            extra={"candidates": len(candidates), "kept": min(top_n, len(ordered))},
        )
        return ordered[:top_n]


@lru_cache(maxsize=1)
def get_reranker() -> BaseReranker:
    """Return the reranker selected by configuration.

    Falls back to :class:`NoOpReranker` when reranking is disabled or when the
    fake-LLM development mode is active.
    """
    settings: Settings = get_settings()
    if not settings.rerank_enabled or settings.dev_fake_llm:
        return NoOpReranker()
    return CrossEncoderReranker(model_name=settings.reranker_model, device=settings.embedding_device)


def reset_reranker_cache() -> None:
    """Clear the cached reranker (used by tests that swap configuration)."""
    get_reranker.cache_clear()
