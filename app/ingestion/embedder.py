"""Embedding models.

Wraps ``sentence-transformers`` behind a small interface so the rest of the
application never depends on the concrete model, and so tests can inject a
deterministic stub without downloading weights.

Vectors are L2-normalised on the way out, which lets the vector store use
cosine distance and lets the semantic cache compare raw dot products.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, List, Sequence

import numpy as np

from app.config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving all-zero rows untouched."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class BaseEmbedder(ABC):
    """Interface every embedder must satisfy."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model, for logs and health output."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector length produced by this embedder."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Embed many texts.

        Args:
            texts: Texts to embed.
            batch_size: Forward-pass batch size.

        Returns:
            Array of shape ``(len(texts), dimension)``, L2-normalised.
        """

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            text: The query.

        Returns:
            Array of shape ``(dimension,)``, L2-normalised.
        """
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder backed by a local ``sentence-transformers`` model.

    The model is loaded lazily on first use so that application startup and the
    test suite do not pay for (or require) a multi-hundred-megabyte download.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        dimension_hint: int | None = None,
    ) -> None:
        """Store configuration; the model itself is loaded on demand.

        Args:
            model_name: Hugging Face model identifier.
            device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
            dimension_hint: Expected vector length, used before the model loads.
        """
        self._model_name = model_name
        self._device = device
        self._dimension_hint = dimension_hint
        self._model: "SentenceTransformer | None" = None

    @property
    def model_name(self) -> str:
        """Hugging Face model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Vector length; forces the model to load if unknown."""
        if self._model is None and self._dimension_hint is None:
            self._load()
        if self._model is not None:
            # sentence-transformers 5.x renamed this accessor; the old name
            # still works but warns, so prefer the new one when present.
            accessor = getattr(self._model, "get_embedding_dimension", None) or (
                self._model.get_sentence_embedding_dimension
            )
            return int(accessor())
        return int(self._dimension_hint or 384)

    def _load(self) -> "SentenceTransformer":
        """Load the underlying model once and cache it on the instance."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model",
                extra={"model": self._model_name, "device": self._device},
            )
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts with the local model, normalising the output."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder(BaseEmbedder):
    """Deterministic, dependency-free embedder for smoke tests and CI.

    Uses the "hashing trick" over lower-cased word tokens: each token maps to a
    fixed dimension with a signed weight, so documents sharing vocabulary land
    close together in cosine space.  Quality is far below a real model - it
    exists so the full pipeline can be exercised without downloading weights.
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialise with the desired vector length."""
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        """Fixed identifier for this stub embedder."""
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        """Vector length produced by this embedder."""
        return self._dimension

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Bag-of-words hashing embedding, L2-normalised."""
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_PATTERN.findall(text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "big") % self._dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, column] += sign
        return _normalise(matrix)


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    """Return the process-wide embedder selected by configuration.

    A hashing embedder is substituted when ``DEV_FAKE_LLM`` is enabled, keeping
    offline smoke tests fast and hermetic.
    """
    settings: Settings = get_settings()
    if settings.dev_fake_llm:
        logger.warning("DEV_FAKE_LLM enabled - using HashingEmbedder, not a real model")
        return HashingEmbedder(dimension=settings.embedding_dim)
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        dimension_hint=settings.embedding_dim,
    )


def embed_texts(texts: List[str], embedder: BaseEmbedder | None = None) -> np.ndarray:
    """Convenience helper embedding ``texts`` with the configured model."""
    return (embedder or get_embedder()).embed_documents(texts)
