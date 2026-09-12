"""Golden-set loading for RAG evaluation.

A golden set pairs a question with the material that *should* answer it.  Labels
can be given at two granularities:

* ``relevant_sources`` - document names.  Robust to chunking changes, so this is
  the recommended default.
* ``relevant_chunk_ids`` - exact chunk ids, for pinning a specific passage.

Relevant sources are resolved to concrete chunk ids against the live index at
evaluation time, which keeps the metrics honest without freezing the chunker's
output into the fixture.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

#: Default golden-set location, relative to the repository root.
DEFAULT_GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"


class GoldenSetError(ValueError):
    """Raised when a golden set file is missing or malformed."""


@dataclass
class GoldenItem:
    """One evaluated question.

    Attributes:
        question: The question to ask.
        relevant_sources: Document names that contain the answer.
        relevant_chunk_ids: Specific chunk ids that contain the answer.
        reference_answer: Optional human-written answer, for manual review.
        expect_refusal: The corpus deliberately does not answer this, so the
            model should refuse.  Such items are excluded from the ranking
            metrics and scored on refusal behaviour instead.
        notes: Optional free-text note about the question.
    """

    question: str
    relevant_sources: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None
    expect_refusal: bool = False
    notes: Optional[str] = None

    @property
    def is_labelled(self) -> bool:
        """Whether the item carries any relevance label at all."""
        return bool(self.relevant_sources or self.relevant_chunk_ids)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to the on-disk shape."""
        return {
            "question": self.question,
            "relevant_sources": list(self.relevant_sources),
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "reference_answer": self.reference_answer,
            "expect_refusal": self.expect_refusal,
            "notes": self.notes,
        }


def _parse_item(raw: Any, position: int) -> GoldenItem:
    """Validate and convert one raw golden-set entry."""
    if not isinstance(raw, dict):
        raise GoldenSetError(f"Item {position} must be an object, got {type(raw).__name__}")

    question = str(raw.get("question", "")).strip()
    if not question:
        raise GoldenSetError(f"Item {position} has no 'question'")

    def as_list(key: str) -> List[str]:
        """Read a list-of-strings field, tolerating a bare string."""
        value = raw.get(key, [])
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(entry) for entry in value]
        raise GoldenSetError(f"Item {position}: '{key}' must be a string or list of strings")

    return GoldenItem(
        question=question,
        relevant_sources=as_list("relevant_sources"),
        relevant_chunk_ids=as_list("relevant_chunk_ids"),
        reference_answer=raw.get("reference_answer"),
        expect_refusal=bool(raw.get("expect_refusal", False)),
        notes=raw.get("notes"),
    )


def load_golden_set(path: Path | str = DEFAULT_GOLDEN_SET_PATH) -> List[GoldenItem]:
    """Load and validate a golden set.

    Args:
        path: JSON file containing either a top-level array of items or an
            object with an ``items`` array.

    Returns:
        The parsed items.

    Raises:
        GoldenSetError: If the file is missing, is not valid JSON, or contains
            malformed items.
    """
    source = Path(path)
    if not source.exists():
        raise GoldenSetError(f"Golden set not found: {source}")

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"Golden set is not valid JSON ({source}): {exc}") from exc

    if isinstance(payload, dict):
        payload = payload.get("items", [])
    if not isinstance(payload, list):
        raise GoldenSetError("Golden set must be a JSON array, or an object with an 'items' array")

    items = [_parse_item(raw, position) for position, raw in enumerate(payload, start=1)]
    logger.info("Loaded golden set", extra={"path": str(source), "items": len(items)})
    return items


def resolve_relevant_chunk_ids(
    item: GoldenItem,
    index_records: Iterable[tuple[str, str, Dict[str, Any]]],
) -> Set[str]:
    """Expand a golden item's labels into concrete chunk ids.

    Args:
        item: The labelled question.
        index_records: Iterable of ``(chunk_id, text, metadata)`` from the
            lexical index.

    Returns:
        The set of chunk ids considered relevant for this question.
    """
    relevant: Set[str] = set(item.relevant_chunk_ids)
    wanted_sources = set(item.relevant_sources)

    if wanted_sources:
        for chunk_id, _text, metadata in index_records:
            if str(metadata.get("source", "")) in wanted_sources:
                relevant.add(chunk_id)

    return relevant


def summarise_golden_set(items: Sequence[GoldenItem]) -> Dict[str, Any]:
    """Describe a golden set, for the top of the evaluation report.

    Args:
        items: The loaded items.

    Returns:
        Counts and the distinct sources referenced.
    """
    sources: Set[str] = set()
    for item in items:
        sources.update(item.relevant_sources)
    return {
        "questions": len(items),
        "labelled": sum(1 for item in items if item.is_labelled),
        "refusal_probes": sum(1 for item in items if item.expect_refusal),
        "sources": sorted(sources),
    }
