"""Retrieval and generation quality metrics.

All functions are pure and operate on plain sequences, which keeps them
unit-testable without any model, index or network access.

Ranking metrics assume the retrieved list is ordered best-first:

* ``precision@k`` - of the first ``k`` results, what fraction is relevant.
* ``recall@k`` - of all relevant items, what fraction appears in the first ``k``.
* ``nDCG@k`` - position-weighted gain, penalising relevant items found late.
* ``MRR`` - reciprocal rank of the first relevant result.
* ``hit_rate@k`` - whether any relevant item appears in the first ``k``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence

#: Token pattern used by the lexical grounding proxy.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

#: Words ignored by the lexical grounding proxy - they carry no evidence.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be been but by for from had has have he her his i if in
    into is it its of on or our she that the their them then there these they
    this to was were what when where which who will with would you your
    """.split()
)


def _identity(value: Any) -> str:
    """Normalise an identifier for set comparison."""
    return str(value)


def precision_at_k(retrieved: Sequence[Any], relevant: Collection[Any], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant.

    Args:
        retrieved: Ordered retrieved identifiers, best first.
        relevant: The set of identifiers considered relevant.
        k: Cut-off rank.

    Returns:
        A value in ``[0, 1]``; ``0.0`` when ``k`` is not positive.
    """
    if k <= 0:
        return 0.0
    window = retrieved[:k]
    if not window:
        return 0.0
    relevant_set = {_identity(item) for item in relevant}
    hits = sum(1 for item in window if _identity(item) in relevant_set)
    return hits / len(window)


def recall_at_k(retrieved: Sequence[Any], relevant: Collection[Any], k: int) -> float:
    """Fraction of all relevant items found within the top ``k``.

    Args:
        retrieved: Ordered retrieved identifiers, best first.
        relevant: The set of identifiers considered relevant.
        k: Cut-off rank.

    Returns:
        A value in ``[0, 1]``; ``0.0`` when there is nothing relevant to find.
    """
    relevant_set = {_identity(item) for item in relevant}
    if not relevant_set or k <= 0:
        return 0.0
    found = {_identity(item) for item in retrieved[:k]}
    return len(found & relevant_set) / len(relevant_set)


def hit_rate_at_k(retrieved: Sequence[Any], relevant: Collection[Any], k: int) -> float:
    """``1.0`` when at least one relevant item appears in the top ``k``.

    Args:
        retrieved: Ordered retrieved identifiers, best first.
        relevant: The set of identifiers considered relevant.
        k: Cut-off rank.

    Returns:
        ``1.0`` or ``0.0``.
    """
    relevant_set = {_identity(item) for item in relevant}
    if not relevant_set or k <= 0:
        return 0.0
    return 1.0 if any(_identity(item) in relevant_set for item in retrieved[:k]) else 0.0


def ndcg_at_k(
    retrieved: Sequence[Any],
    relevant: Collection[Any],
    k: int,
    gains: Optional[Mapping[Any, float]] = None,
) -> float:
    """Normalised discounted cumulative gain at rank ``k``.

    Args:
        retrieved: Ordered retrieved identifiers, best first.
        relevant: Identifiers considered relevant (binary gain of 1 each) when
            ``gains`` is not supplied.
        k: Cut-off rank.
        gains: Optional graded relevance map; identifiers absent from it score
            zero.

    Returns:
        A value in ``[0, 1]``; ``0.0`` when no relevant item exists.
    """
    if k <= 0:
        return 0.0

    relevant_set = {_identity(item) for item in relevant}
    gain_map = {_identity(key): float(value) for key, value in (gains or {}).items()}

    def gain_of(item: Any) -> float:
        """Graded gain for one retrieved identifier."""
        key = _identity(item)
        if key in gain_map:
            return gain_map[key]
        return 1.0 if key in relevant_set else 0.0

    ranked_gains = [gain_of(item) for item in retrieved[:k]]

    def dcg(values: Sequence[float]) -> float:
        """Discounted cumulative gain for a gain sequence."""
        return sum(
            (2.0 ** value - 1.0) / math.log2(position + 1)
            for position, value in enumerate(values, start=1)
            if value > 0
        )

    actual = dcg(ranked_gains)

    # Ideal ordering: every achievable gain, sorted descending.
    ideal_gains = sorted(
        [gain for gain in gain_map.values() if gain > 0]
        or [1.0 for _ in relevant_set],
        reverse=True,
    )[:k]
    ideal = dcg(ideal_gains)

    if ideal <= 0:
        return 0.0
    return min(1.0, actual / ideal)


def reciprocal_rank(retrieved: Sequence[Any], relevant: Collection[Any]) -> float:
    """Reciprocal rank of the first relevant result.

    Args:
        retrieved: Ordered retrieved identifiers, best first.
        relevant: The set of identifiers considered relevant.

    Returns:
        ``1/rank`` of the first hit, or ``0.0`` when nothing relevant is found.
    """
    relevant_set = {_identity(item) for item in relevant}
    for rank, item in enumerate(retrieved, start=1):
        if _identity(item) in relevant_set:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: Sequence[Any], relevant: Collection[Any], k: int = 0) -> float:
    """Mean of the precision values at each relevant rank.

    Args:
        retrieved: Ordered retrieved identifiers, best first.
        relevant: The set of identifiers considered relevant.
        k: Optional cut-off; ``0`` means the whole list.

    Returns:
        A value in ``[0, 1]``.
    """
    relevant_set = {_identity(item) for item in relevant}
    if not relevant_set:
        return 0.0

    window = list(retrieved[:k]) if k > 0 else list(retrieved)
    hits = 0
    total = 0.0
    for rank, item in enumerate(window, start=1):
        if _identity(item) in relevant_set:
            hits += 1
            total += hits / rank
    return total / len(relevant_set)


def lexical_grounding_score(answer: str, contexts: Sequence[str]) -> float:
    """Cheap proxy for faithfulness: answer-token coverage by the context.

    This is **not** an entailment check.  It measures how much of the answer's
    vocabulary appears in the retrieved passages, which catches fabricated
    entities and numbers but not subtle misreading.  Use the LLM judge in
    :mod:`evaluation.judge` when a real faithfulness score is required.

    Args:
        answer: The generated answer.
        contexts: The passages supplied to the model.

    Returns:
        Fraction of answer content tokens present in the context, in ``[0, 1]``.
    """
    answer_tokens = [
        token
        for token in _TOKEN_PATTERN.findall(answer.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]
    if not answer_tokens:
        return 0.0

    context_tokens = set()
    for context in contexts:
        context_tokens.update(_TOKEN_PATTERN.findall(context.lower()))

    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return supported / len(answer_tokens)


def citation_coverage(cited_indices: Sequence[int], retrieved_count: int) -> float:
    """Share of retrieved passages the answer actually cited.

    A high value means the answer leaned on most of what it was given; a very
    low value relative to a long answer suggests the model answered from
    parametric memory rather than the documents.

    Args:
        cited_indices: Citation numbers parsed out of the answer.
        retrieved_count: How many passages were supplied.

    Returns:
        A value in ``[0, 1]``.
    """
    if retrieved_count <= 0:
        return 0.0
    unique = {index for index in cited_indices if 1 <= index <= retrieved_count}
    return len(unique) / retrieved_count


@dataclass
class MetricSummary:
    """Aggregated metrics across a golden set.

    Attributes:
        count: Number of evaluated questions.
        metrics: Mean value per metric name.
        counts: Number of questions contributing to each metric.
        per_question: Raw per-question scores, kept for drill-down.
    """

    count: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    per_question: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for the JSON report."""
        return {
            "count": self.count,
            "metrics": {name: round(value, 4) for name, value in self.metrics.items()},
            "counts": dict(self.counts),
            "per_question": self.per_question,
        }


def summarise(rows: Sequence[Mapping[str, Any]], metric_names: Sequence[str]) -> MetricSummary:
    """Average named numeric metrics across per-question rows.

    A metric is averaged over the rows that actually report it, so a question
    excluded from a metric (for example an unanswerable probe, which has no
    relevant passage to retrieve) neither inflates nor deflates the mean.  Per
    metric counts are returned alongside the values so a partial run is visible
    rather than silent.

    Args:
        rows: Per-question result mappings.
        metric_names: Metric keys to average.

    Returns:
        A :class:`MetricSummary` with means and the raw rows.
    """
    summary = MetricSummary(count=len(rows), per_question=[dict(row) for row in rows])
    if not rows:
        return summary

    for name in metric_names:
        values = [
            float(row[name])
            for row in rows
            if isinstance(row.get(name), (int, float))
        ]
        summary.metrics[name] = sum(values) / len(values) if values else 0.0
        summary.counts[name] = len(values)
    return summary
