"""Tests for the retrieval and grounding metrics."""

from __future__ import annotations

import math

import pytest

from evaluation.metrics import (
    average_precision,
    citation_coverage,
    hit_rate_at_k,
    lexical_grounding_score,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise,
)

RETRIEVED = ["a", "b", "c", "d"]
RELEVANT = {"a", "c"}


def test_precision_at_k() -> None:
    """Precision counts hits within the window over the window size."""
    assert precision_at_k(RETRIEVED, RELEVANT, 1) == 1.0
    assert precision_at_k(RETRIEVED, RELEVANT, 2) == 0.5
    assert precision_at_k(RETRIEVED, RELEVANT, 4) == 0.5
    assert precision_at_k(RETRIEVED, RELEVANT, 0) == 0.0


def test_recall_at_k() -> None:
    """Recall counts hits within the window over everything relevant."""
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == 0.5
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == 1.0
    assert recall_at_k(RETRIEVED, RELEVANT, 10) == 1.0


def test_recall_with_no_relevant_items_is_zero() -> None:
    """Nothing to find means recall is undefined, reported as zero."""
    assert recall_at_k(RETRIEVED, set(), 3) == 0.0


def test_ndcg_at_k_matches_hand_computation() -> None:
    """nDCG is checked against a value derived by hand."""
    # DCG@2 = 1/log2(2) = 1.0 ; IDCG@2 = 1/log2(2) + 1/log2(3) = 1.63093
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))

    assert ndcg_at_k(RETRIEVED, RELEVANT, 2) == pytest.approx(expected, rel=1e-9)


def test_ndcg_rewards_early_relevant_results() -> None:
    """Moving a relevant item up the ranking raises nDCG."""
    early = ndcg_at_k(["a", "b", "c"], {"a"}, 3)
    late = ndcg_at_k(["b", "c", "a"], {"a"}, 3)

    assert early > late


def test_ndcg_handles_graded_gains() -> None:
    """Graded relevance is honoured when a gain map is supplied."""
    gains = {"a": 3.0, "b": 1.0}
    strong_first = ndcg_at_k(["a", "b"], {"a", "b"}, 2, gains=gains)
    weak_first = ndcg_at_k(["b", "a"], {"a", "b"}, 2, gains=gains)

    assert strong_first > weak_first
    assert strong_first == pytest.approx(1.0, rel=1e-9)


def test_ndcg_without_relevant_items_is_zero() -> None:
    """No achievable gain means a zero score, not a division by zero."""
    assert ndcg_at_k(RETRIEVED, set(), 3) == 0.0
    assert ndcg_at_k(RETRIEVED, RELEVANT, 0) == 0.0


def test_reciprocal_rank() -> None:
    """MRR is the reciprocal of the first relevant rank."""
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["b", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["b", "c"], {"a"}) == 0.0


def test_hit_rate() -> None:
    """Hit rate is binary within the window."""
    assert hit_rate_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert hit_rate_at_k(["x", "a"], {"a"}, 1) == 0.0


def test_average_precision_matches_hand_computation() -> None:
    """MAP averages precision at each relevant rank."""
    # Relevant at ranks 1 and 3: (1/1 + 2/3) / 2
    expected = (1.0 + 2.0 / 3.0) / 2.0

    assert average_precision(RETRIEVED, RELEVANT) == pytest.approx(expected, rel=1e-9)


def test_lexical_grounding_score_rewards_supported_answers() -> None:
    """Answers reusing the context's vocabulary score higher."""
    context = ["The notice period is 90 calendar days for senior staff."]

    grounded = lexical_grounding_score("The notice period is 90 calendar days.", context)
    fabricated = lexical_grounding_score("The notice period is 14 working hours.", context)

    assert grounded > fabricated
    assert grounded == pytest.approx(1.0, rel=1e-9)


def test_lexical_grounding_score_handles_empty_answer() -> None:
    """An answer with no content words scores zero rather than dividing by zero."""
    assert lexical_grounding_score("", ["anything"]) == 0.0


def test_citation_coverage() -> None:
    """Coverage is the share of supplied passages that were cited."""
    assert citation_coverage([1, 2, 3], 3) == 1.0
    assert citation_coverage([1, 1, 2], 4) == 0.5
    assert citation_coverage([], 3) == 0.0
    assert citation_coverage([1], 0) == 0.0


def test_citation_coverage_ignores_out_of_range_markers() -> None:
    """A hallucinated citation number must not count towards coverage."""
    assert citation_coverage([9], 3) == 0.0


def test_summarise_averages_only_reported_metrics() -> None:
    """Metrics absent from a row are excluded rather than counted as zero."""
    rows = [
        {"precision@1": 1.0, "faithfulness": 0.9},
        {"precision@1": 0.0, "faithfulness": None},
    ]

    summary = summarise(rows, ["precision@1", "faithfulness"])

    assert summary.metrics["precision@1"] == pytest.approx(0.5)
    assert summary.metrics["faithfulness"] == pytest.approx(0.9)
    assert summary.counts["faithfulness"] == 1
    assert summary.count == 2


def test_summarise_handles_no_rows() -> None:
    """An empty run yields zeroed metrics instead of raising."""
    summary = summarise([], ["mrr"])

    assert summary.count == 0
    assert summary.metrics == {}
