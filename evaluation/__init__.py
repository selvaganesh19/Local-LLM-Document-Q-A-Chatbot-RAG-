"""Offline evaluation harness for the RAG pipeline.

Public surface:

* :mod:`evaluation.metrics` - pure ranking and grounding metrics.
* :mod:`evaluation.judge` - LLM-as-judge faithfulness scoring.
* :mod:`evaluation.dataset` - golden-set loading and label resolution.
* :mod:`evaluation.run_eval` - the CLI that ties them together.
"""

from evaluation.dataset import GoldenItem, load_golden_set
from evaluation.judge import FaithfulnessJudge, FaithfulnessVerdict
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

__all__ = [
    "GoldenItem",
    "load_golden_set",
    "FaithfulnessJudge",
    "FaithfulnessVerdict",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "reciprocal_rank",
    "average_precision",
    "hit_rate_at_k",
    "lexical_grounding_score",
    "citation_coverage",
    "summarise",
]
