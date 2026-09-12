"""RAG evaluation CLI.

Measures retrieval quality (precision@k, recall@k, nDCG@k, MRR, MAP, hit rate)
and answer quality (faithfulness, citation coverage) against a golden set.

Examples:
    # Full run with real models, ingesting the sample corpus first
    python -m evaluation.run_eval --dataset data/documents

    # Retrieval-only, no generation, faster
    python -m evaluation.run_eval --dataset data/documents --retrieval-only

    # Offline smoke test: hashing embeddings and a fake LLM, no downloads
    python -m evaluation.run_eval --fake --dataset data/documents

Reports are written to ``evaluation/reports/`` as both JSON and Markdown.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.config import Settings, get_settings
from app.container import ServiceContainer
from app.generation.ollama_client import FakeOllamaClient
from app.ingestion.embedder import BaseEmbedder, HashingEmbedder
from app.logging_config import setup_logging
from app.retrieval.reranker import BaseReranker, NoOpReranker
from evaluation.dataset import (
    DEFAULT_GOLDEN_SET_PATH,
    GoldenItem,
    load_golden_set,
    resolve_relevant_chunk_ids,
    summarise_golden_set,
)
from evaluation.judge import FaithfulnessJudge, summarise_verdicts
from evaluation.metrics import (
    average_precision,
    citation_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise,
)

logger = logging.getLogger("evaluation")

#: Default cut-offs for the ranking metrics.
DEFAULT_K_VALUES = (1, 3, 5, 10)

#: Metrics averaged across the golden set.
_RANKING_METRICS = ("precision", "recall", "ndcg", "hit_rate")


def build_container(args: argparse.Namespace) -> ServiceContainer:
    """Construct the service container for an evaluation run.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A started container, using fakes when ``--fake`` was passed.
    """
    settings = get_settings()
    overrides: Dict[str, Any] = {}
    if args.chroma_dir:
        overrides["chroma_dir"] = args.chroma_dir
    if overrides:
        settings = settings.model_copy(update=overrides)

    embedder: Optional[BaseEmbedder] = None
    reranker: Optional[BaseReranker] = None
    client = None

    if args.fake:
        logger.warning("Running in --fake mode: hashing embeddings, no real model")
        embedder = HashingEmbedder(dimension=settings.embedding_dim)
        reranker = NoOpReranker()
        client = FakeOllamaClient()

    container = ServiceContainer(
        settings=settings,
        embedder=embedder,
        reranker=reranker,
        client=client,
    )
    container.startup()
    return container


def ingest_datasets(container: ServiceContainer, paths: Sequence[str]) -> None:
    """Ingest documents before evaluation.

    Args:
        container: The started container.
        paths: Files or directories to ingest.
    """
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            logger.error("Skipping missing dataset path", extra={"path": raw})
            continue

        if path.is_dir():
            result = container.pipeline.ingest_directory(path)
        else:
            result = container.pipeline.ingest_files([path])

        logger.info(
            "Ingested dataset",
            extra={"path": raw, "chunks": result.chunks, "sources": len(result.sources)},
        )
        for failure in result.failures:
            logger.warning("Ingestion failure", extra={"error": failure})


async def evaluate_item(
    item: GoldenItem,
    container: ServiceContainer,
    judge: FaithfulnessJudge,
    top_k: int,
    k_values: Sequence[int],
    generate: bool,
) -> Dict[str, Any]:
    """Evaluate a single golden-set question.

    Args:
        item: The labelled question.
        container: The started container.
        judge: Faithfulness judge.
        top_k: Number of chunks to retrieve.
        k_values: Cut-offs for the ranking metrics.
        generate: Whether to run generation and faithfulness scoring.

    Returns:
        A row of metric values for this question.
    """
    retrieval = container.retriever.retrieve(item.question, top_k=top_k)
    retrieved_ids = [chunk.chunk_id for chunk in retrieval.chunks]
    relevant_ids = resolve_relevant_chunk_ids(item, container.bm25_index.iter_records())

    row: Dict[str, Any] = {
        "question": item.question,
        "expect_refusal": item.expect_refusal,
        "relevant_chunks_in_index": len(relevant_ids),
        "retrieved_chunks": len(retrieved_ids),
        "retrieved_sources": sorted({chunk.source for chunk in retrieval.chunks}),
        "retrieved_chunk_ids": retrieved_ids,
        "timings": {key: round(value, 2) for key, value in retrieval.timings.items()},
        "answer": None,
        "faithfulness": None,
        "faithfulness_method": None,
        "citation_coverage": None,
        "cited_indices": [],
        "insufficient_context": None,
        "uncited": None,
        "refusal_correct": None,
    }

    if not item.expect_refusal:
        # Ranking metrics are undefined when the corpus deliberately holds no
        # answer, so refusal probes are scored only on behaviour.
        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
            row[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
            row[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_ids, relevant_ids, k)
        row["mrr"] = reciprocal_rank(retrieved_ids, relevant_ids)
        row["map"] = average_precision(retrieved_ids, relevant_ids, k=top_k)

    if not generate or retrieval.is_empty:
        if item.expect_refusal:
            # Nothing retrieved is the correct outcome for a refusal probe.
            row["refusal_correct"] = True
        return row

    generated = await container.generator.generate(item.question, retrieval.chunks)
    contexts = [chunk.text for chunk in retrieval.chunks]
    verdict = await judge.score(item.question, generated.answer, contexts)

    row["answer"] = generated.answer
    row["faithfulness"] = verdict.score
    row["faithfulness_method"] = verdict.method
    row["faithfulness_reason"] = verdict.reason
    row["citation_coverage"] = citation_coverage(generated.cited_indices, len(retrieval.chunks))
    row["cited_indices"] = generated.cited_indices
    row["insufficient_context"] = generated.insufficient_context
    row["uncited"] = generated.uncited

    if item.expect_refusal:
        # A refusal probe passes when the model declines rather than inventing.
        row["refusal_correct"] = bool(generated.insufficient_context)

    return row


async def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute a full evaluation run.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The complete report payload.
    """
    items = load_golden_set(args.golden_set)
    if not items:
        raise SystemExit("Golden set is empty - nothing to evaluate.")

    container = build_container(args)
    try:
        if args.dataset:
            ingest_datasets(container, args.dataset)

        indexed = container.vector_store.count()
        if indexed == 0:
            logger.warning("The index is empty; all retrieval metrics will be zero")

        k_values = sorted(set(args.k_values) | {args.top_k})
        judge = FaithfulnessJudge(
            client=container.client if not args.retrieval_only else None,
            enabled=not args.no_judge and not args.retrieval_only,
        )

        rows: List[Dict[str, Any]] = []
        for position, item in enumerate(items, start=1):
            logger.info(
                "Evaluating question",
                extra={"position": position, "total": len(items), "question": item.question[:70]},
            )
            rows.append(
                await evaluate_item(
                    item,
                    container,
                    judge,
                    top_k=args.top_k,
                    k_values=k_values,
                    generate=not args.retrieval_only,
                )
            )

        metric_names = [f"{name}@{k}" for k in k_values for name in _RANKING_METRICS]
        metric_names += ["mrr", "map", "citation_coverage"]
        if not args.retrieval_only:
            metric_names.append("faithfulness")

        summary = summarise(rows, metric_names)

        report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "top_k": args.top_k,
                "k_values": k_values,
                "rerank_enabled": container.settings.rerank_enabled,
                "embedding_model": container.embedder.model_name,
                "reranker_model": container.retriever.reranker_model,
                "generation_model": container.client.model,
                "fake_mode": bool(args.fake),
                "retrieval_only": bool(args.retrieval_only),
                "indexed_chunks": indexed,
            },
            "golden_set": summarise_golden_set(items),
            "metrics": {name: round(value, 4) for name, value in summary.metrics.items()},
            "metric_counts": dict(summary.counts),
            "faithfulness_detail": _faithfulness_detail(rows),
            "refusal_probes": _refusal_summary(rows),
            "per_question": rows,
        }
        return report
    finally:
        await container.shutdown()


def _refusal_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise behaviour on deliberately unanswerable questions.

    Args:
        rows: Per-question result rows.

    Returns:
        Count of refusal probes and how many were correctly refused.
    """
    probes = [row for row in rows if row.get("expect_refusal")]
    if not probes:
        return {"probes": 0, "correct": 0, "accuracy": None}

    scored = [row for row in probes if isinstance(row.get("refusal_correct"), bool)]
    correct = sum(1 for row in scored if row["refusal_correct"])
    return {
        "probes": len(probes),
        "correct": correct,
        "accuracy": round(correct / len(scored), 4) if scored else None,
    }


def _faithfulness_detail(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise how faithfulness was measured across the run.

    Args:
        rows: Per-question result rows.

    Returns:
        Mean faithfulness plus the LLM/lexical split.
    """
    scored = [row for row in rows if isinstance(row.get("faithfulness"), (int, float))]
    if not scored:
        return {"mean": 0.0, "judged_count": 0, "lexical_count": 0, "llm_share": 0.0}

    judged = [row for row in scored if row.get("faithfulness_method") == "llm"]
    return {
        "mean": round(sum(float(row["faithfulness"]) for row in scored) / len(scored), 4),
        "judged_count": len(judged),
        "lexical_count": len(scored) - len(judged),
        "llm_share": round(len(judged) / len(scored), 4),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the report as a human-readable Markdown document.

    Args:
        report: The report payload from :func:`run_evaluation`.

    Returns:
        The Markdown document.
    """
    config = report["config"]
    lines: List[str] = [
        "# RAG evaluation report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    for key, value in config.items():
        lines.append(f"| `{key}` | {value} |")

    golden = report["golden_set"]
    lines += [
        "",
        "## Golden set",
        "",
        f"- Questions: **{golden['questions']}**",
        f"- Labelled: **{golden['labelled']}**",
        f"- Sources: {', '.join(golden['sources']) or '—'}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Score |",
        "| --- | --- |",
    ]
    for name, value in report["metrics"].items():
        lines.append(f"| {name} | {value:.4f} |")

    detail = report.get("faithfulness_detail") or {}
    if detail:
        lines += [
            "",
            "## Faithfulness detail",
            "",
            f"- Mean: **{detail.get('mean', 0.0):.4f}**",
            f"- LLM-judged: {detail.get('judged_count', 0)} "
            f"(share {detail.get('llm_share', 0.0):.0%})",
            f"- Lexical proxy: {detail.get('lexical_count', 0)}",
        ]

    probes = report.get("refusal_probes") or {}
    if probes.get("probes"):
        accuracy = probes.get("accuracy")
        lines += [
            "",
            "## Refusal probes",
            "",
            f"- Unanswerable questions: **{probes['probes']}**",
            f"- Correctly refused: **{probes['correct']}**"
            + (f" (accuracy {accuracy:.0%})" if accuracy is not None else ""),
        ]

    lines += [
        "",
        "## Per question",
        "",
        "| # | Question | MRR | nDCG@k | Recall@k | Faithfulness | Cited |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    k_values = config.get("k_values") or []
    best_k = max(k_values) if k_values else 0

    for position, row in enumerate(report["per_question"], start=1):
        question = str(row["question"]).replace("|", "\\|")
        question = question if len(question) <= 70 else question[:67] + "..."
        faithfulness = row.get("faithfulness")
        faithfulness_text = f"{float(faithfulness):.2f}" if isinstance(faithfulness, (int, float)) else "—"
        cited = ", ".join(str(index) for index in row.get("cited_indices") or []) or "—"

        lines.append(
            f"| {position} | {question} | {row.get('mrr', 0.0):.3f} "
            f"| {row.get(f'ndcg@{best_k}', 0.0):.3f} | {row.get(f'recall@{best_k}', 0.0):.3f} "
            f"| {faithfulness_text} | {cited} |"
        )

    lines += [
        "",
        "---",
        "",
        "Retrieval metrics treat a chunk as relevant when its source document is",
        "listed in the golden item's `relevant_sources` (or its id appears in",
        "`relevant_chunk_ids`). Faithfulness is scored by the local model as a",
        "judge where available, falling back to a lexical coverage proxy; the",
        "`faithfulness detail` block records the split.",
        "",
    ]
    return "\n".join(lines)


def write_report(report: Dict[str, Any], output_dir: Path, stamp: str) -> tuple[Path, Path]:
    """Write the JSON and Markdown reports.

    Args:
        report: The report payload.
        output_dir: Destination directory, created if needed.
        stamp: Timestamp string used in the file names.

    Returns:
        The ``(json_path, markdown_path)`` written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"eval_{stamp}.json"
    markdown_path = output_dir / f"eval_{stamp}.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_eval",
        description="Evaluate the RAG pipeline against a golden question set.",
    )
    parser.add_argument(
        "--golden-set",
        default=str(DEFAULT_GOLDEN_SET_PATH),
        help="Path to the golden set JSON file.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="File or directory to ingest before evaluating. Repeatable.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve per question.")
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="Cut-offs for precision/recall/nDCG/hit-rate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "reports"),
        help="Directory for the JSON and Markdown reports.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and faithfulness scoring.",
    )
    parser.add_argument("--no-judge", action="store_true", help="Use the lexical faithfulness proxy.")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Offline mode: hashing embeddings and a fake LLM, no model downloads.",
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help="Override the vector store directory (useful for throwaway runs).",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    setup_logging(args.log_level, json_logs=False)

    try:
        report = asyncio.run(run_evaluation(args))
    except SystemExit as exc:
        print(str(exc))
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path, markdown_path = write_report(report, Path(args.output_dir), stamp)

    print("\nAggregate metrics")
    print("-" * 40)
    for name, value in report["metrics"].items():
        print(f"  {name:<18} {value:.4f}")
    detail = report.get("faithfulness_detail") or {}
    if detail.get("mean") is not None:
        print(f"  {'faithfulness src':<18} llm={detail.get('judged_count', 0)} "
              f"lexical={detail.get('lexical_count', 0)}")
    print(f"\nJSON report:     {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
