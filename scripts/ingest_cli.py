"""Command-line ingestion helper.

Indexes documents without going through the HTTP API, which is handy for bulk
imports, cron jobs and container entrypoint hooks.

Examples:
    # Index everything under data/documents
    python -m scripts.ingest_cli --path data/documents

    # Wipe the index first, then re-import
    python -m scripts.ingest_cli --path data/documents --reset

    # Index a single phrase under a chosen source name
    python -m scripts.ingest_cli --text "Deploys happen on Tuesdays." --name notes.txt

    # Inspect what is currently indexed
    python -m scripts.ingest_cli --list

    # Remove one document
    python -m scripts.ingest_cli --delete handbook.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

from app.config import get_settings
from app.container import ServiceContainer
from app.logging_config import setup_logging

logger = logging.getLogger("ingest")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.ingest_cli",
        description="Index, list or remove documents for the local RAG service.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", help="File or directory to index.")
    source.add_argument("--text", help="Literal text to index.")
    source.add_argument("--list", action="store_true", help="List indexed documents and exit.")
    source.add_argument("--delete", metavar="SOURCE", help="Remove one indexed document.")

    parser.add_argument("--name", default="pasted-text.txt", help="Source name for --text.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the entire index before ingesting. Destructive.",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Do not descend into sub-directories.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success, 1 when a path cannot be read).
    """
    args = parse_args(argv)
    setup_logging(args.log_level, json_logs=False)

    settings = get_settings()
    if not settings.llm_configured:
        # Ingestion itself needs no model, so this is a warning, not a failure.
        logger.warning(
            "Ollama is not configured; indexing still works but /api/chat will return 503"
        )

    container = ServiceContainer(settings=settings)
    container.startup()

    try:
        return _run(args, container)
    finally:
        # Ingestion is synchronous; nothing async to close, but flush traces.
        from app.observability import tracing

        tracing.flush()


def _run(args: argparse.Namespace, container: ServiceContainer) -> int:
    """Execute the requested operation."""
    pipeline = container.pipeline

    if args.list:
        documents = container.vector_store.list_sources()
        if not documents:
            print("Index is empty.")
            return 0
        print(f"{'source':<50} {'chunks':>7}  type")
        print("-" * 70)
        for entry in documents:
            print(f"{entry['source']:<50} {entry['chunks']:>7}  {entry['type']}")
        stats = pipeline.index_stats()
        print(f"\n{stats['vectors']} vectors across {stats['sources']} document(s)")
        return 0

    if args.delete:
        result = pipeline.delete_source(args.delete)
        if not result["vectors"] and not result["lexical"]:
            print(f"No indexed chunks found for '{args.delete}'.")
            return 1
        print(f"Removed {result['vectors']} vector(s) and {result['lexical']} lexical entr(ies).")
        return 0

    if args.reset:
        logger.warning("Resetting the index: every indexed document will be removed")
        pipeline.reset()

    if args.text:
        result = pipeline.ingest_text(args.text, args.name)
    else:
        from pathlib import Path

        target = Path(args.path)
        if not target.exists():
            print(f"Path does not exist: {target}", file=sys.stderr)
            return 1
        if target.is_dir():
            result = pipeline.ingest_directory(target, recursive=not args.no_recursive)
        else:
            result = pipeline.ingest_files([target])

    print(
        f"Indexed {result.chunks} chunk(s) from {result.documents} document(s) "
        f"in {result.duration_ms:.0f} ms using {result.embedder}."
    )
    if result.sources:
        print(f"Sources: {', '.join(result.sources)}")
    for failure in result.failures:
        print(f"  skipped: {failure}", file=sys.stderr)

    return 1 if result.failures and not result.chunks else 0


if __name__ == "__main__":
    sys.exit(main())
