"""Ingest Olist documents into the persistent Chroma collection.

Examples:
    python scripts/ingest.py --sample 1000
    python scripts/ingest.py --reset --doc-types order,review
    python scripts/ingest.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.documents import Document

from rag.config import DOC_TYPES, Settings, configure_logging
from rag.ingestion.chunking import chunk_documents
from rag.ingestion.documents import build_documents, write_documents_parquet
from rag.ingestion.extract import DatasetValidationError, load_raw_tables
from rag.ingestion.indexer import CollectionConfigError, build_embeddings, index_documents
from rag.ingestion.transform import sample_tables, transform_tables

SUMMARY_ORDER = ("order", "review", "category_summary", "state_summary")


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)
    started = time.perf_counter()
    try:
        doc_types = resolve_doc_types(settings, args.doc_types)
        tables = load_raw_tables(settings.resolve_path(settings.raw_dir))
        sample_size = 0
        if args.sample is not None:
            tables, sample_size = sample_tables(tables, args.sample, settings.random_seed)
        transformed = transform_tables(tables)
        documents = build_documents(transformed, doc_types)
        write_documents_parquet(documents, settings.resolve_path(settings.processed_dir))
        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        embeddings = build_embeddings(settings)
        collection_count = index_documents(
            chunks,
            settings,
            embeddings,
            reset=args.reset,
            sample_size=sample_size,
            dataset_as_of=transformed.dataset_as_of,
        )
    except (DatasetValidationError, CollectionConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - started
    print(_summary(documents, len(chunks), elapsed, collection_count))


def resolve_doc_types(settings: Settings, raw: str | None) -> set[str]:
    """Use the CLI list when present, otherwise the feature flags.

    Orders stay enabled unless the caller passes an explicit list without them.
    """
    if raw is None:
        chosen = {"order"}
        if settings.enable_review_documents:
            chosen.add("review")
        if settings.enable_aggregate_documents:
            chosen.update({"category_summary", "state_summary"})
        return chosen
    chosen = {part.strip() for part in raw.split(",") if part.strip()}
    if not chosen:
        raise ValueError("--doc-types did not contain any document type.")
    unknown = sorted(chosen - set(DOC_TYPES))
    if unknown:
        raise ValueError(
            f"Unknown doc types: {', '.join(unknown)}. Expected: {', '.join(DOC_TYPES)}."
        )
    return chosen


def _summary(
    documents: list[Document],
    chunk_count: int,
    elapsed: float,
    collection_count: int,
) -> str:
    counts = Counter(str(document.metadata["doc_type"]) for document in documents)
    rendered = ", ".join(f"{name}={counts[name]}" for name in SUMMARY_ORDER if counts[name])
    return "\n".join(
        [
            "Ingestion summary",
            f"documents per doc_type: {rendered}",
            f"chunks indexed: {chunk_count}",
            f"elapsed time: {elapsed:.1f}s",
            f"collection size: {collection_count}",
        ]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Olist documents into Chroma.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Ingest only N orders, chosen with the configured random seed.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection before indexing.",
    )
    parser.add_argument(
        "--doc-types",
        default=None,
        help="Comma-separated list: order,review,category_summary,state_summary.",
    )
    return parser


if __name__ == "__main__":
    main()
