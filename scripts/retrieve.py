"""Search the collection and print the retrieval JSON plus the prompt.

The question is cleaned, filters named in it are applied, and the same
embedding model that built the collection ranks the reviews. The prompt is
assembled from those reviews. No language model is called.

Example:
    python scripts/retrieve.py "Em São Paulo quais são os motivos de nota baixa?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import DOC_TYPES, Settings, configure_logging
from rag.generation.prompt import build_generation
from rag.ingestion.indexer import CollectionConfigError, build_embeddings
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.payload import to_payload
from rag.retrieval.retrieve import CollectionNotFoundError, retrieve
from rag.retrieval.select import SelectionPolicy


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)
    try:
        policy = SelectionPolicy(
            candidate_k=args.candidate_k,
            k=args.k,
            max_distance=args.max_distance,
            min_keep=args.min_keep,
        )
        embeddings = build_embeddings(settings)
        result = retrieve(
            args.question,
            settings,
            embeddings,
            filters=_filters(args.doc_type),
            policy=policy,
        )
    except (CollectionNotFoundError, CollectionConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    payload = to_payload(result)
    generated = build_generation(payload, min_reviews=policy.min_keep)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("\n---\n")
    if generated.prompt is None:
        print(generated.refusal)
        return
    print(generated.prompt.render(), end="")


def _filters(doc_type: str) -> RetrievalFilters:
    if doc_type == "all":
        return RetrievalFilters()
    return RetrievalFilters(doc_type=doc_type)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve reviews for one question.")
    parser.add_argument("question", help="Question to embed and search, in the user's words.")
    parser.add_argument(
        "--doc-type",
        default="review",
        choices=[*DOC_TYPES, "all"],
        help="Metadata filter applied before the search. Default: review.",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-distance", type=float, default=0.65)
    parser.add_argument("--min-keep", type=int, default=2)
    return parser


if __name__ == "__main__":
    main()
