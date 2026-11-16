"""Score a few fixed questions against the collection. No model writes an answer.

For each question the script reports how many of the selected reviews mention
an expected word, the mean vector similarity, and whether a metadata filter
was applied.

Example:
    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import Settings, configure_logging
from rag.ingestion.indexer import CollectionConfigError, build_embeddings
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.metrics import keyword_hit_rate, mean_similarity
from rag.retrieval.retrieve import CollectionNotFoundError, retrieve

# Terms are alternatives: a review counts when its text has one of them.
CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "O que os clientes relatam sobre atraso na entrega?",
        ("atraso", "atrasou", "demorou", "não recebi", "nao recebi"),
    ),
    (
        "Em São Paulo quais são os motivos de nota baixa?",
        ("defeito", "ruim", "qualidade", "quebrado", "não recomendo", "veio"),
    ),
    (
        "O que os clientes mais elogiam na entrega?",
        ("rápido", "rapido", "antes do prazo", "elogio", "ótimo", "otimo"),
    ),
)


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    try:
        embeddings = build_embeddings(settings)
    except (CollectionConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"{'pergunta':<52} {'acerto':>8} {'sim.':>8}  filtros")
    for question, terms in CHECKS:
        try:
            result = retrieve(
                question,
                settings,
                embeddings,
                filters=RetrievalFilters(doc_type="review"),
            )
        except (CollectionNotFoundError, CollectionConfigError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        rate = keyword_hit_rate(result.selected, list(terms))
        similarity = mean_similarity(result.selected)
        print(
            f"{question[:52]:<52} {rate:8.0%} {similarity:8.2f}  {result.filters.describe()}"
        )


if __name__ == "__main__":
    main()
