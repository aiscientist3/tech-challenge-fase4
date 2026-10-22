"""Comandos de ingestão e de busca vetorial, sem geração de resposta."""

from __future__ import annotations

import argparse

from voc_rag.config import Settings, get_settings
from voc_rag.ingestion.index import SearchHit, index_documents, search_neighbors
from voc_rag.ingestion.prepare import prepare_documents, read_parquet, write_parquet


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="voc-rag",
        description="Carrega as avaliações da Olist no Chroma e consulta os vizinhos.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Prepara os documentos e faz o upsert no Chroma.")
    ingest.add_argument(
        "--from-processed",
        action="store_true",
        help="Relê data/processed/reviews.parquet em vez de refazer os joins.",
    )

    search = commands.add_parser("search", help="Lista os vizinhos de uma frase. Não gera resposta.")
    search.add_argument("query", help="Texto usado como consulta vetorial.")
    search.add_argument("--n", type=int, default=5, help="Quantidade de vizinhos. Padrão: 5.")

    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        if args.command == "ingest":
            _ingest(settings, from_processed=args.from_processed)
        else:
            if args.n < 1:
                parser.error("--n precisa ser pelo menos 1")
            _search(settings, args.query, args.n)
    except Exception as exc:
        if _is_connection_error(exc):
            raise SystemExit(
                "Não foi possível falar com o Chroma em "
                f"{settings.chroma_host}:{settings.chroma_port}. "
                "Suba o servidor com 'docker compose up -d'. "
                f"Detalhe: {exc}"
            ) from exc
        raise


def _ingest(settings: Settings, *, from_processed: bool) -> None:
    parquet_path = settings.data_processed_dir / "reviews.parquet"
    if from_processed:
        documents = read_parquet(parquet_path)
        print(f"Lidos {len(documents)} documentos de {parquet_path}.")
    else:
        result = prepare_documents(settings.data_raw_dir)
        documents = result.documents
        print(
            "Preparação: "
            f"{result.rows_read} avaliações lidas, "
            f"{result.dropped_duplicates} duplicadas removidas, "
            f"{result.dropped_empty} sem texto, "
            f"{result.dropped_without_score} sem nota, "
            f"{len(documents)} documentos."
        )
        write_parquet(documents, parquet_path)
        print(f"Documentos gravados em {parquet_path}.")

    if not documents:
        print("Nenhum documento para indexar.")
        return

    indexed = index_documents(documents, settings)
    print(f"Coleção {settings.chroma_collection} atualizada com {indexed} documentos.")


def _search(settings: Settings, query: str, n_results: int) -> None:
    hits = search_neighbors(query, n_results, settings)
    if not hits:
        print("Nenhum vizinho encontrado.")
        return
    for position, hit in enumerate(hits, start=1):
        print(_format_hit(position, hit))
        print()


def _format_hit(position: int, hit: SearchHit) -> str:
    score = hit.metadata.get("review_score", "")
    category = hit.metadata.get("product_category", "")
    header = (
        f"{position}. {hit.review_id}  distancia={hit.distance:.4f}"
        f"  nota={score}  categoria={category}"
    )
    return f"{header}\n{hit.document}"


def _is_connection_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ConnectionRefusedError",
        "ReadTimeout",
    }:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "connection refused",
            "failed to connect",
            "could not connect",
            "all connection attempts failed",
        )
    )


if __name__ == "__main__":
    main()
