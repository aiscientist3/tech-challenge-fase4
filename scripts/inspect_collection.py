"""Print collection stats and three similarity searches.

This is a sanity check that the index is readable. It is not the retrieval layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_chroma import Chroma

from rag.config import DOC_TYPES, Settings, configure_logging
from rag.ingestion.indexer import build_embeddings, chroma_client, collection_names

# Portuguese queries exercise the multilingual index without building a retriever.
SANITY_QUERIES = (
    "entrega atrasada",
    "produto veio com defeito",
    "pagamento com boleto",
)


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    client = chroma_client(settings)
    names = collection_names(client)
    if settings.collection_name not in names:
        print(
            f"Collection '{settings.collection_name}' does not exist in "
            f"{settings.resolve_path(settings.chroma_dir)}.",
            file=sys.stderr,
        )
        sys.exit(1)
    collection = client.get_collection(settings.collection_name)
    embeddings = build_embeddings(settings)
    store = Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=embeddings,
    )
    metadata = dict(collection.metadata or {})
    print(f"collection: {settings.collection_name}")
    print(f"persist_directory: {settings.resolve_path(settings.chroma_dir)}")
    print(f"count: {collection.count()}")
    print("metadata:")
    for key in sorted(metadata):
        print(f"  {key}: {metadata[key]}")
    print("documents per doc_type:")
    for doc_type in DOC_TYPES:
        found = collection.get(where={"doc_type": doc_type}, include=[])
        print(f"  {doc_type}: {len(found.get('ids', []))}")
    for query in SANITY_QUERIES:
        print(f"\nquery: {query}")
        results = store.similarity_search_with_score(query, k=3)
        if not results:
            print("  (no results)")
            continue
        for rank, (document, score) in enumerate(results, start=1):
            snippet = " ".join(document.page_content.split())
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            print(
                f"  {rank}. score={score:.4f} "
                f"doc_type={document.metadata.get('doc_type')} "
                f"order_id={document.metadata.get('order_id')} "
                f"review_score={document.metadata.get('review_score')}"
            )
            print(f"     {snippet}")


if __name__ == "__main__":
    main()
