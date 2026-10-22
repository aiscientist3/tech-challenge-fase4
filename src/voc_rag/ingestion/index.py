"""Carga e consulta dos vetores na API HTTP do Chroma."""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from voc_rag.config import Settings
from voc_rag.embeddings import embed_passages, embed_query
from voc_rag.ingestion.prepare import ReviewDocument

# Distância cosseno: 0 é idêntico. A coleção não calcula embedding;
# os vetores sempre entram prontos, com o modelo E5.
COLLECTION_CONFIGURATION = {"hnsw": {"space": "cosine"}}


@dataclass(frozen=True)
class SearchHit:
    review_id: str
    distance: float
    document: str
    metadata: dict


def index_documents(documents: list[ReviewDocument], settings: Settings) -> int:
    if not documents:
        return 0

    client = chroma_client(settings)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        configuration=COLLECTION_CONFIGURATION,
    )
    batch_size = _batch_size(client, settings.upsert_batch_size)
    indexed = 0
    total = len(documents)
    for start in range(0, total, batch_size):
        chunk = documents[start : start + batch_size]
        embeddings = embed_passages(
            [document.document for document in chunk],
            settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        collection.upsert(
            ids=[document.review_id for document in chunk],
            documents=[document.document for document in chunk],
            metadatas=[document.metadata() for document in chunk],
            embeddings=embeddings,
        )
        indexed += len(chunk)
        print(f"Enviados {indexed}/{total} documentos para {settings.chroma_collection}.")
    return indexed


def search_neighbors(query: str, n_results: int, settings: Settings) -> list[SearchHit]:
    client = chroma_client(settings)
    collection = client.get_collection(name=settings.chroma_collection)
    vector = embed_query(query, settings.embedding_model)
    found = collection.query(
        query_embeddings=[vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    ids = found.get("ids") or [[]]
    if not ids or not ids[0]:
        return []

    documents = found.get("documents") or [[]]
    metadatas = found.get("metadatas") or [[]]
    distances = found.get("distances") or [[]]
    hits: list[SearchHit] = []
    for review_id, document, metadata, distance in zip(
        ids[0],
        documents[0],
        metadatas[0],
        distances[0],
        strict=True,
    ):
        hits.append(
            SearchHit(
                review_id=review_id,
                distance=float(distance),
                document=document or "",
                metadata=dict(metadata or {}),
            )
        )
    return hits


def chroma_client(settings: Settings):
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _batch_size(client, configured: int) -> int:
    getter = getattr(client, "get_max_batch_size", None)
    if getter is None:
        return configured
    try:
        limit = int(getter())
    except Exception:
        return configured
    if limit < 1:
        return configured
    return min(configured, limit)
