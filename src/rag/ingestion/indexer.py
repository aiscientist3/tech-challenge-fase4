"""Embed documents and upsert them into a persistent Chroma collection.

Ids are a hash of doc type, entity id, and chunk index. Upserting those ids
makes a second run replace the same rows instead of appending duplicates.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from tqdm import tqdm

from rag.config import Settings

logger = logging.getLogger(__name__)


class CollectionConfigError(Exception):
    """Raised when an existing collection was built with incompatible settings."""


def document_id(doc_type: str, entity_id: str, chunk_index: int) -> str:
    """Stable id for one chunk. The same inputs always return the same hash."""
    payload = f"{doc_type}|{entity_id}|{chunk_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_embeddings(settings: Settings) -> Embeddings:
    """Build the configured embedding function.

    The Hugging Face model runs locally. The OpenAI branch imports its package
    only when selected, so the default install does not need an API key.
    """
    provider = settings.embedding_provider
    logger.info("Loading embeddings provider=%s model=%s.", provider, settings.embedding_model)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        # CPU keeps the index reproducible on machines that also have a GPU.
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "embedding_provider 'openai' requires the langchain-openai package. "
                "Install it and set the provider's API key, or use the Hugging Face provider."
            ) from exc
        return OpenAIEmbeddings(model=settings.embedding_model)
    raise ValueError(f"Unknown embedding provider: {provider}")


def index_documents(
    documents: Sequence[Document],
    settings: Settings,
    embeddings: Embeddings,
    *,
    reset: bool,
    sample_size: int,
    dataset_as_of: str,
) -> int:
    """Upsert chunks in batches. Returns the collection count afterwards."""
    if not documents:
        raise ValueError("No documents to index. Check the doc type selection and the raw data.")
    client = chroma_client(settings)
    collection = _open_collection(
        client,
        settings,
        reset=reset,
        sample_size=sample_size,
        dataset_as_of=dataset_as_of,
    )
    grouped = _group_chunks(documents)
    if not reset and collection.count() > 0 and settings.max_extra_chunks > 0:
        _delete_stale_chunks(collection, grouped, settings)
    batches = list(_batches(documents, settings.batch_size))
    logger.info("Upserting %s chunks in %s batches.", len(documents), len(batches))
    for batch in tqdm(batches, desc="Indexing documents", unit="batch"):
        _upsert_batch(collection, embeddings, batch)
    count = int(collection.count())
    logger.info("Collection %s now has %s chunks.", settings.collection_name, count)
    return count


def chroma_client(settings: Settings) -> chromadb.ClientAPI:
    path = settings.resolve_path(settings.chroma_dir)
    path.mkdir(parents=True, exist_ok=True)
    try:
        from chromadb.config import Settings as ChromaSettings

        return chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    except TypeError:
        return chromadb.PersistentClient(path=str(path))


def _open_collection(
    client: chromadb.ClientAPI,
    settings: Settings,
    *,
    reset: bool,
    sample_size: int,
    dataset_as_of: str,
) -> chromadb.Collection:
    name = settings.collection_name
    if reset and name in collection_names(client):
        logger.info("Deleting collection %s because --reset was set.", name)
        client.delete_collection(name)
    metadata = _collection_metadata(settings, sample_size=sample_size, dataset_as_of=dataset_as_of)
    if name in collection_names(client):
        collection = client.get_collection(name)
        _guard_existing(collection.metadata, settings, sample_size)
        # Chroma rejects hnsw:space on modify even when the value is unchanged.
        # The distance function stays the one set at creation.
        editable = {key: value for key, value in metadata.items() if key != "hnsw:space"}
        collection.modify(metadata=editable)
        return collection
    logger.info("Creating collection %s with cosine space.", name)
    return client.create_collection(name=name, metadata=metadata)


def _guard_existing(
    metadata: dict[str, object] | None,
    settings: Settings,
    sample_size: int,
) -> None:
    stored = dict(metadata or {})
    model = stored.get("embedding_model")
    if model != settings.embedding_model:
        raise CollectionConfigError(
            f"The collection '{settings.collection_name}' was built with embedding model "
            f"'{model}', but the current configuration uses '{settings.embedding_model}'. "
            "Re-run with --reset to recreate the collection. "
            "Mixing vectors from different models makes similarity search meaningless."
        )
    stored_sample = stored.get("sample_size")
    if stored_sample is None or int(stored_sample) != int(sample_size):
        raise CollectionConfigError(
            f"The collection '{settings.collection_name}' was ingested with "
            f"sample_size={stored_sample}, but this run uses sample_size={sample_size} "
            "(0 means the full orders table). Re-run with --reset so summaries and "
            "order documents stay from the same population."
        )


def _collection_metadata(
    settings: Settings,
    *,
    sample_size: int,
    dataset_as_of: str,
) -> dict[str, str | int | float | bool]:
    return {
        # hnsw:space is accepted only when the collection is created.
        # distance_space records the same choice for later inspection.
        "hnsw:space": "cosine",
        "distance_space": "cosine",
        "embedding_model": settings.embedding_model,
        "embedding_provider": settings.embedding_provider,
        "chunk_size": int(settings.chunk_size),
        "chunk_overlap": int(settings.chunk_overlap),
        "dataset_version": settings.dataset_version,
        "dataset_as_of": dataset_as_of,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_size": int(sample_size),
    }


def collection_names(client: chromadb.ClientAPI) -> set[str]:
    names: set[str] = set()
    for item in client.list_collections():
        if isinstance(item, str):
            names.add(item)
        else:
            names.add(str(item.name))
    return names


def _group_chunks(documents: Sequence[Document]) -> dict[tuple[str, str], list[Document]]:
    grouped: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for document in documents:
        metadata = document.metadata
        if "doc_type" not in metadata or "entity_id" not in metadata or "chunk_index" not in metadata:
            raise ValueError("Each chunk needs doc_type, entity_id, and chunk_index metadata.")
        key = (str(metadata["doc_type"]), str(metadata["entity_id"]))
        grouped[key].append(document)
    for key, chunks in grouped.items():
        chunks.sort(key=lambda document: int(document.metadata["chunk_index"]))
        indexes = [int(document.metadata["chunk_index"]) for document in chunks]
        if indexes != list(range(len(chunks))):
            raise ValueError(f"Chunk indexes for {key[0]}:{key[1]} must be contiguous from 0. Got {indexes}.")
    return grouped


def _delete_stale_chunks(
    collection: chromadb.Collection,
    grouped: dict[tuple[str, str], list[Document]],
    settings: Settings,
) -> None:
    """Delete tail ids that a shorter re-index would otherwise leave behind."""
    stale: list[str] = []
    for (doc_type, entity_id), chunks in grouped.items():
        start = len(chunks)
        stale.extend(
            document_id(doc_type, entity_id, index)
            for index in range(start, start + settings.max_extra_chunks)
        )
    deleted = 0
    for batch in _batches(stale, settings.batch_size):
        found = collection.get(ids=list(batch), include=[])
        found_ids = list(found.get("ids", []))
        if found_ids:
            collection.delete(ids=found_ids)
            deleted += len(found_ids)
    if deleted:
        logger.info("Deleted %s stale chunks left by a previous ingestion.", deleted)


def _upsert_batch(
    collection: chromadb.Collection,
    embeddings: Embeddings,
    documents: Sequence[Document],
) -> None:
    ids = [
        document_id(
            str(document.metadata["doc_type"]),
            str(document.metadata["entity_id"]),
            int(document.metadata["chunk_index"]),
        )
        for document in documents
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("A batch contains duplicate document ids.")
    texts = [document.page_content for document in documents]
    vectors = embeddings.embed_documents(texts)
    if len(vectors) != len(texts):
        raise ValueError("The embedding function returned a different number of vectors than texts.")
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=[_sanitize_metadata(document.metadata) for document in documents],
        embeddings=[[float(value) for value in vector] for vector in vectors],
    )


def _sanitize_metadata(metadata: dict[str, object]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            raise ValueError(f"Metadata field '{key}' is None. Chroma does not accept nulls.")
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            clean[key] = int(value)
        elif isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"Metadata field '{key}' is not a finite float.")
            clean[key] = float(value)
        else:
            clean[key] = str(value)
    return clean


def _batches(items: Sequence[object], size: int) -> list[Sequence[object]]:
    return [items[start : start + size] for start in range(0, len(items), size)]
