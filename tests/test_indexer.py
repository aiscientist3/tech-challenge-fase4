"""Upsert idempotency using a fake embedding model and a temp Chroma directory."""

from __future__ import annotations

import pytest

from rag.ingestion.chunking import chunk_documents
from rag.ingestion.indexer import CollectionConfigError, index_documents
from tests.fake_embeddings import FakeEmbeddings


def _chunks(documents, settings):
    return chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def test_ingesting_twice_does_not_duplicate_chunks(settings, documents) -> None:
    chunks = _chunks(documents, settings)
    embeddings = FakeEmbeddings()
    first = index_documents(
        chunks,
        settings,
        embeddings,
        reset=True,
        sample_size=0,
        dataset_as_of="2018-02-20T09:00:00",
    )
    second = index_documents(
        chunks,
        settings,
        embeddings,
        reset=False,
        sample_size=0,
        dataset_as_of="2018-02-20T09:00:00",
    )
    assert first == second == len(chunks)


def test_existing_collection_rejects_a_different_embedding_model(settings, documents) -> None:
    chunks = _chunks(documents, settings)
    embeddings = FakeEmbeddings()
    index_documents(
        chunks,
        settings,
        embeddings,
        reset=True,
        sample_size=0,
        dataset_as_of="2018-02-20T09:00:00",
    )
    other = settings.model_copy(update={"embedding_model": "other-model"})
    with pytest.raises(CollectionConfigError, match="embedding model"):
        index_documents(
            chunks,
            other,
            embeddings,
            reset=False,
            sample_size=0,
            dataset_as_of="2018-02-20T09:00:00",
        )


def test_existing_collection_rejects_a_different_sample_size(settings, documents) -> None:
    chunks = _chunks(documents, settings)
    embeddings = FakeEmbeddings()
    index_documents(
        chunks,
        settings,
        embeddings,
        reset=True,
        sample_size=0,
        dataset_as_of="2018-02-20T09:00:00",
    )
    with pytest.raises(CollectionConfigError, match="sample_size"):
        index_documents(
            chunks,
            settings,
            embeddings,
            reset=False,
            sample_size=2,
            dataset_as_of="2018-02-20T09:00:00",
        )
