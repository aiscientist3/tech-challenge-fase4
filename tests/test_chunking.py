"""Conditional splitting keeps parent metadata on every chunk."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.ingestion.chunking import chunk_documents
from rag.ingestion.indexer import document_id


def test_short_document_stays_one_chunk() -> None:
    document = Document(
        page_content="Order abc placed on 2018-01-01.",
        metadata={"doc_type": "order", "entity_id": "abc", "order_id": "abc"},
    )
    chunks = chunk_documents([document], chunk_size=1000, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0].page_content == document.page_content
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["order_id"] == "abc"


def test_long_document_splits_and_keeps_metadata() -> None:
    text = "palavra " * 200
    assert len(text) > 1000
    document = Document(
        page_content=text,
        metadata={"doc_type": "review", "entity_id": "rev-1", "review_score": 1, "order_id": "abc"},
    )
    chunks = chunk_documents([document], chunk_size=1000, chunk_overlap=150)
    assert len(chunks) >= 2
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.metadata["entity_id"] == "rev-1"
        assert chunk.metadata["review_score"] == 1
        assert chunk.metadata["doc_type"] == "review"
        assert len(chunk.page_content) <= 1000
    ids = [
        document_id(chunk.metadata["doc_type"], chunk.metadata["entity_id"], chunk.metadata["chunk_index"])
        for chunk in chunks
    ]
    assert ids == [
        document_id("review", "rev-1", index) for index in range(len(chunks))
    ]
    assert len(set(ids)) == len(ids)


def test_document_id_depends_only_on_its_inputs() -> None:
    assert document_id("order", "abc", 0) == document_id("order", "abc", 0)
    assert document_id("order", "abc", 0) != document_id("order", "abc", 1)
    assert document_id("order", "abc", 0) != document_id("review", "abc", 0)
    assert len(document_id("order", "abc", 0)) == 64
