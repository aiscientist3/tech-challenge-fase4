"""Split only the documents that are longer than the configured chunk size.

An order document is usually a few hundred characters. Splitting it would
separate the review from the items it describes, so short documents stay whole.
Long review comments still go through a recursive splitter, and every piece
keeps the parent metadata plus its own chunk index.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Return one or more chunks per document, each with `chunk_index` set."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunked: list[Document] = []
    split_count = 0
    for document in documents:
        text = document.page_content
        if len(text) > chunk_size:
            pieces = splitter.split_text(text) or [text]
            split_count += 1
        else:
            pieces = [text]
        for index, piece in enumerate(pieces):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = int(index)
            chunked.append(Document(page_content=piece, metadata=metadata))
    logger.info(
        "Chunked %s documents into %s chunks (%s documents were longer than %s characters).",
        len(documents),
        len(chunked),
        split_count,
        chunk_size,
    )
    return chunked
