"""Deterministic embeddings so indexer tests do not download a model."""

from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Map each text to a stable vector derived from its SHA-256 digest."""

    def __init__(self, size: int = 8) -> None:
        self.size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[index % len(digest)] / 127.5) - 1.0 for index in range(self.size)]
