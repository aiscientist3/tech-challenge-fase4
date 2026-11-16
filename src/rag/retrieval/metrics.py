"""Small checks for a retrieved list. No labeled dataset and no model call."""

from __future__ import annotations

import re
import unicodedata

from rag.retrieval.select import RetrievedDocument


def keyword_hit_rate(documents: list[RetrievedDocument], terms: list[str]) -> float:
    """Share of documents whose text contains at least one expected term."""
    if not documents or not terms:
        return 0.0
    folded_terms = [_fold(term) for term in terms if term.strip()]
    if not folded_terms:
        return 0.0
    hits = sum(1 for document in documents if _contains_any(document.text, folded_terms))
    return hits / len(documents)


def mean_similarity(documents: list[RetrievedDocument]) -> float:
    """Mean of `1 - distance` over the selected documents."""
    if not documents:
        return 0.0
    total = sum(max(0.0, 1.0 - document.distance) for document in documents)
    return total / len(documents)


def _contains_any(text: str, terms: list[str]) -> bool:
    folded = _fold(text)
    return any(re.search(rf"\b{re.escape(term)}\b", folded) for term in terms)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_accents.casefold()
