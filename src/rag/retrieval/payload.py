"""JSON body returned by retrieval for the next pipeline stage.

`similarity` is `1 - distance` because the collection stores normalized
vectors in cosine space. Lower distance and higher similarity name the same
ordering. The original question is kept beside the normalized one so a later
prompt can quote what the user wrote.
"""

from __future__ import annotations

from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.select import RetrievalResult, RetrievedDocument


def to_payload(result: RetrievalResult) -> dict[str, object]:
    """Serialize one retrieval result."""
    return {
        "query": {
            "original_question": result.original_question,
            "normalized_question": result.question,
            "filters": _filters(result.filters),
        },
        "retrieved_reviews": [_review(document) for document in result.selected],
        "insufficient_evidence": result.insufficient_evidence,
    }


def similarity(distance: float) -> float:
    """Convert a cosine distance into a similarity rounded to 4 decimals."""
    return round(1.0 - float(distance), 4)


def _filters(filters: RetrievalFilters) -> dict[str, object]:
    payload: dict[str, object] = {}
    if filters.doc_type is not None:
        payload["doc_type"] = filters.doc_type
    if filters.review_score_min is not None:
        payload["review_score_min"] = filters.review_score_min
    if filters.review_score_max is not None:
        payload["review_score_max"] = filters.review_score_max
    if filters.product_category is not None:
        payload["product_category"] = filters.product_category
    if filters.purchase_year is not None:
        payload["purchase_year"] = filters.purchase_year
    if filters.customer_state is not None:
        payload["customer_state"] = filters.customer_state
    return payload


def _review(document: RetrievedDocument) -> dict[str, object]:
    return {
        "review_id": document.entity_id,
        "text": document.text,
        "review_score": document.review_score,
        "similarity": similarity(document.distance),
        "order_id": document.order_id,
        "customer_state": document.customer_state,
        "product_category": document.product_category,
        "purchase_date": _purchase_date(document),
    }


def _purchase_date(document: RetrievedDocument) -> str:
    if document.purchase_year < 1:
        return ""
    if document.purchase_month < 1:
        return str(document.purchase_year)
    return f"{document.purchase_year:04d}-{document.purchase_month:02d}"
