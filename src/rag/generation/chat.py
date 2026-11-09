"""Shape a retrieval result as the JSON body of POST /chat.

The frontend already expects `answer`, `insufficient_evidence`, and `sources`.
`score` on each source is the Chroma cosine distance: lower means closer.
"""

from __future__ import annotations

from rag.generation.answer import draft_answer
from rag.generation.prompt import REFUSAL
from rag.retrieval.payload import to_payload
from rag.retrieval.select import RetrievalResult, RetrievedDocument

_COMMENT_MARK = "Review comment:"


def to_chat_response(
    result: RetrievalResult,
    *,
    max_reviews: int = 4,
    answer: str | None = None,
) -> dict[str, object]:
    """Build the chat JSON from one retrieval result.

    `answer` replaces the quoted comments when a model has already written
    the reply. The refusal path ignores that text.
    """
    if max_reviews < 1:
        raise ValueError("max_reviews must be >= 1.")
    payload = to_payload(result)
    reviews = payload.get("retrieved_reviews")
    if isinstance(reviews, list):
        payload["retrieved_reviews"] = reviews[:max_reviews]
    draft = draft_answer(payload, max_reviews=max_reviews).rstrip("\n")
    insufficient = result.insufficient_evidence or draft == REFUSAL
    if insufficient or answer is None or not answer.strip():
        answer = draft
    else:
        answer = answer.strip()
    sources = [] if insufficient else [_source(doc) for doc in result.selected[:max_reviews]]
    return {
        "answer": answer,
        "insufficient_evidence": insufficient,
        "sources": sources,
    }


def _source(document: RetrievedDocument) -> dict[str, object]:
    return {
        "id": document.id,
        "doc_type": document.doc_type,
        "order_id": document.order_id,
        "review_score": document.review_score,
        "product_category": document.product_category,
        "purchase_date": _purchase_date(document),
        "score": round(document.distance, 4),
        "excerpt": _excerpt(document.text),
    }


def _purchase_date(document: RetrievedDocument) -> str:
    if document.purchase_year < 1:
        return ""
    if document.purchase_month < 1:
        return str(document.purchase_year)
    return f"{document.purchase_year:04d}-{document.purchase_month:02d}"


def _excerpt(text: str, limit: int = 320) -> str:
    comment = text
    mark = text.lower().rfind(_COMMENT_MARK.lower())
    if mark >= 0:
        comment = text[mark + len(_COMMENT_MARK) :]
    comment = " ".join(comment.split())
    if len(comment) <= limit:
        return comment
    return comment[: limit - 3].rstrip() + "..."
