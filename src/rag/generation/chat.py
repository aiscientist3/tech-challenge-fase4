"""Shape a retrieval result as the JSON body of POST /chat.

The frontend already expects `answer`, `insufficient_evidence`, and `sources`.
`score` on each source is the Chroma cosine distance: lower means closer.
"""

from __future__ import annotations

from rag.generation.answer import draft_answer
from rag.generation.ground import ground_answer
from rag.generation.prompt import REFUSAL, without_instruction_echo
from rag.retrieval.payload import to_payload
from rag.retrieval.select import RetrievalResult, RetrievedDocument

_COMMENT_MARK = "Review comment:"


def to_chat_response(
    result: RetrievalResult,
    *,
    max_reviews: int | None = None,
    answer: str | None = None,
) -> dict[str, object]:
    """Build the chat JSON from one retrieval result.

    `answer` replaces the quoted comments when a model has already written
    the reply. The refusal path ignores that text. Every selected review is
    returned unless `max_reviews` sets a smaller cap.
    """
    if max_reviews is not None and max_reviews < 1:
        raise ValueError("max_reviews must be >= 1.")
    payload = to_payload(result)
    reviews = payload.get("retrieved_reviews")
    if isinstance(reviews, list) and max_reviews is not None:
        payload["retrieved_reviews"] = reviews[:max_reviews]
    limit = len(result.selected) if max_reviews is None else max_reviews
    has_tally = bool(result.score_focus) and bool(payload.get("score_tally"))
    if limit < 1 and not has_tally:
        return {"answer": REFUSAL, "insufficient_evidence": True, "sources": []}
    draft = REFUSAL if limit < 1 else draft_answer(payload, max_reviews=limit).rstrip("\n")
    insufficient = result.insufficient_evidence or (draft == REFUSAL and not has_tally)
    if answer and answer.strip() and not result.insufficient_evidence:
        answer = without_instruction_echo(answer.strip(), question=result.original_question) or REFUSAL
        if not has_tally:
            answer = ground_answer(answer, _states(reviews, limit))
    else:
        answer = draft
    sources = [] if insufficient else [_source(doc) for doc in result.selected[:limit]]
    return {
        "answer": answer,
        "insufficient_evidence": insufficient,
        "sources": sources,
    }


def _states(reviews: object, limit: int) -> list[str]:
    if not isinstance(reviews, list):
        return []
    states: list[str] = []
    for item in reviews[:limit]:
        if not isinstance(item, dict):
            continue
        state = item.get("customer_state")
        states.append(state.strip().upper() if isinstance(state, str) else "")
    return states


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
