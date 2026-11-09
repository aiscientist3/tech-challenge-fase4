"""Chat JSON built from a retrieval result, without a server or a model."""

from rag.generation.chat import to_chat_response
from rag.generation.prompt import REFUSAL
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.select import RetrievalResult, RetrievedDocument


def _document(text: str, distance: float = 0.42, entity_id: str = "review-1") -> RetrievedDocument:
    return RetrievedDocument(
        id=entity_id,
        text=text,
        distance=distance,
        doc_type="review",
        entity_id=entity_id,
        order_id="order-1",
        review_score=1,
        product_category="bed_bath_table",
        customer_state="SP",
        purchase_year=2018,
        purchase_month=5,
    )


def _result(selected: list[RetrievedDocument], *, insufficient: bool) -> RetrievalResult:
    return RetrievalResult(
        original_question="O que atrasou?",
        question="O que atrasou?",
        filters=RetrievalFilters(doc_type="review"),
        selected=selected,
        dropped=[],
        insufficient_evidence=insufficient,
        candidate_count=len(selected),
    )


def test_chat_response_quotes_comments_and_uses_distance_as_score() -> None:
    result = _result([
        _document("Review comment: Chegou atrasado.", 0.42),
        _document("Review comment: Não recebi o pedido.", 0.50, "review-2"),
    ], insufficient=False)
    body = to_chat_response(result)
    assert body["insufficient_evidence"] is False
    assert "Chegou atrasado." in body["answer"]
    sources = body["sources"]
    assert isinstance(sources, list) and len(sources) == 2
    assert sources[0]["score"] == 0.42
    assert sources[0]["excerpt"] == "Chegou atrasado."
    assert sources[0]["purchase_date"] == "2018-05"
    assert sources[0]["doc_type"] == "review"


def test_chat_response_refuses_when_retrieval_is_thin() -> None:
    body = to_chat_response(_result([_document("Review comment: Só uma.")], insufficient=True))
    assert body["insufficient_evidence"] is True
    assert body["answer"] == REFUSAL
    assert body["sources"] == []
