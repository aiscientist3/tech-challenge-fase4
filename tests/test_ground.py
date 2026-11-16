"""The written answer is kept only when the reviews support a regional comparison."""

from rag.generation.chat import to_chat_response
from rag.generation.ground import ground_answer
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.select import RetrievalResult, RetrievedDocument


def test_a_plain_answer_stays() -> None:
    answer = "Vários comentários falam de atraso na entrega [2] e de pedido incompleto [5]."
    assert ground_answer(answer, ["SP", "BA", "RJ"]) == answer


def test_a_denied_comparison_stays() -> None:
    answer = "Esses comentários não são mais frequentes no Norte."
    assert ground_answer(answer, ["SP", "BA"]) == answer


def test_north_and_northeast_claim_needs_the_cited_states() -> None:
    answer = (
        "Os problemas de entrega são mais frequentes em regiões como Norte e Nordeste. "
        "Isso aparece em 1, 4, 5, 6, 9."
    )
    states = ["SP", "RJ", "MG", "SP", "BA", "RS", "PR", "AM", "PE"]
    corrected = ground_answer(answer, states)
    assert "não dá para dizer" in corrected
    assert "o Norte" in corrected
    assert "o Nordeste" in corrected


def test_a_comparison_stays_when_the_cited_reviews_show_it() -> None:
    answer = "O atraso é mais frequente no Nordeste. Isso aparece em [1], [2], [3], [4] e [5]."
    states = ["BA", "PE", "CE", "SP", "SP"]
    assert ground_answer(answer, states) == answer


def test_chat_response_replaces_an_unsupported_region_claim() -> None:
    selected = [
        _document("Review comment: A entrega atrasou.", state)
        for state in ("SP", "RJ", "MG")
    ]
    body = to_chat_response(
        _result(selected),
        answer="Os problemas são mais frequentes no Nordeste. Isso aparece em [1], [2] e [3].",
    )
    assert "não dá para dizer" in body["answer"]
    assert body["insufficient_evidence"] is False
    assert len(body["sources"]) == 3


def _document(text: str, state: str) -> RetrievedDocument:
    return RetrievedDocument(
        id=state,
        text=text,
        distance=0.4,
        doc_type="review",
        entity_id=state,
        order_id=f"order-{state}",
        review_score=1,
        product_category="bed_bath_table",
        customer_state=state,
        purchase_year=2018,
        purchase_month=5,
    )


def _result(selected: list[RetrievedDocument]) -> RetrievalResult:
    return RetrievalResult(
        original_question="A região muda a entrega?",
        question="A região muda a entrega?",
        filters=RetrievalFilters(doc_type="review"),
        selected=selected,
        dropped=[],
        insufficient_evidence=False,
        candidate_count=len(selected),
    )
