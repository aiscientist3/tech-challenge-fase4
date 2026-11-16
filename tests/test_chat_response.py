"""Chat JSON built from a retrieval result, without a server or a model."""

from rag.generation.chat import to_chat_response
from rag.generation.prompt import REFUSAL
from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.group import group_regions
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


def test_chat_response_returns_every_selected_review() -> None:
    selected = [
        _document(f"Review comment: Texto {index}.", 0.40 + index / 100, f"review-{index}")
        for index in range(8)
    ]
    body = to_chat_response(_result(selected, insufficient=False), answer="Resumo dos comentários.")
    sources = body["sources"]
    assert isinstance(sources, list) and len(sources) == 8
    assert sources[0]["id"] == "review-0"
    assert sources[7]["id"] == "review-7"


def test_region_comparison_keeps_the_written_paragraph() -> None:
    groups = group_regions([("BA", 1), ("PE", 1), ("PA", 1), ("SP", 1), ("RS", 1)])
    result = RetrievalResult(
        original_question="O Norte e o Nordeste têm mais problemas de entrega que o Sul e o Sudeste?",
        question="o norte e o nordeste tem mais problemas de entrega que o sul e o sudeste?",
        filters=RetrievalFilters(doc_type="review"),
        selected=[
            _document("Review comment: Atrasou na Bahia.", entity_id="review-ba"),
            _document("Review comment: Atrasou em São Paulo.", entity_id="review-sp"),
        ],
        dropped=[],
        insufficient_evidence=False,
        candidate_count=5,
        score_groups=groups,
        score_focus="count",
        compare_left=("Norte", "Nordeste"),
        compare_right=("Sul", "Sudeste"),
    )
    written = (
        "Norte e Nordeste aparecem mais nesta busca. "
        "Em [1] a entrega atrasou na Bahia, e em [2] também houve atraso em São Paulo."
    )
    body = to_chat_response(result, answer=written)
    assert body["answer"] == written
    sources = body["sources"]
    assert isinstance(sources, list) and len(sources) == 2


def test_chat_response_refuses_when_retrieval_is_thin() -> None:
    body = to_chat_response(_result([_document("Review comment: Só uma.")], insufficient=True))
    assert body["insufficient_evidence"] is True
    assert body["answer"] == REFUSAL
    assert body["sources"] == []
