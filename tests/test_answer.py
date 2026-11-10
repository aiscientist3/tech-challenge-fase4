"""Extractive answer from a retrieval payload, without a collection or a model."""

from rag.generation.answer import draft_answer
from rag.generation.prompt import REFUSAL


def _payload(reviews: list[dict[str, object]], *, insufficient: bool = False) -> dict[str, object]:
    return {
        "query": {
            "original_question": "O que atrasou a entrega?",
            "normalized_question": "O que atrasou a entrega?",
            "filters": {"doc_type": "review"},
        },
        "retrieved_reviews": reviews,
        "insufficient_evidence": insufficient,
    }


def _review(review_id: str, text: str, score: int = 1) -> dict[str, object]:
    return {
        "review_id": review_id,
        "text": text,
        "review_score": score,
        "similarity": 0.58,
        "order_id": "abc",
        "customer_state": "SP",
        "product_category": "baby",
        "purchase_date": "2018-05",
    }


def test_answer_quotes_the_comment_and_hides_the_english_wrapper() -> None:
    payload = _payload([
        _review("101", "Review score: 1/5. Review comment: O pedido chegou atrasado."),
        _review("103", "Review comment: Veio com defeito e sem nota fiscal."),
    ])
    text = draft_answer(payload)
    assert "O que atrasou a entrega?" in text
    assert "O pedido chegou atrasado." in text
    assert "Veio com defeito e sem nota fiscal." in text
    assert "Review score" not in text
    assert "[1] nota 1/5 · pedido abc · similaridade 0.58" in text
    assert "Nenhum modelo reescreveu isso." in text


def test_insufficient_payload_is_the_refusal() -> None:
    payload = _payload([
        _review("101", "Review comment: Atraso."),
        _review("103", "Review comment: Demorou."),
    ], insufficient=True)
    assert draft_answer(payload) == REFUSAL


def test_one_review_is_not_enough() -> None:
    payload = _payload([_review("101", "Review comment: Só uma.")])
    assert draft_answer(payload) == REFUSAL
