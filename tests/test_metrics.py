"""Keyword hit rate over a retrieved list, without a collection."""

from rag.retrieval.metrics import keyword_hit_rate, mean_similarity
from rag.retrieval.select import RetrievedDocument


def test_hit_rate_counts_reviews_that_mention_the_subject() -> None:
    documents = [
        _document("A entrega atrasou três dias."),
        _document("Gostei muito do produto."),
        _document("Não recebi o pedido."),
    ]
    rate = keyword_hit_rate(documents, ["atraso", "atrasou", "não recebi"])
    assert rate == 2 / 3


def test_empty_selection_has_no_hit_rate() -> None:
    assert keyword_hit_rate([], ["atraso"]) == 0.0
    assert mean_similarity([]) == 0.0


def test_mean_similarity_uses_one_minus_distance() -> None:
    documents = [_document("um", distance=0.20), _document("dois", distance=0.40)]
    assert mean_similarity(documents) == 0.7


def _document(text: str, distance: float = 0.4) -> RetrievedDocument:
    return RetrievedDocument(
        id=text,
        text=text,
        distance=distance,
        doc_type="review",
        entity_id=text,
        order_id="order",
        review_score=1,
        product_category="",
        customer_state="SP",
        purchase_year=2018,
        purchase_month=1,
    )
