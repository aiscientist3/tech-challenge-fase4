from voc_rag.cli import _format_hit, _is_connection_error
from voc_rag.ingestion.index import SearchHit


def test_chroma_down_is_a_connection_error_and_a_missing_csv_is_not():
    chroma_down = ValueError(
        "Could not connect to a Chroma server. Are you sure it is running?"
    )
    assert _is_connection_error(chroma_down)
    assert not _is_connection_error(
        FileNotFoundError("olist_order_reviews_dataset.csv ausente")
    )


def test_search_output_lists_the_neighbor_without_an_answer():
    hit = SearchHit(
        review_id="rev-a",
        distance=0.125,
        document="Nota: 2/5\nComentário: atrasou",
        metadata={"review_score": 2, "product_category": "cama_mesa_banho"},
    )
    text = _format_hit(1, hit)
    assert "rev-a" in text
    assert "distancia=0.1250" in text
    assert "nota=2" in text
    assert "atrasou" in text
    assert "Resposta" not in text
