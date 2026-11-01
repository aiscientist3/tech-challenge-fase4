"""Prompt assembly without a language model or a vector collection."""

from __future__ import annotations

from rag.generation.prompt import REFUSAL, build_generation


def _payload(reviews: list[dict[str, object]], *, insufficient: bool = False, question: str = "Quais são os problemas de entrega?") -> dict[str, object]:
    return {
        "query": {
            "original_question": question,
            "normalized_question": question,
            "filters": {"doc_type": "review"},
        },
        "retrieved_reviews": reviews,
        "insufficient_evidence": insufficient,
    }


def _review(review_id: str, text: str, score: int = 1, **extra: object) -> dict[str, object]:
    return {
        "review_id": review_id,
        "text": text,
        "review_score": score,
        "similarity": 0.4,
        "order_id": "order-1",
        "customer_state": "SP",
        "product_category": "baby",
        "purchase_date": "2018-05",
        **extra,
    }


def test_prompt_uses_the_original_question_and_only_evidence_fields() -> None:
    payload = _payload(
        [
            _review("101", "O produto chegou três dias depois do prazo.", 1),
            _review("103", "Meu pedido demorou muito para chegar.", 2),
        ],
        question="  Quais são os problemas de entrega?  ",
    )
    generated = build_generation(payload)
    assert generated.insufficient_evidence is False
    assert generated.prompt is not None
    rendered = generated.prompt.render()
    assert "Always respond in Brazilian Portuguese." in generated.prompt.system
    assert "Pergunta:\nQuais são os problemas de entrega?" in generated.prompt.user
    assert "[101] nota 1/5" in generated.prompt.user
    assert "O produto chegou três dias depois do prazo." in generated.prompt.user
    assert "[103] nota 2/5" in generated.prompt.user
    assert "order-1" not in rendered
    assert "baby" not in rendered
    assert "0.4" not in rendered
    assert rendered.startswith("SYSTEM\n")
    assert "\nUSER\n" in rendered


def test_insufficient_retrieval_does_not_build_a_prompt() -> None:
    payload = _payload([_review("101", "Atraso."), _review("103", "Demorou.")], insufficient=True)
    generated = build_generation(payload)
    assert generated.prompt is None
    assert generated.refusal == REFUSAL
    assert generated.insufficient_evidence is True


def test_one_usable_review_is_not_enough() -> None:
    payload = _payload([
        _review("101", "Atraso."),
        _review("102", "   "),
        _review("103", "Sem nota.", score=0),
    ])
    generated = build_generation(payload)
    assert generated.prompt is None


def test_duplicate_review_id_is_kept_once() -> None:
    payload = _payload([
        _review("101", "Primeiro texto."),
        _review("101", "Texto repetido."),
        _review("103", "Outro pedido."),
    ])
    generated = build_generation(payload)
    assert generated.prompt is not None
    assert generated.prompt.user.count("[101]") == 1
    assert "Texto repetido." not in generated.prompt.user
    assert "Outro pedido." in generated.prompt.user


def test_context_stops_before_the_character_budget() -> None:
    payload = _payload([
        _review("101", "curto"),
        _review("102", "x" * 500),
        _review("103", "ainda cabe"),
    ])
    generated = build_generation(payload, max_context_chars=80)
    assert generated.prompt is not None
    assert "[101]" in generated.prompt.user
    assert "[102]" not in generated.prompt.user
    assert "[103]" not in generated.prompt.user
