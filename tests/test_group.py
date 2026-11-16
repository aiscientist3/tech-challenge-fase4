"""Grouping the vector hits by region and score."""

from rag.generation.prompt import build_generation
from rag.retrieval.group import RegionTally, example_indexes, group_regions, prefer_counted, tally_text
from rag.retrieval.interpret import asks_region_by_score


def test_a_region_score_question_asks_for_the_low_band() -> None:
    question = "com avaliacoes de todas as regioes do brasil, qual seria a regiao com mais notas baixas?"
    assert asks_region_by_score(question) == "low"


def test_a_named_region_comparison_counts_the_two_sides() -> None:
    from rag.retrieval.interpret import region_question

    question = (
        "os problemas de entrega, tem a ver com a regiao do brasil? "
        "as regioes norte e nordeste tem mais problemas na entrega que regioes sul e sudeste?"
    )
    assert region_question(question) == (
        "count",
        ("Norte", "Nordeste"),
        ("Sul", "Sudeste"),
    )


def test_two_sides_use_how_many_reviews_each_one_brought() -> None:
    groups = group_regions([
        ("BA", 1),
        ("PE", 2),
        ("PA", 1),
        ("SP", 1),
        ("RS", 2),
    ])
    text = tally_text(groups, "count", ("Norte", "Nordeste"), ("Sul", "Sudeste"))
    assert "Norte e Nordeste: 3 avaliações." in text
    assert "Sul e Sudeste: 2 avaliações." in text
    assert "O lado com mais resultados é Norte e Nordeste." in text
    answer = "Pelos comentários desta lista, não dá para dizer que o Sudeste e o Sul tenham mais problemas."
    corrected = prefer_counted(answer, groups, "count", ("Norte", "Nordeste"), ("Sul", "Sudeste"))
    assert "Norte e Nordeste aparece mais" in corrected


def test_a_delivery_question_does_not_ask_for_a_region_count() -> None:
    assert asks_region_by_score("O que os clientes relatam sobre problemas de entrega?") is None


def test_hits_are_grouped_by_region_and_score() -> None:
    groups = group_regions([
        ("BA", 1),
        ("PE", 2),
        ("BA", 5),
        ("SP", 1),
        ("SP", 5),
        ("SP", 4),
        ("XX", 1),
    ])
    by_region = {group.region: group for group in groups}
    assert by_region["Nordeste"] == RegionTally("Nordeste", low=2, high=1)
    assert by_region["Sudeste"] == RegionTally("Sudeste", low=1, high=2)
    assert "XX" not in by_region


def test_the_tally_ranks_the_region_with_the_higher_share() -> None:
    groups = group_regions([
        ("BA", 1),
        ("BA", 2),
        ("BA", 5),
        ("SP", 1),
        ("SP", 5),
        ("SP", 4),
    ])
    text = tally_text(groups, "low")
    assert text.index("Nordeste") < text.index("Sudeste")
    assert "67%" in text
    assert "A maior parcela é Nordeste." in text


def test_a_wrong_region_is_replaced_by_the_count() -> None:
    groups = group_regions([
        ("BA", 1),
        ("BA", 2),
        ("BA", 5),
        ("SP", 5),
        ("RJ", 4),
    ])
    answer = "A região com mais notas baixas é SP, com uma proporção maior."
    corrected = prefer_counted(answer, groups, "low")
    assert corrected.startswith("Nesta busca, Nordeste")
    assert "67%" in corrected


def test_the_prompt_includes_the_region_percentages() -> None:
    payload = {
        "query": {
            "original_question": "Qual região tem mais notas baixas?",
            "normalized_question": "Qual região tem mais notas baixas?",
            "filters": {"doc_type": "review"},
        },
        "retrieved_reviews": [
            {
                "review_id": "1",
                "text": "A entrega atrasou.",
                "review_score": 1,
                "similarity": 0.4,
                "order_id": "o",
                "customer_state": "BA",
                "product_category": "baby",
                "purchase_date": "2018-05",
            },
            {
                "review_id": "2",
                "text": "Chegou no prazo.",
                "review_score": 5,
                "similarity": 0.4,
                "order_id": "p",
                "customer_state": "SP",
                "product_category": "baby",
                "purchase_date": "2018-05",
            },
        ],
        "insufficient_evidence": False,
        "score_tally": "Parcela de notas baixas (1 a 3) em cada região, nesta busca:\n- Nordeste: 67% (2 de 3)\nA maior parcela é Nordeste.",
    }
    generated = build_generation(payload)
    assert generated.prompt is not None
    assert "67%" in generated.prompt.user
    assert "A maior parcela é Nordeste." in generated.prompt.user
    assert "Nordeste\nNota 1/5, BA" in generated.prompt.user
    assert "A entrega atrasou." in generated.prompt.user


def test_each_side_keeps_the_nearest_reviews_up_to_eight() -> None:
    states = ["BA", "PE", "MA", "AL", "RN", "PB", "PI", "CE", "SE", "SP", "RJ", "RS", "SC", "PR", "MG", "ES", "AM", "PA"]
    indexes = example_indexes(
        states,
        (),
        "count",
        ("Norte", "Nordeste"),
        ("Sul", "Sudeste"),
    )
    assert indexes == [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]


def test_a_repeated_order_is_not_sampled_twice() -> None:
    indexes = example_indexes(
        ["BA", "BA", "PE", "SP"],
        (),
        "count",
        ("Nordeste",),
        ("Sudeste",),
        ["same", "same", "other", "sp"],
    )
    assert indexes == [0, 2, 3]


def test_a_comparison_prompt_carries_comments_from_both_sides() -> None:
    payload = {
        "query": {
            "original_question": "O Norte e o Nordeste têm mais problemas de entrega que o Sul e o Sudeste?",
            "normalized_question": "O Norte e o Nordeste têm mais problemas de entrega que o Sul e o Sudeste?",
            "filters": {"doc_type": "review"},
        },
        "retrieved_reviews": [
            {
                "review_id": "1",
                "text": "A transportadora atrasou na Bahia.",
                "review_score": 1,
                "similarity": 0.4,
                "order_id": "o",
                "customer_state": "BA",
                "product_category": "baby",
                "purchase_date": "2018-05",
            },
            {
                "review_id": "2",
                "text": "Não chegou o pedido.",
                "review_score": 2,
                "similarity": 0.4,
                "order_id": "p",
                "customer_state": "PE",
                "product_category": "baby",
                "purchase_date": "2018-05",
            },
            {
                "review_id": "3",
                "text": "Demorou em São Paulo.",
                "review_score": 1,
                "similarity": 0.4,
                "order_id": "q",
                "customer_state": "SP",
                "product_category": "baby",
                "purchase_date": "2018-05",
            },
            {
                "review_id": "4",
                "text": "Atraso no Rio Grande do Sul.",
                "review_score": 2,
                "similarity": 0.4,
                "order_id": "r",
                "customer_state": "RS",
                "product_category": "baby",
                "purchase_date": "2018-05",
            },
        ],
        "insufficient_evidence": False,
        "compare_left": ["Norte", "Nordeste"],
        "compare_right": ["Sul", "Sudeste"],
        "score_tally": (
            "Nesta busca, Norte e Nordeste: 2 avaliações.\n"
            "Sul e Sudeste: 2 avaliações.\n"
            "Os dois lados apareceram na mesma quantidade."
        ),
    }
    generated = build_generation(payload)
    assert generated.prompt is not None
    user = generated.prompt.user
    assert "Norte e Nordeste\nNota 1/5, BA" in user
    assert "A transportadora atrasou na Bahia." in user
    assert "Não chegou o pedido." in user
    assert "Sul e Sudeste\nNota 1/5, SP" in user
    assert "Demorou em São Paulo." in user
    assert "Atraso no Rio Grande do Sul." in user
    assert "quantidade em algarismos e sem colchetes" in user
