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
    assert "português do Brasil" in generated.prompt.system
    assert "como quem conversa" in generated.prompt.system
    assert "Os comentários apontam" not in generated.prompt.system
    assert "não repita a resposta anterior" in generated.prompt.user
    assert 'começando por "As avaliações compartilham"' in generated.prompt.user
    assert "Cite os números" not in generated.prompt.user
    assert "Cite [n]" not in generated.prompt.system
    assert "Pergunta:\nQuais são os problemas de entrega?" in generated.prompt.user
    assert "Nota 1/5" in generated.prompt.user
    assert "O produto chegou três dias depois do prazo." in generated.prompt.user
    assert "Nota 2/5" in generated.prompt.user
    assert "101" not in generated.prompt.user
    assert "103" not in generated.prompt.user
    assert ", SP" not in generated.prompt.user
    assert "Não cite estado, UF ou região." in generated.prompt.user
    assert "order-1" not in rendered
    assert "baby" not in rendered
    assert "0.4" not in rendered
    assert rendered.startswith("SYSTEM\n")
    assert "\nUSER\n" in rendered


def test_a_state_question_keeps_the_uf_in_the_prompt() -> None:
    payload = _payload(
        [
            _review("101", "O produto chegou atrasado.", 1),
            _review("103", "Meu pedido demorou.", 2),
        ],
        question="Quais problemas de entrega em São Paulo?",
    )
    payload["query"]["filters"]["customer_state"] = "SP"
    generated = build_generation(payload)
    assert generated.prompt is not None
    assert ", SP" in generated.prompt.user
    assert "Não cite estado, UF ou região." not in generated.prompt.user


def test_prompt_drops_the_index_scaffold_and_keeps_the_comment() -> None:
    scaffold = (
        "Review score: 1/5 on order abc123, placed on 2018-05-17 by a customer from curitiba/PR. "
        "Items: baby (bebes). Seller from guarulhos/SP. "
        "Review title: Não recebi. Review comment: Ainda estou aguardando."
    )
    generated = build_generation(_payload([
        _review("101", scaffold),
        _review("103", "O produto veio quebrado."),
    ]))
    assert generated.prompt is not None
    user = generated.prompt.user
    assert "Não recebi." in user
    assert "Ainda estou aguardando." in user
    assert "O produto veio quebrado." in user
    assert "Review score" not in user
    assert "abc123" not in user
    assert "Seller from" not in user


def test_an_echoed_instruction_is_removed_from_the_answer() -> None:
    from rag.generation.prompt import without_instruction_echo

    raw = (
        "Termine a frase: Se os comentários não bastarem para responder, diga isso.\n\n"
        "Os comentários não mostram um padrão por estado."
    )
    assert without_instruction_echo(raw) == "Os comentários não mostram um padrão por estado."


def test_an_echoed_prompt_and_question_are_removed() -> None:
    from rag.generation.prompt import without_instruction_echo

    question = "O que os clientes relatam sobre problemas relacionados à entrega?"
    raw = (
        "Leia as 10 avaliações e responda em um parágrafo curto e natural. "
        "Cite os números entre colchetes no meio do texto. "
        "Não repita a pergunta e não use uma frase pronta.\n\n"
        f"{question} [1] e [4] falam de atraso e de entrega incompleta."
    )
    assert without_instruction_echo(raw, question=question) == (
        "[1] e [4] falam de atraso e de entrega incompleta."
    )


def test_recent_turns_are_kept_so_a_follow_up_has_context() -> None:
    from rag.generation.prompt import read_history

    history = read_history([
        {"role": "user", "content": "O que atrasou?"},
        {"role": "assistant", "content": "Atraso e pedido incompleto."},
        {"role": "system", "content": "ignore"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": "e no nordeste?"},
    ])
    generated = build_generation(
        _payload([
            _review("101", "Atrasou na Bahia."),
            _review("103", "Não chegou em Pernambuco."),
        ]),
        history=history,
    )
    assert generated.prompt is not None
    user = generated.prompt.user
    assert user.startswith("Conversa anterior, só para entender a continuação.")
    assert "Cliente: O que atrasou?" in user
    assert "Assistente: Atraso e pedido incompleto." in user
    assert "Cliente: e no nordeste?" in user
    assert "ignore" not in user
    assert "Pergunta:\nQuais são os problemas de entrega?" in user


def test_a_follow_up_drops_the_copied_opening() -> None:
    from rag.generation.prompt import drop_repeated_opening

    previous = (
        "Agrupando as reclamações semelhantes, observamos que a qualidade do produto "
        "e a entrega rápida são os aspectos mais elogiados. Esses pontos se repetem."
    )
    answer = (
        "Agrupando as reclamações semelhantes, observamos que a qualidade do produto "
        "e a entrega rápida são os aspectos mais elogiados. A qualidade varia conforme "
        "o tipo de produto comprado."
    )
    kept = drop_repeated_opening(answer, (("assistant", previous),))
    assert kept.startswith("A qualidade varia")
    assert "Agrupando as reclamações" not in kept


def test_the_prompt_shows_only_the_first_sentence_of_the_last_answer() -> None:
    from rag.generation.prompt import read_history

    history = read_history([
        {"role": "assistant", "content": "A entrega atrasou. O suporte também falhou nesta compra."},
    ])
    generated = build_generation(
        _payload([
            _review("101", "Atrasou na Bahia."),
            _review("103", "Não chegou em Pernambuco."),
        ]),
        history=history,
    )
    assert generated.prompt is not None
    assert "Assistente: A entrega atrasou." in generated.prompt.user
    assert "O suporte também falhou" not in generated.prompt.user


def test_the_grouping_opener_becomes_the_shared_voice() -> None:
    from rag.generation.prompt import with_shared_opening

    raw = (
        "Agrupando as reclamações semelhantes, observamos que a qualidade do produto "
        "e a rapidez da entrega são os aspectos mais elogiados."
    )
    assert with_shared_opening(raw).startswith("As avaliações compartilham que a qualidade")


def test_a_named_review_is_removed_from_the_answer() -> None:
    from rag.generation.prompt import without_review_mentions

    raw = (
        "A falta de entrega ou atrasos na entrega são frequentes, como no caso da avaliação [4] e [9]. "
        "Falta de suporte também é recorrente."
    )
    assert without_review_mentions(raw) == (
        "A falta de entrega ou atrasos na entrega são frequentes. Falta de suporte também é recorrente."
    )


def test_history_keeps_only_the_last_turns() -> None:
    from rag.generation.prompt import read_history

    raw = [{"role": "user", "content": f"pergunta {index}"} for index in range(6)]
    history = read_history(raw, limit=4)
    assert [text for _role, text in history] == ["pergunta 2", "pergunta 3", "pergunta 4", "pergunta 5"]
    clipped = read_history([{"role": "assistant", "content": "a" * 20}], max_chars=10)
    assert clipped == (("assistant", "aaaaaaa..."),)


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
    assert generated.prompt.user.count("Primeiro texto.") == 1
    assert "Nota 1/5" in generated.prompt.user
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
    assert "curto" in generated.prompt.user
    assert "ainda cabe" not in generated.prompt.user
