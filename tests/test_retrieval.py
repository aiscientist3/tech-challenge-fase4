"""Retrieval and selection without a generative model or a downloaded embedder."""

from __future__ import annotations

import pytest

from rag.ingestion.chunking import chunk_documents
from rag.ingestion.indexer import CollectionConfigError, document_id, index_documents
from rag.retrieval.filters import RetrievalFilters, to_chroma_where
from rag.retrieval.hybrid import hybrid_score
from rag.retrieval.interpret import interpret_question, merge_filters
from rag.retrieval.payload import to_payload
from rag.retrieval.retrieve import prepare_question, retrieve
from rag.retrieval.select import RetrievedDocument, SelectionPolicy, select_documents
from tests.fake_embeddings import FakeEmbeddings


def test_where_combines_score_bounds_and_category() -> None:
    where = to_chroma_where(
        RetrievalFilters(
            doc_type="review",
            review_score_min=1,
            review_score_max=2,
            product_category="bed_bath_table",
            purchase_year=2018,
            customer_state="sp",
        )
    )
    assert where == {
        "$and": [
            {"doc_type": "review"},
            {"product_category": "bed_bath_table"},
            {"purchase_year": 2018},
            {"customer_state": "SP"},
            {"review_score": {"$in": [1, 2]}},
        ]
    }


def test_empty_filters_do_not_constrain_the_search() -> None:
    assert to_chroma_where(RetrievalFilters()) is None
    assert to_chroma_where(None) is None


def test_score_range_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="review_score_min"):
        RetrievalFilters(review_score_min=4, review_score_max=2)


def test_selection_drops_far_duplicate_and_extra_chunks() -> None:
    policy = SelectionPolicy(candidate_k=10, k=2, max_distance=0.50, min_keep=2)
    close = _document("close", distance=0.20, order_id="order-1", entity_id="rev-1")
    same_order = _document("same-order", distance=0.30, order_id="order-1", entity_id="rev-2")
    same_entity = _document("same-entity", distance=0.25, order_id="order-1", entity_id="rev-1")
    other = _document("other", distance=0.40, order_id="order-2", entity_id="rev-3")
    extra = _document("extra", distance=0.45, order_id="order-3", entity_id="rev-4")
    far = _document("far", distance=0.80, order_id="order-4", entity_id="rev-5")
    selected, dropped = select_documents(
        [far, extra, other, same_order, same_entity, close],
        policy,
    )
    assert [document.id for document in selected] == ["close", "other"]
    reasons = {item.document.id: item.reason for item in dropped}
    assert reasons == {
        "same-entity": "duplicate_chunk",
        "same-order": "duplicate_order",
        "extra": "beyond_k",
        "far": "distance",
    }


def test_hybrid_score_trusts_the_vector_and_still_lifts_a_real_answer() -> None:
    assert hybrid_score(0.20, 0.5) > hybrid_score(0.50, 0.5)
    assert hybrid_score(0.55, 0.95) > hybrid_score(0.40, 0.05)


def test_a_comment_that_answers_ranks_above_a_nearer_one() -> None:
    policy = SelectionPolicy(candidate_k=5, k=2, max_distance=0.65, min_keep=1)
    praise = _document(
        "praise",
        distance=0.40,
        order_id="order-1",
        entity_id="rev-1",
        text="Gostei muito do produto, ficou ótimo.",
    )
    delay = _document(
        "delay",
        distance=0.55,
        order_id="order-2",
        entity_id="rev-2",
        text="A entrega atrasou e o pedido demorou.",
    )
    selected, _dropped = select_documents(
        [praise, delay],
        policy,
        question="O que os clientes relatam sobre atraso na entrega?",
        score_passages=_answer_score,
    )
    assert [document.id for document in selected] == ["delay", "praise"]
    assert selected[0].relevance == pytest.approx(0.95)


def test_a_low_score_stays_when_the_list_still_has_room() -> None:
    policy = SelectionPolicy(candidate_k=5, k=2, max_distance=0.65, min_keep=1)
    praise = _document(
        "praise",
        distance=0.40,
        order_id="order-1",
        entity_id="rev-1",
        text="A entrega chegou antes do prazo e o produto é ótimo.",
    )
    delay = _document(
        "delay",
        distance=0.55,
        order_id="order-2",
        entity_id="rev-2",
        text="A entrega atrasou e o pedido não chegou no prazo.",
    )
    selected, dropped = select_documents(
        [praise, delay],
        policy,
        question="O que os clientes relatam sobre atraso na entrega?",
        score_passages=_answer_score,
    )
    assert [document.id for document in selected] == ["delay", "praise"]
    assert dropped == []


def test_a_comment_that_answers_beats_a_nearer_miss() -> None:
    policy = SelectionPolicy(candidate_k=5, k=3, max_distance=0.65, min_keep=1)
    contact = _document(
        "contact",
        distance=0.30,
        order_id="order-1",
        entity_id="rev-1",
        text="Entrei em contato e recebi o produto certo.",
    )
    on_time = _document(
        "on-time",
        distance=0.40,
        order_id="order-2",
        entity_id="rev-2",
        text="Gostei muito do produto, a entrega dentro do prazo.",
    )
    missing = _document(
        "missing",
        distance=0.62,
        order_id="order-3",
        entity_id="rev-3",
        text="O produto não foi entregue. Não fiquei satisfeito com a entrega.",
    )
    selected, _dropped = select_documents(
        [contact, on_time, missing],
        policy,
        question="O que os clientes relatam sobre atraso na entrega?",
        score_passages=_answer_score,
    )
    assert [document.id for document in selected] == ["missing", "on-time", "contact"]


def test_default_search_reads_forty_candidates() -> None:
    assert SelectionPolicy().candidate_k == 40


def test_summaries_without_an_order_id_stay_distinct() -> None:
    policy = SelectionPolicy(candidate_k=5, k=2, max_distance=0.65, min_keep=1)
    first = _document("cat-a", distance=0.10, order_id="", entity_id="bed_bath_table", doc_type="category_summary")
    second = _document("cat-b", distance=0.20, order_id="", entity_id="health_beauty", doc_type="category_summary")
    selected, dropped = select_documents([first, second], policy)
    assert [document.id for document in selected] == ["cat-a", "cat-b"]
    assert dropped == []


def test_exact_review_text_is_the_only_hit_inside_a_tight_distance(settings, documents) -> None:
    chunks = chunk_documents(documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    review = next(
        chunk
        for chunk in chunks
        if chunk.metadata["doc_type"] == "review" and int(chunk.metadata["review_score"]) <= 2
    )
    embeddings = FakeEmbeddings()
    index_documents(
        chunks,
        settings,
        embeddings,
        reset=True,
        sample_size=0,
        dataset_as_of="2018-02-20T09:00:00",
    )
    result = retrieve(
        review.page_content,
        settings,
        embeddings,
        filters=RetrievalFilters(doc_type="review", review_score_max=2),
        policy=SelectionPolicy(candidate_k=10, k=5, max_distance=1e-6, min_keep=1),
    )
    assert result.insufficient_evidence is False
    assert len(result.selected) == 1
    assert result.selected[0].id == document_id("review", str(review.metadata["entity_id"]), 0)
    assert result.selected[0].distance == pytest.approx(0.0, abs=1e-6)
    assert result.selected[0].review_score <= 2
    assert "Entrega demorou muito" in result.selected[0].text


def test_unknown_category_has_no_evidence(settings, documents) -> None:
    chunks = chunk_documents(documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    embeddings = FakeEmbeddings()
    index_documents(
        chunks,
        settings,
        embeddings,
        reset=True,
        sample_size=0,
        dataset_as_of="2018-02-20T09:00:00",
    )
    result = retrieve(
        "atraso na entrega",
        settings,
        embeddings,
        filters=RetrievalFilters(product_category="categoria_inexistente"),
    )
    assert result.candidate_count == 0
    assert result.selected == []
    assert result.insufficient_evidence is True


def test_query_refuses_a_different_embedding_model(settings, documents) -> None:
    chunks = chunk_documents(documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    index_documents(
        chunks,
        settings,
        FakeEmbeddings(),
        reset=True,
        sample_size=0,
        dataset_as_of="2018-02-20T09:00:00",
    )
    other = settings.model_copy(update={"embedding_model": "other-model"})
    with pytest.raises(CollectionConfigError, match="embedding model"):
        retrieve("atraso na entrega", other, FakeEmbeddings())


def test_prepare_question_keeps_the_original_and_collapses_whitespace() -> None:
    prepared = prepare_question("  atraso   na\nentrega ")
    assert prepared.original_question == "  atraso   na\nentrega "
    assert prepared.normalized_question == "atraso na entrega"


def test_prepare_question_rejects_a_blank_question() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        prepare_question(" \n\t ")


def test_high_and_low_scores_are_not_one_band() -> None:
    from rag.retrieval.interpret import contrasts_high_and_low

    question = "por que algumas entregas tem nota alta e outras nota baixa?"
    filters = interpret_question(question)
    assert filters.review_score_min is None
    assert filters.review_score_max is None
    assert contrasts_high_and_low(question) is True
    assert contrasts_high_and_low("quais são os motivos de nota baixa?") is False


def test_comparison_questions_keep_a_wider_neighborhood() -> None:
    assert SelectionPolicy().max_distance == 0.75


def test_a_region_becomes_the_states_in_that_region() -> None:
    filters = interpret_question("no nordeste as entregas atrasam?")
    assert filters.customer_state is None
    assert filters.review_score_min is None
    assert filters.customer_states == ("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE")
    assert to_chroma_where(filters) == {
        "customer_state": {"$in": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]}
    }


def test_a_comparison_searches_every_named_region() -> None:
    filters = interpret_question(
        "as regioes norte e nordeste tem mais problemas na entrega que regioes sul e sudeste?"
    )
    assert filters.review_score_min == 1
    assert filters.review_score_max == 3
    assert filters.customer_states is not None
    for uf in ("PA", "BA", "SP", "RS"):
        assert uf in filters.customer_states


def test_a_named_state_wins_over_a_region_word() -> None:
    filters = interpret_question("em são paulo as entregas atrasam?")
    assert filters.customer_state == "SP"
    assert filters.customer_states is None


def test_negative_reviews_in_sao_paulo_become_score_and_state_filters() -> None:
    filters = interpret_question("Em São Paulo quais são os motivos de nota baixa?")
    assert filters.customer_state == "SP"
    assert filters.review_score_min == 1
    assert filters.review_score_max == 3


def test_negative_evaluations_set_the_low_score_range() -> None:
    filters = interpret_question("Avaliações negativas sobre entrega")
    assert filters.review_score_min == 1
    assert filters.review_score_max == 3
    assert filters.customer_state is None


def test_asking_for_problems_keeps_the_low_scores() -> None:
    filters = interpret_question("Quais são os principais problemas relatados pelos clientes?")
    assert filters.review_score_min == 1
    assert filters.review_score_max == 3


def test_sem_problemas_does_not_force_a_low_score() -> None:
    filters = interpret_question("Quais elogios de quem comprou sem problemas?")
    assert filters.review_score_min == 4
    assert filters.review_score_max == 5


def test_delivery_delay_does_not_invent_a_score_filter() -> None:
    filters = interpret_question("O que os clientes relatam sobre atraso na entrega?")
    assert filters.review_score_min is None
    assert filters.review_score_max is None
    assert filters.customer_state is None


def test_para_the_word_is_not_the_state_of_para() -> None:
    filters = interpret_question("O que falta para a entrega chegar?")
    assert filters.customer_state is None


def test_praise_in_minas_gerais_sets_a_high_score() -> None:
    filters = interpret_question("Quais elogios aparecem em Minas Gerais?")
    assert filters.customer_state == "MG"
    assert filters.review_score_min == 4
    assert filters.review_score_max == 5


def test_explicit_filter_wins_over_the_question() -> None:
    merged = merge_filters(
        RetrievalFilters(doc_type="review", review_score_max=2),
        interpret_question("Avaliações negativas em São Paulo"),
    )
    assert merged.doc_type == "review"
    assert merged.review_score_max == 2
    assert merged.review_score_min == 1
    assert merged.customer_state == "SP"


def test_low_scores_in_sao_paulo_exclude_a_high_score_review(settings, documents) -> None:
    chunks = chunk_documents(documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    embeddings = FakeEmbeddings()
    index_documents(
        chunks,
        settings,
        embeddings,
        reset=True,
        sample_size=0,
        dataset_as_of="2018-02-20T09:00:00",
    )
    question = "  em São Paulo quais são os motivos de nota baixa?  "
    result = retrieve(
        question,
        settings,
        embeddings,
        filters=RetrievalFilters(doc_type="review"),
        policy=SelectionPolicy(candidate_k=20, k=10, max_distance=2.0, min_keep=1),
    )
    assert result.original_question == question
    assert result.question == "em São Paulo quais são os motivos de nota baixa?"
    assert result.filters.customer_state == "SP"
    assert result.filters.review_score_max == 3
    assert result.selected
    assert {item.review_score for item in result.selected} <= {1, 2, 3}
    assert {item.customer_state for item in result.selected} == {"SP"}
    payload = to_payload(result)
    assert payload["query"]["original_question"] == question
    assert payload["query"]["filters"]["customer_state"] == "SP"
    assert payload["query"]["filters"]["review_score_max"] == 3
    first = payload["retrieved_reviews"][0]
    assert first["similarity"] == pytest.approx(1 - result.selected[0].distance, abs=1e-4)
    assert first["text"] == result.selected[0].text


def _document(
    chunk_id: str,
    *,
    distance: float,
    order_id: str,
    entity_id: str,
    doc_type: str = "review",
    text: str | None = None,
    relevance: float = 0.0,
) -> RetrievedDocument:
    return RetrievedDocument(
        id=chunk_id,
        text=chunk_id if text is None else text,
        distance=distance,
        doc_type=doc_type,
        entity_id=entity_id,
        order_id=order_id,
        review_score=1,
        product_category="bed_bath_table",
        customer_state="SP",
        purchase_year=2018,
        purchase_month=1,
        relevance=relevance,
    )


def _answer_score(_question: str, texts: list[str]) -> list[float]:
    """Stand-in for the cross-encoder. A delay comment answers; a side remark does not."""
    scores: list[float] = []
    for text in texts:
        folded = text.casefold()
        if "não foi entregue" in folded or "atrasou" in folded or "demorou" in folded:
            scores.append(0.95)
        elif "dentro do prazo" in folded:
            scores.append(0.35)
        else:
            scores.append(0.05)
    return scores
