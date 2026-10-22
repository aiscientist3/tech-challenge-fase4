from pathlib import Path

import pytest

from voc_rag.embeddings import PASSAGE_PREFIX
from voc_rag.ingestion.prepare import (
    UNKNOWN_DELAY_DAYS,
    build_document_text,
    category_label,
    prepare_documents,
    read_parquet,
    write_parquet,
)

LATE_REVIEW = """Nota: 2/5
Categoria: cama_mesa_banho
UF: SP
Entrega: atrasada em 12 dias
Título: Não recebi
Comentário: O prazo estourou e ninguém respondeu."""


def test_prepare_keeps_longest_comment_and_drops_empty(raw_dir: Path):
    result = prepare_documents(raw_dir)

    assert result.rows_read == 11
    assert result.dropped_duplicates == 2
    assert result.dropped_empty == 1
    assert result.dropped_without_score == 1
    assert [document.review_id for document in result.documents] == [
        "rev-a",
        "rev-c",
        "rev-d",
        "rev-e",
        "rev-g",
        "rev-h",
        "rev-i",
    ]
    kept = result.documents[0]
    assert kept.document == LATE_REVIEW
    assert PASSAGE_PREFIX not in kept.document


def test_delivery_delay_and_missing_dates(raw_dir: Path):
    by_id = {document.review_id: document for document in prepare_documents(raw_dir).documents}

    late = by_id["rev-a"]
    assert late.delivery_delay_days == 12
    assert late.late_delivery is True
    assert late.customer_state == "SP"
    assert late.customer_city == "sao paulo"
    assert late.review_date == "2018-01-10"
    assert late.aspects == ""

    early = by_id["rev-c"]
    assert early.delivery_delay_days == -4
    assert early.late_delivery is False
    assert "Entrega: no prazo" in early.document
    assert "Título:" not in early.document
    assert early.aspects == "produto"

    unknown = by_id["rev-d"]
    assert unknown.delivery_delay_days == UNKNOWN_DELAY_DAYS
    assert unknown.late_delivery is False
    assert "Entrega:" not in unknown.document
    assert unknown.customer_state == "nao_informada"
    assert unknown.customer_city == "nao_informada"
    assert unknown.aspects == "atendimento"

    on_time = by_id["rev-e"]
    assert on_time.delivery_delay_days == 0
    assert on_time.late_delivery is False
    assert on_time.product_category == "beleza_saude, cama_mesa_banho"
    assert on_time.aspects == "produto"

    whitespace = by_id["rev-g"]
    assert "Título: Atraso na entrega" in whitespace.document
    assert "Comentário: chegou atrasado" in whitespace.document
    assert whitespace.aspects == "entrega"

    tie = by_id["rev-h"]
    assert tie.order_id == "order-3"
    assert tie.product_category == "beleza_saude"

    missing_items = by_id["rev-i"]
    assert missing_items.product_category == "nao_informada"
    assert "Entrega: no prazo" in missing_items.document


def test_document_text_uses_singular_day_and_omits_unknown_delivery():
    one_day = build_document_text(
        review_score=2,
        product_category="cama_mesa_banho",
        customer_state="SP",
        delivery_delay_days=1,
        review_title="Atraso",
        review_comment="A entrega atrasou.",
    )
    assert "Entrega: atrasada em 1 dia" in one_day

    unknown = build_document_text(
        review_score=3,
        product_category="nao_informada",
        customer_state="nao_informada",
        delivery_delay_days=UNKNOWN_DELAY_DAYS,
        review_title="",
        review_comment="Oi",
    )
    assert "Entrega:" not in unknown
    assert "Título:" not in unknown
    assert unknown.endswith("Comentário: Oi")


def test_category_label_prefers_portuguese_name():
    assert category_label("cama_mesa_banho", "bed_bath_table") == "cama_mesa_banho"
    assert category_label("", "health_beauty") == "health_beauty"
    assert category_label(None, None) == ""


def test_parquet_roundtrip_preserves_documents(raw_dir: Path, tmp_path: Path):
    documents = prepare_documents(raw_dir).documents
    path = tmp_path / "processed" / "reviews.parquet"
    write_parquet(documents, path)

    assert read_parquet(path) == documents


def test_missing_csv_names_the_file(tmp_path: Path):
    raw = tmp_path / "empty"
    raw.mkdir()
    with pytest.raises(FileNotFoundError, match="olist_order_reviews_dataset.csv"):
        prepare_documents(raw)
