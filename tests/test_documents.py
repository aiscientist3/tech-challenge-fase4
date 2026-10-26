"""Rendered text and metadata for the fixture orders."""

from __future__ import annotations

import re

import pytest

from rag.config import MISSING_DELIVERY_DELAY_DAYS, MISSING_REVIEW_SCORE
from rag.ingestion.documents import build_documents, write_documents_parquet

FORBIDDEN = re.compile(r"\b(?:nan|none)\b", re.IGNORECASE)

LATE_TEXT = (
    "Order order-late placed on 2017-12-20 by a customer from sao paulo/SP. "
    "Status: delivered. "
    "Items: 1x furniture_decor (moveis_decoracao) sold by seller from belo horizonte/MG "
    "for R$ 30.00 + R$ 5.00 freight; "
    "1x bed_bath_table (cama_mesa_banho) sold by seller from campinas/SP "
    "for R$ 10.00 + R$ 2.00 freight. "
    "Payment: credit_card in 2 installments, total R$ 22.50; "
    "boleto in 1 installment, total R$ 28.10. "
    "Estimated delivery: 2018-01-05; delivered: 2018-01-10 (5 days late). "
    "Review score: 2/5. Review title: Atraso. Review comment: Entrega demorou muito. "
    "Review score: 5/5. Review comment: Produto bom."
)


def _one(documents, doc_type: str, **metadata):
    matches = [
        document
        for document in documents
        if document.metadata["doc_type"] == doc_type
        and all(document.metadata[key] == value for key, value in metadata.items())
    ]
    assert len(matches) == 1, matches
    return matches[0]


def test_late_order_is_a_single_document_with_payment_total(documents) -> None:
    orders = [document for document in documents if document.metadata["doc_type"] == "order"]
    late_orders = [document for document in orders if document.metadata["order_id"] == "order-late"]
    assert len(late_orders) == 1
    late = late_orders[0]
    assert late.page_content == LATE_TEXT
    assert late.metadata["total_value"] == pytest.approx(50.60)
    assert late.metadata["review_score"] == 2
    assert late.metadata["product_category"] == "furniture_decor"
    assert late.metadata["seller_state"] == "MG"
    assert late.metadata["customer_state"] == "SP"
    assert late.metadata["purchase_year"] == 2017
    assert late.metadata["purchase_month"] == 12
    assert late.metadata["delivery_delay_days"] == 5
    assert len(orders) == 3


def test_null_fields_are_omitted_and_metadata_has_no_none(documents) -> None:
    early = _one(documents, "order", order_id="order-early")
    opened = _one(documents, "order", order_id="order-open")
    assert "Review comment" not in early.page_content
    assert "Review title" not in early.page_content
    assert "Review score: 4/5." in early.page_content
    assert early.metadata["delivery_delay_days"] == -4
    assert "delivered:" not in opened.page_content
    assert "Estimated delivery: 2018-03-01." in opened.page_content
    assert "uncategorized product" in opened.page_content
    assert opened.metadata["customer_city"] == ""
    assert opened.metadata["product_category"] == ""
    assert opened.metadata["review_score"] == MISSING_REVIEW_SCORE
    assert opened.metadata["delivery_delay_days"] == MISSING_DELIVERY_DELAY_DAYS
    for document in documents:
        assert FORBIDDEN.search(document.page_content) is None
        for key, value in document.metadata.items():
            assert value is not None, key
            assert isinstance(value, (str, int, float, bool))
            if isinstance(value, str):
                assert FORBIDDEN.search(value) is None


def test_review_documents_keep_the_original_portuguese_comment(documents) -> None:
    reviews = [document for document in documents if document.metadata["doc_type"] == "review"]
    assert len(reviews) == 2
    first = _one(documents, "review", entity_id="rev-late-1")
    assert "Entrega demorou muito." in first.page_content
    assert "Atraso." in first.page_content
    assert first.metadata["review_score"] == 2
    second = _one(documents, "review", entity_id="rev-late-2")
    assert "Produto bom." in second.page_content
    assert "Review title" not in second.page_content


def test_order_only_selection_skips_reviews(transformed) -> None:
    documents = build_documents(transformed, {"order"})
    assert {document.metadata["doc_type"] for document in documents} == {"order"}


def test_parquet_export_matches_documents(tmp_path, documents) -> None:
    path = write_documents_parquet(documents, tmp_path)
    frame = __import__("pandas").read_parquet(path)
    assert len(frame) == len(documents)
    assert int(frame["text"].str.contains(r"\bnan\b", case=False, regex=True).sum()) == 0
    assert frame["review_score"].isna().sum() == 0
