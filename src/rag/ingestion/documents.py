"""Render natural-language documents and Chroma-safe metadata.

The sentence template is English so the fields are labeled consistently.
Portuguese category names and review comments are copied from the source
cells and are not translated again.
"""

from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from rag.config import (
    DOCUMENTS_FILENAME,
    DOC_TYPES,
    MISSING_DELIVERY_DELAY_DAYS,
    MISSING_REVIEW_SCORE,
)
from rag.ingestion.transform import (
    ItemGroup,
    OrderRecord,
    PaymentLine,
    ReviewLine,
    ReviewRecord,
    SummaryRecord,
    TransformedData,
)

logger = logging.getLogger(__name__)

ORDER_SOURCE_FILES = ",".join(
    (
        "olist_customers_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
        "product_category_name_translation.csv",
    )
)
REVIEW_SOURCE_FILES = ORDER_SOURCE_FILES
CATEGORY_SOURCE_FILES = ",".join(
    (
        "olist_order_items_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_products_dataset.csv",
        "product_category_name_translation.csv",
    )
)
STATE_SOURCE_FILES = ",".join(
    (
        "olist_customers_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_orders_dataset.csv",
    )
)

PARQUET_COLUMNS = (
    "text",
    "doc_type",
    "entity_id",
    "order_id",
    "customer_state",
    "customer_city",
    "seller_state",
    "product_category",
    "order_status",
    "purchase_year",
    "purchase_month",
    "review_score",
    "total_value",
    "delivery_delay_days",
    "source_files",
)


def build_documents(data: TransformedData, doc_types: set[str]) -> list[Document]:
    """Render the requested document types. Orders are always the main unit."""
    unknown = sorted(doc_types - set(DOC_TYPES))
    if unknown:
        raise ValueError(f"Unknown doc types: {', '.join(unknown)}")
    documents: list[Document] = []
    if "order" in doc_types:
        documents.extend(_order_document(order) for order in data.orders)
    if "review" in doc_types:
        documents.extend(_review_document(review) for review in data.reviews)
    if "category_summary" in doc_types:
        documents.extend(_category_document(summary) for summary in data.category_summaries)
    if "state_summary" in doc_types:
        documents.extend(_state_document(summary) for summary in data.state_summaries)
    logger.info("Rendered %s documents for types %s.", len(documents), sorted(doc_types))
    return documents


def write_documents_parquet(documents: list[Document], directory: Path) -> Path:
    """Save text and metadata before embedding so the transform can be inspected."""
    rows: list[dict[str, object]] = []
    for document in documents:
        row: dict[str, object] = {"text": document.page_content}
        row.update(document.metadata)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in PARQUET_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    frame = frame[list(PARQUET_COLUMNS)]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DOCUMENTS_FILENAME
    frame.to_parquet(path, index=False)
    logger.info("Wrote %s documents to %s.", len(frame), path)
    return path


def render_order(order: OrderRecord) -> str:
    """Render one order. Missing fields are omitted rather than printed as nan."""
    parts: list[str] = [_placed_sentence(order)]
    if order.status:
        parts.append(f"Status: {order.status}.")
    if order.items:
        rendered_items = "; ".join(render_item(item) for item in order.items)
        parts.append(f"Items: {rendered_items}.")
    if order.payments:
        rendered_payments = "; ".join(render_payment(payment) for payment in order.payments)
        parts.append(f"Payment: {rendered_payments}.")
    delivery = render_delivery(order)
    if delivery:
        parts.append(delivery)
    review_text = render_order_reviews(order.reviews)
    if review_text:
        parts.append(review_text)
    return _join(parts)


def render_item(item: ItemGroup) -> str:
    category = category_phrase(item.category_en, item.category_pt)
    seller_location = location_phrase(item.seller_city, item.seller_state)
    seller = f"sold by seller from {seller_location}" if seller_location else "sold by seller"
    price = format_money(item.unit_price)
    if item.quantity > 1:
        price = f"{price} each"
    return (
        f"{item.quantity}x {category} {seller} for {price} + "
        f"{format_money(item.freight_value)} freight"
    )


def render_payment(payment: PaymentLine) -> str:
    head: list[str] = []
    if payment.payment_type:
        head.append(payment.payment_type)
    if payment.installments is not None:
        noun = "installment" if payment.installments == 1 else "installments"
        head.append(f"in {payment.installments} {noun}")
    prefix = " ".join(head)
    total = f"total {format_money(payment.value)}"
    if prefix:
        return f"{prefix}, {total}"
    return total


def render_delivery(order: OrderRecord) -> str:
    estimated = format_date(order.estimated_delivery) if order.estimated_delivery is not None else None
    delivered = format_date(order.delivered_at) if order.delivered_at is not None else None
    if estimated and delivered and order.delivery_delay_days is not None:
        return (
            f"Estimated delivery: {estimated}; delivered: {delivered} "
            f"({delay_phrase(order.delivery_delay_days)})."
        )
    if estimated and delivered:
        return f"Estimated delivery: {estimated}; delivered: {delivered}."
    if delivered:
        return f"Delivered: {delivered}."
    if estimated:
        return f"Estimated delivery: {estimated}."
    return ""


def render_order_reviews(reviews: list[ReviewLine]) -> str:
    parts: list[str] = []
    for review in reviews:
        bits: list[str] = []
        if review.score is not None:
            bits.append(f"Review score: {review.score}/5.")
        if review.title:
            bits.append(f"Review title: {finish_sentence(review.title)}")
        if review.comment:
            bits.append(f"Review comment: {finish_sentence(review.comment)}")
        if bits:
            parts.append(" ".join(bits))
    return " ".join(parts)


def render_review(review: ReviewRecord) -> str:
    if review.score is not None:
        head = f"Review score: {review.score}/5 on order {review.order_id}"
    else:
        head = f"Review on order {review.order_id}"
    if review.purchase_at is not None:
        head += f", placed on {format_date(review.purchase_at)}"
    location = location_phrase(review.customer_city, review.customer_state)
    if location:
        head += f" by a customer from {location}"
    parts = [head + "."]
    labels = _unique_category_labels(review.categories)
    if labels:
        parts.append("Items: " + ", ".join(labels) + ".")
    seller_location = location_phrase(review.seller_city, review.seller_state)
    if seller_location:
        parts.append(f"Seller from {seller_location}.")
    if review.title:
        parts.append(f"Review title: {finish_sentence(review.title)}")
    # The comment is the reason this document exists, so it is always present.
    parts.append(f"Review comment: {finish_sentence(review.comment)}")
    return _join(parts)


def render_category_summary(summary: SummaryRecord) -> str:
    if summary.category_pt and summary.product_category not in {"", "uncategorized", summary.category_pt}:
        name = f"{summary.product_category} ({summary.category_pt})"
    elif summary.product_category and summary.product_category != "uncategorized":
        name = summary.product_category
    elif summary.category_pt:
        name = summary.category_pt
    else:
        name = "uncategorized product"
    return _render_summary(f"Product category {name}", summary, "average item price")


def render_state_summary(summary: SummaryRecord) -> str:
    state = summary.customer_state or "unknown"
    return _render_summary(f"Customer state {state}", summary, "average order value")


def category_phrase(category_en: str | None, category_pt: str | None) -> str:
    if category_en and category_pt:
        return f"{category_en} ({category_pt})"
    if category_en:
        return category_en
    if category_pt:
        return category_pt
    return "uncategorized product"


def location_phrase(city: str | None, state: str | None) -> str | None:
    if city and state:
        return f"{city}/{state}"
    return city or state


def format_money(value: float) -> str:
    return f"R$ {value:.2f}"


def format_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def delay_phrase(days: int) -> str:
    if days > 0:
        unit = "day" if days == 1 else "days"
        return f"{days} {unit} late"
    if days < 0:
        amount = abs(days)
        unit = "day" if amount == 1 else "days"
        return f"{amount} {unit} early"
    return "on time"


def finish_sentence(text: str) -> str:
    """Add a final period without changing a comment that already ends one."""
    if text[-1] in ".!?":
        return text
    return f"{text}."


def order_metadata(order: OrderRecord) -> dict[str, str | int | float]:
    scores = [review.score for review in order.reviews if review.score is not None]
    # The minimum keeps a bad review visible to a metadata filter even when
    # the same order also has a positive score. The text still lists each score.
    score = min(scores) if scores else None
    return _metadata(
        doc_type="order",
        entity_id=order.order_id,
        order_id=order.order_id,
        customer_state=order.customer_state,
        customer_city=order.customer_city,
        seller_state=order.seller_state,
        product_category=order.product_category,
        order_status=order.status,
        purchase_at=order.purchase_at,
        review_score=score,
        total_value=order.total_value,
        delivery_delay_days=order.delivery_delay_days,
        source_files=ORDER_SOURCE_FILES,
    )


def review_metadata(review: ReviewRecord) -> dict[str, str | int | float]:
    return _metadata(
        doc_type="review",
        entity_id=review.review_id,
        order_id=review.order_id,
        customer_state=review.customer_state,
        customer_city=review.customer_city,
        seller_state=review.seller_state,
        product_category=review.product_category,
        order_status=review.status,
        purchase_at=review.purchase_at,
        review_score=review.score,
        total_value=review.total_value,
        delivery_delay_days=review.delivery_delay_days,
        source_files=REVIEW_SOURCE_FILES,
    )


def summary_metadata(summary: SummaryRecord, source_files: str) -> dict[str, str | int | float]:
    return _metadata(
        doc_type=summary.doc_type,
        entity_id=summary.entity_id,
        order_id="",
        customer_state=summary.customer_state,
        customer_city=None,
        seller_state=None,
        product_category=summary.product_category,
        order_status=None,
        purchase_at=None,
        review_score=_round_half_up(summary.average_review_score),
        total_value=0.0 if summary.average_price is None else round(summary.average_price, 2),
        delivery_delay_days=_round_half_up(summary.average_delay_days),
        source_files=source_files,
    )


def _order_document(order: OrderRecord) -> Document:
    return Document(page_content=render_order(order), metadata=order_metadata(order))


def _review_document(review: ReviewRecord) -> Document:
    return Document(page_content=render_review(review), metadata=review_metadata(review))


def _category_document(summary: SummaryRecord) -> Document:
    return Document(
        page_content=render_category_summary(summary),
        metadata=summary_metadata(summary, CATEGORY_SOURCE_FILES),
    )


def _state_document(summary: SummaryRecord) -> Document:
    return Document(
        page_content=render_state_summary(summary),
        metadata=summary_metadata(summary, STATE_SOURCE_FILES),
    )


def _metadata(
    *,
    doc_type: str,
    entity_id: str,
    order_id: str,
    customer_state: str | None,
    customer_city: str | None,
    seller_state: str | None,
    product_category: str | None,
    order_status: str | None,
    purchase_at: pd.Timestamp | None,
    review_score: int | None,
    total_value: float,
    delivery_delay_days: int | None,
    source_files: str,
) -> dict[str, str | int | float]:
    if purchase_at is None or pd.isna(purchase_at):
        year = 0
        month = 0
    else:
        stamp = pd.Timestamp(purchase_at)
        year = int(stamp.year)
        month = int(stamp.month)
    # Plain Python scalars only. Numpy scalars and None are rejected by Chroma.
    return {
        "doc_type": doc_type,
        "entity_id": entity_id,
        "order_id": order_id or "",
        "customer_state": customer_state or "",
        "customer_city": customer_city or "",
        "seller_state": seller_state or "",
        "product_category": product_category or "",
        "order_status": order_status or "",
        "purchase_year": year,
        "purchase_month": month,
        "review_score": MISSING_REVIEW_SCORE if review_score is None else int(review_score),
        "total_value": float(round(total_value, 2)),
        "delivery_delay_days": (
            MISSING_DELIVERY_DELAY_DAYS if delivery_delay_days is None else int(delivery_delay_days)
        ),
        "source_files": source_files,
    }


def _render_summary(subject: str, summary: SummaryRecord, price_label: str) -> str:
    noun = "order" if summary.order_count == 1 else "orders"
    parts = [f"{subject}: {summary.order_count} {noun}."]
    if summary.average_review_score is not None:
        parts.append(f"Average review score: {summary.average_review_score:.2f}/5.")
    if summary.average_delay_days is not None:
        parts.append(f"Average delivery delay: {summary.average_delay_days:.1f} days.")
    if summary.average_price is not None:
        parts.append(f"{price_label.capitalize()}: {format_money(summary.average_price)}.")
    return _join(parts)


def _unique_category_labels(categories: list[tuple[str | None, str | None]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for category_en, category_pt in categories:
        label = category_phrase(category_en, category_pt)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _round_half_up(value: float | None) -> int | None:
    if value is None:
        return None
    quantized = Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(quantized)


def _placed_sentence(order: OrderRecord) -> str:
    if order.purchase_at is not None:
        sentence = f"Order {order.order_id} placed on {format_date(order.purchase_at)}"
    else:
        sentence = f"Order {order.order_id} placed"
    location = location_phrase(order.customer_city, order.customer_state)
    if location:
        sentence += f" by a customer from {location}"
    return sentence + "."


def _join(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())
