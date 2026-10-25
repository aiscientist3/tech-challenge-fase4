"""Join Olist tables into one record per order, plus optional summaries.

Raw rows embed poorly: an item line has no city, and a review line has no
items. The order is the unit that already ties those facts together. Child
rows are aggregated onto that order instead of becoming extra documents.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ItemGroup:
    """One product sold inside an order, after identical lines are grouped."""

    quantity: int
    category_pt: str | None
    category_en: str | None
    seller_city: str | None
    seller_state: str | None
    unit_price: float
    freight_value: float
    product_id: str


@dataclass(slots=True)
class PaymentLine:
    """One payment row, kept in `payment_sequential` order."""

    payment_type: str | None
    installments: int | None
    value: float


@dataclass(slots=True)
class ReviewLine:
    """One review attached to an order. Comment and title may be missing."""

    review_id: str
    score: int | None
    title: str | None
    comment: str | None
    created_at: pd.Timestamp | None


@dataclass(slots=True)
class OrderRecord:
    """Everything the order document and its metadata need."""

    order_id: str
    status: str | None
    purchase_at: pd.Timestamp | None
    customer_city: str | None
    customer_state: str | None
    estimated_delivery: pd.Timestamp | None
    delivered_at: pd.Timestamp | None
    delivery_delay_days: int | None
    item_count: int
    total_value: float
    items: list[ItemGroup]
    payments: list[PaymentLine]
    reviews: list[ReviewLine]
    product_category: str
    seller_state: str
    seller_city: str | None


@dataclass(slots=True)
class ReviewRecord:
    """A review that has a comment, enriched with its order context."""

    review_id: str
    order_id: str
    score: int | None
    title: str | None
    comment: str
    purchase_at: pd.Timestamp | None
    customer_city: str | None
    customer_state: str | None
    seller_city: str | None
    seller_state: str | None
    product_category: str
    categories: list[tuple[str | None, str | None]]
    total_value: float
    delivery_delay_days: int | None
    status: str | None


@dataclass(slots=True)
class SummaryRecord:
    """A short aggregate used for questions a single order cannot answer."""

    doc_type: str
    entity_id: str
    product_category: str
    category_pt: str | None
    customer_state: str
    order_count: int
    average_review_score: float | None
    average_delay_days: float | None
    average_price: float | None


@dataclass(slots=True)
class TransformedData:
    """All document sources produced from one load of the CSV files."""

    orders: list[OrderRecord]
    reviews: list[ReviewRecord]
    category_summaries: list[SummaryRecord]
    state_summaries: list[SummaryRecord]
    dataset_as_of: str


def clean_text(value: object) -> str | None:
    """Return a stripped string, or None when the cell is missing.

    Pandas stringifies missing values as ``nan`` or ``None``. Treating those
    tokens as empty keeps them out of the rendered text.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "<na>", "nat", "null"}:
        return None
    return text


def delivery_delay_days(
    delivered: pd.Timestamp | None,
    estimated: pd.Timestamp | None,
) -> int | None:
    """Calendar days from the estimated delivery date to the delivered date.

    Positive means late. The estimated column is a date stored at midnight, so
    the clock time on the delivered timestamp must not change the day count.
    None means the order has no comparable pair of dates.
    """
    if delivered is None or estimated is None:
        return None
    if pd.isna(delivered) or pd.isna(estimated):
        return None
    return int((pd.Timestamp(delivered).date() - pd.Timestamp(estimated).date()).days)


def sample_tables(
    tables: dict[str, pd.DataFrame],
    sample_size: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], int]:
    """Keep `sample_size` orders and the child rows that belong to them.

    The returned integer is the sample size stored on the collection.
    Zero is reserved for a run that did not pass ``--sample``.
    """
    if sample_size < 1:
        raise ValueError("sample size must be a positive integer")
    orders = tables["orders"]
    population = len(orders)
    if population == 0:
        raise ValueError("The orders table is empty.")
    if sample_size >= population:
        logger.warning(
            "Sample size %s covers all %s orders. The collection will record sample_size=%s.",
            sample_size,
            population,
            population,
        )
        chosen = orders.sort_values("order_id")
        recorded = population
    else:
        chosen = orders.sample(n=sample_size, random_state=seed).sort_values("order_id")
        recorded = sample_size
    order_ids = set(chosen["order_id"].astype(str))
    customer_ids = set(chosen["customer_id"].dropna().astype(str))
    sampled = dict(tables)
    sampled["orders"] = chosen.copy()
    for key in ("order_items", "payments", "reviews"):
        frame = tables[key]
        sampled[key] = frame[frame["order_id"].astype(str).isin(order_ids)].copy()
    customers = tables["customers"]
    sampled["customers"] = customers[customers["customer_id"].astype(str).isin(customer_ids)].copy()
    logger.info("Selected %s orders with random_state=%s.", recorded, seed)
    return sampled, recorded


def transform_tables(tables: dict[str, pd.DataFrame]) -> TransformedData:
    """Build order, review, and summary records from the loaded tables."""
    orders = _dedupe(tables["orders"], "order_id", "orders")
    customers = _dedupe(tables["customers"], "customer_id", "customers")
    products = _dedupe(tables["products"], "product_id", "products")
    sellers = _dedupe(tables["sellers"], "seller_id", "sellers")
    translation = _dedupe(
        tables["category_translation"],
        "product_category_name",
        "category translation",
    )
    items = _join_items(tables["order_items"], products, translation, sellers)
    payments = _clean_payments(tables["payments"])
    reviews = _clean_reviews(tables["reviews"])

    item_groups = _group_items(items)
    payment_lines = _group_payments(payments)
    review_lines = _group_reviews(reviews)
    order_records = _build_orders(orders, customers, item_groups, payment_lines, review_lines)
    review_records = _build_review_records(order_records)
    category_summaries = _category_summaries(items, order_records)
    state_summaries = _state_summaries(order_records)
    dataset_as_of = _dataset_as_of(orders)
    logger.info(
        "Transformed %s orders, %s review comments, %s category summaries, %s state summaries.",
        len(order_records),
        len(review_records),
        len(category_summaries),
        len(state_summaries),
    )
    return TransformedData(
        orders=order_records,
        reviews=review_records,
        category_summaries=category_summaries,
        state_summaries=state_summaries,
        dataset_as_of=dataset_as_of,
    )


def _dedupe(frame: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    duplicate_count = int(frame[key].duplicated().sum())
    if duplicate_count:
        # A repeated primary key would multiply totals during the joins.
        logger.warning("Dropping %s duplicate %s rows in %s. Keeping the first.", duplicate_count, key, label)
        return frame.drop_duplicates(key, keep="first").copy()
    return frame.copy()


def _join_items(
    items: pd.DataFrame,
    products: pd.DataFrame,
    translation: pd.DataFrame,
    sellers: pd.DataFrame,
) -> pd.DataFrame:
    product_cols = products[["product_id", "product_category_name"]]
    translation_cols = translation[["product_category_name", "product_category_name_english"]]
    seller_cols = sellers[["seller_id", "seller_city", "seller_state"]]
    # Left joins keep an item even when the product, translation, or seller is missing.
    # The text renderer drops the missing piece instead of emitting "nan".
    joined = items.merge(product_cols, on="product_id", how="left")
    joined = joined.merge(translation_cols, on="product_category_name", how="left")
    joined = joined.merge(seller_cols, on="seller_id", how="left")
    joined = joined[joined["order_id"].notna()].copy()
    null_prices = int(joined["price"].isna().sum())
    if null_prices:
        logger.warning("Replacing %s null item prices with 0 so order totals stay numeric.", null_prices)
    joined["price"] = joined["price"].fillna(0.0).round(2)
    joined["freight_value"] = joined["freight_value"].fillna(0.0)
    # Two category names in the public extract have no English row. Keeping the
    # Portuguese name avoids inventing a translation for those lines.
    missing_translation = joined.loc[
        joined["product_category_name"].notna() & joined["product_category_name_english"].isna(),
        "product_category_name",
    ]
    missing_names = sorted({name for name in missing_translation.tolist() if clean_text(name)})
    if missing_names:
        logger.info(
            "Categories without an English translation keep the Portuguese name: %s.",
            ", ".join(missing_names),
        )
    return joined


def _clean_payments(payments: pd.DataFrame) -> pd.DataFrame:
    frame = payments[payments["order_id"].notna()].copy()
    frame["payment_value"] = frame["payment_value"].fillna(0.0).round(2)
    return frame


def _clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    frame = reviews[reviews["order_id"].notna()].copy()
    duplicate_count = int(frame["review_id"].duplicated().sum())
    if duplicate_count:
        logger.warning("Dropping %s duplicate review_id rows. Keeping the first.", duplicate_count)
        frame = frame.drop_duplicates("review_id", keep="first")
    return frame


def _group_items(items: pd.DataFrame) -> dict[str, list[ItemGroup]]:
    grouped: dict[str, list[ItemGroup]] = defaultdict(list)
    if items.empty:
        return grouped
    # Olist stores one row per unit. Grouping equal product/seller/price lines
    # lets the text say "2x" without hiding a different price on another line.
    keys = ["order_id", "product_id", "seller_id", "price"]
    ordered = items.sort_values(["order_id", "order_item_id"], kind="mergesort")
    for key, part in ordered.groupby(keys, dropna=False, sort=False):
        order_id = clean_text(key[0])
        if order_id is None:
            continue
        first = part.iloc[0]
        # key is (order_id, product_id, seller_id, price). A null seller stays
        # in the group because dropna=False is set on the groupby.
        product_id = clean_text(key[1]) or ""
        unit_price = round(float(key[3]), 2)
        grouped[order_id].append(
            ItemGroup(
                quantity=int(len(part)),
                category_pt=clean_text(first["product_category_name"]),
                category_en=clean_text(first["product_category_name_english"]),
                seller_city=clean_text(first["seller_city"]),
                seller_state=clean_text(first["seller_state"]),
                unit_price=unit_price,
                freight_value=round(float(part["freight_value"].sum()), 2),
                product_id=product_id,
            )
        )
    for order_id, lines in grouped.items():
        lines.sort(key=lambda item: (-item.unit_price, item.product_id))
    return grouped


def _group_payments(payments: pd.DataFrame) -> dict[str, list[PaymentLine]]:
    grouped: dict[str, list[PaymentLine]] = defaultdict(list)
    if payments.empty:
        return grouped
    ordered = payments.sort_values(["order_id", "payment_sequential"], kind="mergesort")
    for row in ordered.itertuples(index=False):
        order_id = clean_text(row.order_id)
        if order_id is None:
            continue
        installments = None if pd.isna(row.payment_installments) else int(row.payment_installments)
        # A handful of payment rows store 0 installments, which is not a real plan.
        if installments is not None and installments <= 0:
            installments = None
        grouped[order_id].append(
            PaymentLine(
                payment_type=clean_text(row.payment_type),
                installments=installments,
                value=round(float(row.payment_value), 2),
            )
        )
    return grouped


def _group_reviews(reviews: pd.DataFrame) -> dict[str, list[ReviewLine]]:
    grouped: dict[str, list[ReviewLine]] = defaultdict(list)
    if reviews.empty:
        return grouped
    for row in reviews.itertuples(index=False):
        order_id = clean_text(row.order_id)
        review_id = clean_text(row.review_id)
        if order_id is None or review_id is None:
            continue
        score = None if pd.isna(row.review_score) else int(row.review_score)
        created = row.review_creation_date
        created_at = None if pd.isna(created) else pd.Timestamp(created)
        grouped[order_id].append(
            ReviewLine(
                review_id=review_id,
                score=score,
                title=clean_text(row.review_comment_title),
                comment=clean_text(row.review_comment_message),
                created_at=created_at,
            )
        )
    for lines in grouped.values():
        lines.sort(key=lambda review: (_timestamp_sort_key(review.created_at), review.review_id))
    return grouped


def _build_orders(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    item_groups: dict[str, list[ItemGroup]],
    payment_lines: dict[str, list[PaymentLine]],
    review_lines: dict[str, list[ReviewLine]],
) -> list[OrderRecord]:
    customer_cols = customers[
        ["customer_id", "customer_city", "customer_state"]
    ]
    merged = orders.merge(customer_cols, on="customer_id", how="left")
    merged = merged.sort_values("order_id", kind="mergesort")
    records: list[OrderRecord] = []
    for row in merged.itertuples(index=False):
        order_id = clean_text(row.order_id)
        if order_id is None:
            continue
        items = item_groups.get(order_id, [])
        payments = payment_lines.get(order_id, [])
        reviews = review_lines.get(order_id, [])
        purchase_at = _timestamp_or_none(row.order_purchase_timestamp)
        estimated = _timestamp_or_none(row.order_estimated_delivery_date)
        delivered = _timestamp_or_none(row.order_delivered_customer_date)
        representative = items[0] if items else None
        records.append(
            OrderRecord(
                order_id=order_id,
                status=clean_text(row.order_status),
                purchase_at=purchase_at,
                customer_city=clean_text(row.customer_city),
                customer_state=clean_text(row.customer_state),
                estimated_delivery=estimated,
                delivered_at=delivered,
                delivery_delay_days=delivery_delay_days(delivered, estimated),
                item_count=sum(item.quantity for item in items),
                total_value=_order_total(order_id, items, payments),
                items=items,
                payments=payments,
                reviews=reviews,
                product_category=_category_metadata(representative),
                seller_state=(representative.seller_state or "") if representative else "",
                seller_city=representative.seller_city if representative else None,
            )
        )
    return records


def _order_total(order_id: str, items: list[ItemGroup], payments: list[PaymentLine]) -> float:
    """Prefer the amount the customer paid.

    `payment_value` already includes freight and vouchers. Item price plus
    freight is only the fallback for an order that has no payment rows.
    """
    if payments:
        return round(sum(payment.value for payment in payments), 2)
    item_total = sum((item.unit_price * item.quantity) + item.freight_value for item in items)
    if not items:
        logger.debug("Order %s has neither payments nor items. Total is 0.", order_id)
    return round(item_total, 2)


def _category_metadata(item: ItemGroup | None) -> str:
    """English category used for exact metadata filters.

    Chroma metadata cannot store a list, so a multi-item order keeps the
    category of its most expensive line. The text still lists every line.
    """
    if item is None:
        return ""
    return item.category_en or item.category_pt or ""


def _build_review_records(orders: list[OrderRecord]) -> list[ReviewRecord]:
    records: list[ReviewRecord] = []
    seen: set[str] = set()
    for order in orders:
        categories = [(item.category_en, item.category_pt) for item in order.items]
        for review in order.reviews:
            if review.comment is None or review.review_id in seen:
                continue
            seen.add(review.review_id)
            records.append(
                ReviewRecord(
                    review_id=review.review_id,
                    order_id=order.order_id,
                    score=review.score,
                    title=review.title,
                    comment=review.comment,
                    purchase_at=order.purchase_at,
                    customer_city=order.customer_city,
                    customer_state=order.customer_state,
                    seller_city=order.seller_city,
                    seller_state=order.seller_state or None,
                    product_category=order.product_category,
                    categories=categories,
                    total_value=order.total_value,
                    delivery_delay_days=order.delivery_delay_days,
                    status=order.status,
                )
            )
    records.sort(key=lambda review: review.review_id)
    return records


def _category_summaries(items: pd.DataFrame, orders: list[OrderRecord]) -> list[SummaryRecord]:
    """One summary per category present on at least one item line.

    An order with two categories counts in both. Its reviews are averaged
    into each of those categories once, because the score belongs to the
    order rather than to a single line.
    """
    if items.empty:
        return []
    orders_by_id = {order.order_id: order for order in orders}
    prices: dict[str, list[float]] = defaultdict(list)
    order_ids: dict[str, set[str]] = defaultdict(set)
    portuguese_names: dict[str, list[str]] = defaultdict(list)
    for row in items.itertuples(index=False):
        order_id = clean_text(row.order_id)
        if order_id is None or order_id not in orders_by_id:
            continue
        category_pt = clean_text(row.product_category_name)
        category_en = clean_text(row.product_category_name_english)
        key = category_en or category_pt or "uncategorized"
        prices[key].append(float(row.price))
        order_ids[key].add(order_id)
        if category_pt:
            portuguese_names[key].append(category_pt)
    summaries: list[SummaryRecord] = []
    for key in sorted(order_ids):
        related = [orders_by_id[order_id] for order_id in order_ids[key]]
        pt_name = _most_common(portuguese_names.get(key, []))
        summaries.append(
            SummaryRecord(
                doc_type="category_summary",
                entity_id=key,
                product_category=key,
                category_pt=pt_name,
                customer_state="",
                order_count=len(related),
                average_review_score=_mean_scores(related),
                average_delay_days=_mean_delays(related),
                average_price=_mean(prices[key]),
            )
        )
    return summaries


def _state_summaries(orders: list[OrderRecord]) -> list[SummaryRecord]:
    grouped: dict[str, list[OrderRecord]] = defaultdict(list)
    skipped = 0
    for order in orders:
        if not order.customer_state:
            skipped += 1
            continue
        grouped[order.customer_state].append(order)
    if skipped:
        logger.info("Skipping %s orders with no customer state in state summaries.", skipped)
    summaries: list[SummaryRecord] = []
    for state in sorted(grouped):
        related = grouped[state]
        summaries.append(
            SummaryRecord(
                doc_type="state_summary",
                entity_id=state,
                product_category="",
                category_pt=None,
                customer_state=state,
                order_count=len(related),
                average_review_score=_mean_scores(related),
                average_delay_days=_mean_delays(related),
                average_price=_mean([order.total_value for order in related]),
            )
        )
    return summaries


def _mean_scores(orders: list[OrderRecord]) -> float | None:
    scores = [review.score for order in orders for review in order.reviews if review.score is not None]
    return _mean([float(score) for score in scores])


def _mean_delays(orders: list[OrderRecord]) -> float | None:
    delays = [order.delivery_delay_days for order in orders if order.delivery_delay_days is not None]
    return _mean([float(delay) for delay in delays])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _most_common(values: list[str]) -> str | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _dataset_as_of(orders: pd.DataFrame) -> str:
    if orders.empty or "order_purchase_timestamp" not in orders.columns:
        return ""
    latest = orders["order_purchase_timestamp"].max()
    if pd.isna(latest):
        return ""
    return pd.Timestamp(latest).isoformat()


def _timestamp_or_none(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value)


def _timestamp_sort_key(value: pd.Timestamp | None) -> tuple[int, float]:
    if value is None or pd.isna(value):
        return (1, 0.0)
    return (0, float(pd.Timestamp(value).timestamp()))
