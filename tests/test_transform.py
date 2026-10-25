"""Joins, totals, and delivery delay on the fixture orders."""

from __future__ import annotations

import pandas as pd
import pytest

from rag.config import MISSING_DELIVERY_DELAY_DAYS
from rag.ingestion.extract import load_raw_tables
from rag.ingestion.transform import delivery_delay_days, sample_tables, transform_tables
from tests.conftest import FIXTURES


def test_delivery_delay_uses_calendar_dates() -> None:
    late = delivery_delay_days(pd.Timestamp("2018-01-10 16:00:00"), pd.Timestamp("2018-01-05 00:00:00"))
    early = delivery_delay_days(pd.Timestamp("2018-02-01 12:00:00"), pd.Timestamp("2018-02-05 00:00:00"))
    missing = delivery_delay_days(None, pd.Timestamp("2018-03-01 00:00:00"))
    assert late == 5
    assert early == -4
    assert missing is None


def test_two_items_and_two_payments_stay_one_order(transformed) -> None:
    matches = [order for order in transformed.orders if order.order_id == "order-late"]
    assert len(matches) == 1
    order = matches[0]
    assert order.item_count == 2
    assert len(order.payments) == 2
    assert order.total_value == pytest.approx(50.60)
    assert order.delivery_delay_days == 5
    assert order.product_category == "furniture_decor"
    assert order.seller_state == "MG"
    scores = [review.score for review in order.reviews]
    assert scores == [2, 5]


def test_early_and_undelivered_orders(transformed) -> None:
    by_id = {order.order_id: order for order in transformed.orders}
    early = by_id["order-early"]
    assert early.delivery_delay_days == -4
    assert early.reviews[0].comment is None
    assert early.reviews[0].title is None
    assert early.total_value == pytest.approx(16.00)
    opened = by_id["order-open"]
    assert opened.delivery_delay_days is None
    assert opened.reviews == []
    assert opened.payments == []
    assert opened.total_value == pytest.approx(9.00)
    assert opened.customer_city is None
    assert opened.customer_state == "RJ"
    assert opened.product_category == ""


def test_only_reviews_with_comments_become_review_records(transformed) -> None:
    assert [review.review_id for review in transformed.reviews] == ["rev-late-1", "rev-late-2"]
    assert transformed.reviews[0].comment == "Entrega demorou muito"


def test_category_and_state_averages(transformed) -> None:
    categories = {summary.entity_id: summary for summary in transformed.category_summaries}
    bed = categories["bed_bath_table"]
    assert bed.order_count == 2
    assert bed.average_price == pytest.approx(12.50)
    assert bed.average_review_score == pytest.approx((2 + 5 + 4) / 3)
    assert bed.average_delay_days == pytest.approx(0.5)
    furniture = categories["furniture_decor"]
    assert furniture.order_count == 1
    assert furniture.average_price == pytest.approx(30.00)
    assert furniture.average_review_score == pytest.approx(3.5)
    unknown = categories["uncategorized"]
    assert unknown.average_review_score is None
    assert unknown.average_delay_days is None
    states = {summary.customer_state: summary for summary in transformed.state_summaries}
    assert states["SP"].average_price == pytest.approx(50.60)
    assert states["RJ"].average_delay_days is None
    assert states["RJ"].order_count == 1
    assert MISSING_DELIVERY_DELAY_DAYS == -9999


def test_sample_is_stable(transformed) -> None:
    tables = load_raw_tables(FIXTURES)
    first, recorded = sample_tables(tables, 2, seed=42)
    second, _ = sample_tables(tables, 2, seed=42)
    assert recorded == 2
    assert first["orders"]["order_id"].tolist() == second["orders"]["order_id"].tolist()
    assert set(first["order_items"]["order_id"]) <= set(first["orders"]["order_id"])
    assert len(transformed.orders) == 3
