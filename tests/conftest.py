from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_frames(raw)
    return raw


def _write_frames(raw: Path) -> None:
    pd.DataFrame(
        [
            {
                "review_id": "rev-a",
                "order_id": "order-1",
                "review_score": 2,
                "review_comment_title": "Não recebi",
                "review_comment_message": "curto",
                "review_creation_date": "2018-01-10 00:00:00",
            },
            {
                "review_id": "rev-a",
                "order_id": "order-1",
                "review_score": 2,
                "review_comment_title": "Não recebi",
                "review_comment_message": "O prazo estourou e ninguém respondeu.",
                "review_creation_date": "2018-01-10 00:00:00",
            },
            {
                "review_id": "rev-b",
                "order_id": "order-2",
                "review_score": 1,
                "review_comment_title": None,
                "review_comment_message": "   ",
                "review_creation_date": "2018-01-02 00:00:00",
            },
            {
                "review_id": "rev-c",
                "order_id": "order-3",
                "review_score": 5,
                "review_comment_title": None,
                "review_comment_message": "Produto de ótima qualidade",
                "review_creation_date": "2018-02-02 00:00:00",
            },
            {
                "review_id": "rev-d",
                "order_id": "order-4",
                "review_score": 1,
                "review_comment_title": "Troca",
                "review_comment_message": "Atendimento ruim na troca",
                "review_creation_date": "2018-03-01 00:00:00",
            },
            {
                "review_id": "rev-e",
                "order_id": "order-5",
                "review_score": 3,
                "review_comment_title": "Dois itens",
                "review_comment_message": "Veio com defeito",
                "review_creation_date": "2018-04-01 00:00:00",
            },
            {
                "review_id": "rev-f",
                "order_id": "order-1",
                "review_score": None,
                "review_comment_title": "Sem nota",
                "review_comment_message": "Texto sem nota",
                "review_creation_date": "2018-01-12 00:00:00",
            },
            {
                "review_id": "rev-g",
                "order_id": "order-1",
                "review_score": 4,
                "review_comment_title": "  Atraso   na entrega  ",
                "review_comment_message": "  chegou   atrasado ",
                "review_creation_date": "2018-01-11 00:00:00",
            },
            {
                "review_id": "rev-h",
                "order_id": "order-3",
                "review_score": 5,
                "review_comment_title": "A",
                "review_comment_message": "mesmo texto",
                "review_creation_date": "2018-02-03 00:00:00",
            },
            {
                "review_id": "rev-h",
                "order_id": "order-1",
                "review_score": 5,
                "review_comment_title": "A",
                "review_comment_message": "mesmo texto",
                "review_creation_date": "2018-02-04 00:00:00",
            },
            {
                "review_id": "rev-i",
                "order_id": "order-9",
                "review_score": 4,
                "review_comment_title": None,
                "review_comment_message": "Chegou bem",
                "review_creation_date": "2018-05-03 00:00:00",
            },
        ]
    ).to_csv(raw / "olist_order_reviews_dataset.csv", index=False)

    pd.DataFrame(
        [
            _order("order-1", "cust-1", "2018-01-20 00:00:00", "2018-01-08 00:00:00"),
            _order("order-2", "cust-2", "2018-01-01 00:00:00", "2018-01-01 00:00:00"),
            _order("order-3", "cust-3", "2018-02-01 00:00:00", "2018-02-05 00:00:00"),
            _order("order-4", "cust-4", None, "2018-03-10 00:00:00"),
            _order("order-5", "cust-5", "2018-04-02 00:00:00", "2018-04-02 00:00:00"),
            _order("order-9", "cust-1", "2018-05-02 00:00:00", "2018-05-02 00:00:00"),
        ]
    ).to_csv(raw / "olist_orders_dataset.csv", index=False)

    pd.DataFrame(
        [
            {"customer_id": "cust-1", "customer_city": "sao paulo", "customer_state": "sp"},
            {"customer_id": "cust-2", "customer_city": "curitiba", "customer_state": "PR"},
            {"customer_id": "cust-3", "customer_city": "Rio de Janeiro", "customer_state": "RJ"},
            {"customer_id": "cust-4", "customer_city": None, "customer_state": None},
            {"customer_id": "cust-5", "customer_city": "Salvador", "customer_state": "BA"},
        ]
    ).to_csv(raw / "olist_customers_dataset.csv", index=False)

    pd.DataFrame(
        [
            {"order_id": "order-1", "product_id": "prod-1"},
            {"order_id": "order-2", "product_id": "prod-1"},
            {"order_id": "order-3", "product_id": "prod-2"},
            {"order_id": "order-4", "product_id": "prod-1"},
            {"order_id": "order-5", "product_id": "prod-1"},
            {"order_id": "order-5", "product_id": "prod-2"},
        ]
    ).to_csv(raw / "olist_order_items_dataset.csv", index=False)

    pd.DataFrame(
        [
            {"product_id": "prod-1", "product_category_name": "cama_mesa_banho"},
            {"product_id": "prod-2", "product_category_name": "beleza_saude"},
            {"product_id": "prod-3", "product_category_name": None},
        ]
    ).to_csv(raw / "olist_products_dataset.csv", index=False)

    pd.DataFrame(
        [
            {
                "product_category_name": "cama_mesa_banho",
                "product_category_name_english": "bed_bath_table",
            },
            {
                "product_category_name": "beleza_saude",
                "product_category_name_english": "health_beauty",
            },
        ]
    ).to_csv(raw / "product_category_name_translation.csv", index=False)


def _order(order_id: str, customer_id: str, delivered: str | None, estimated: str) -> dict:
    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "order_delivered_customer_date": delivered,
        "order_estimated_delivery_date": estimated,
    }
