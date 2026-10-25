"""Load the Olist CSV files with an explicit schema.

Geolocation is validated here and can be loaded for profiling, but the default
ingestion path does not read its rows. The file repeats coordinates for the
same zip prefix about a million times, while city and state already live on
the customer and seller tables that the document template uses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from rag.config import KAGGLE_DATASET_URL

logger = logging.getLogger(__name__)

# Column lists follow the public dataset, including the upstream typo
# `product_name_lenght`. Renaming it would make schema checks fail on the
# real file.
EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "olist_orders_dataset.csv": (
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ),
    "olist_order_items_dataset.csv": (
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ),
    "olist_order_payments_dataset.csv": (
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value",
    ),
    "olist_order_reviews_dataset.csv": (
        "review_id",
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message",
        "review_creation_date",
        "review_answer_timestamp",
    ),
    "olist_customers_dataset.csv": (
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ),
    "olist_sellers_dataset.csv": (
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    ),
    "olist_products_dataset.csv": (
        "product_id",
        "product_category_name",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ),
    "olist_geolocation_dataset.csv": (
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ),
    "product_category_name_translation.csv": (
        "product_category_name",
        "product_category_name_english",
    ),
}

TABLE_FILES: dict[str, str] = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

_DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "olist_orders_dataset.csv": (
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ),
    "olist_order_items_dataset.csv": ("shipping_limit_date",),
    "olist_order_reviews_dataset.csv": (
        "review_creation_date",
        "review_answer_timestamp",
    ),
}

# Zip prefixes are zero-padded strings in the CSV. Reading them as integers
# would turn 01001 into 1001 and split one prefix into two keys.
_DTYPES: dict[str, dict[str, str]] = {
    "olist_orders_dataset.csv": {
        "order_id": "string",
        "customer_id": "string",
        "order_status": "string",
    },
    "olist_order_items_dataset.csv": {
        "order_id": "string",
        "order_item_id": "Int64",
        "product_id": "string",
        "seller_id": "string",
        "price": "float64",
        "freight_value": "float64",
    },
    "olist_order_payments_dataset.csv": {
        "order_id": "string",
        "payment_sequential": "Int64",
        "payment_type": "string",
        "payment_installments": "Int64",
        "payment_value": "float64",
    },
    "olist_order_reviews_dataset.csv": {
        "review_id": "string",
        "order_id": "string",
        "review_score": "Int64",
        "review_comment_title": "string",
        "review_comment_message": "string",
    },
    "olist_customers_dataset.csv": {
        "customer_id": "string",
        "customer_unique_id": "string",
        "customer_zip_code_prefix": "string",
        "customer_city": "string",
        "customer_state": "string",
    },
    "olist_sellers_dataset.csv": {
        "seller_id": "string",
        "seller_zip_code_prefix": "string",
        "seller_city": "string",
        "seller_state": "string",
    },
    "olist_products_dataset.csv": {
        "product_id": "string",
        "product_category_name": "string",
        "product_name_lenght": "float64",
        "product_description_lenght": "float64",
        "product_photos_qty": "float64",
        "product_weight_g": "float64",
        "product_length_cm": "float64",
        "product_height_cm": "float64",
        "product_width_cm": "float64",
    },
    "olist_geolocation_dataset.csv": {
        "geolocation_zip_code_prefix": "string",
        "geolocation_lat": "float64",
        "geolocation_lng": "float64",
        "geolocation_city": "string",
        "geolocation_state": "string",
    },
    "product_category_name_translation.csv": {
        "product_category_name": "string",
        "product_category_name_english": "string",
    },
}


class DatasetValidationError(Exception):
    """Raised when raw files or columns do not match the Olist contract."""


def validate_raw_files(raw_dir: Path) -> None:
    """Check that every required CSV exists and has the expected columns.

    Only the header is read. Geolocation stays cheap to validate even though
    the file has about a million rows.
    """
    missing_files: list[str] = []
    column_problems: list[str] = []
    for filename, expected in EXPECTED_COLUMNS.items():
        path = raw_dir / filename
        if not path.is_file():
            missing_files.append(filename)
            continue
        try:
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            column_problems.append(f"{filename}: could not be read ({exc})")
            continue
        actual = {str(column).strip() for column in header.columns}
        missing = [column for column in expected if column not in actual]
        if missing:
            column_problems.append(
                f"{filename}: missing columns {', '.join(missing)}"
            )
    if missing_files or column_problems:
        raise DatasetValidationError(
            _validation_message(missing_files, column_problems)
        )


def load_raw_tables(
    raw_dir: Path,
    *,
    include_geolocation: bool = False,
) -> dict[str, pd.DataFrame]:
    """Validate and load the raw tables.

    `include_geolocation` is for the profiler. Ingestion leaves it false so
    the million-row coordinate file is never materialized.
    """
    validate_raw_files(raw_dir)
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILES.items():
        if key == "geolocation" and not include_geolocation:
            logger.info(
                "Geolocation file passed schema validation and will not be loaded. "
                "City and state already come from customers and sellers, and the "
                "file repeats coordinates for the same zip prefix."
            )
            continue
        if key == "geolocation":
            logger.info("Loading geolocation rows for profiling.")
        tables[key] = _read_table(raw_dir / filename, filename)
        logger.info("Loaded %s rows from %s.", len(tables[key]), filename)
    return tables


def _read_table(path: Path, filename: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            dtype=_DTYPES[filename],
            encoding="utf-8-sig",
        )
    except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
        raise DatasetValidationError(
            _validation_message([], [f"{filename}: could not be read ({exc})"])
        ) from exc
    for column in _DATE_COLUMNS.get(filename, ()):
        # Coerce unparseable dates to NaT instead of failing the whole file.
        # Downstream text rendering omits those fields.
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _validation_message(missing_files: list[str], column_problems: list[str]) -> str:
    lines = ["Olist raw dataset is incomplete or does not match the expected schema."]
    if missing_files:
        lines.append("Missing files:")
        lines.extend(f"- {name}" for name in missing_files)
    if column_problems:
        lines.append("Column problems:")
        lines.extend(f"- {problem}" for problem in column_problems)
    lines.append(
        "Download the CSV files from "
        f"{KAGGLE_DATASET_URL} and place them in the raw data directory."
    )
    return "\n".join(lines)
