"""Schema validation against the fixture contract, not the real dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rag.config import KAGGLE_DATASET_URL
from rag.ingestion.extract import DatasetValidationError, load_raw_tables, validate_raw_files
from tests.conftest import FIXTURES, copy_fixtures


def test_fixtures_match_the_contract() -> None:
    validate_raw_files(FIXTURES)
    tables = load_raw_tables(FIXTURES)
    assert "geolocation" not in tables
    assert len(tables["orders"]) == 3
    assert len(tables["order_items"]) == 4


def test_missing_file_names_the_file_and_the_download_link(tmp_path: Path) -> None:
    raw_dir = copy_fixtures(tmp_path / "raw")
    (raw_dir / "olist_orders_dataset.csv").unlink()
    with pytest.raises(DatasetValidationError, match="olist_orders_dataset.csv") as caught:
        validate_raw_files(raw_dir)
    message = str(caught.value)
    assert "Missing files:" in message
    assert KAGGLE_DATASET_URL in message


def test_missing_column_names_the_column_and_the_download_link(tmp_path: Path) -> None:
    raw_dir = copy_fixtures(tmp_path / "raw")
    orders_path = raw_dir / "olist_orders_dataset.csv"
    frame = pd.read_csv(orders_path)
    frame.drop(columns=["order_status"]).to_csv(orders_path, index=False)
    with pytest.raises(DatasetValidationError, match="order_status") as caught:
        validate_raw_files(raw_dir)
    message = str(caught.value)
    assert "olist_orders_dataset.csv" in message
    assert KAGGLE_DATASET_URL in message
