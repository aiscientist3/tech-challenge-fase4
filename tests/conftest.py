"""Shared fixtures. Tests never read data/raw or download an embedding model."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pytest

from rag.config import Settings
from rag.ingestion.documents import build_documents
from rag.ingestion.extract import load_raw_tables
from rag.ingestion.transform import TransformedData, transform_tables

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        raw_dir=FIXTURES,
        processed_dir=tmp_path / "processed",
        chroma_dir=tmp_path / "chroma",
        collection_name="olist_test",
        batch_size=2,
        chunk_size=1000,
        chunk_overlap=150,
        random_seed=42,
        log_level="WARNING",
    )


@pytest.fixture(scope="session")
def transformed() -> TransformedData:
    return transform_tables(load_raw_tables(FIXTURES))


@pytest.fixture(scope="session")
def documents(transformed: TransformedData):
    return build_documents(
        transformed,
        {"order", "review", "category_summary", "state_summary"},
    )


def copy_fixtures(destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(FIXTURES, destination)
    return destination
