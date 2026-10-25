"""Runtime configuration for extraction and ingestion.

Environment variables and an optional `.env` file override the defaults.
Paths are resolved from the repository root unless they are already absolute,
so scripts behave the same regardless of the caller's working directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Sentinels are real metadata values because Chroma rejects None.
# -9999 is far outside a delivery delay measured in days, so a later filter
# can separate "not delivered" from an order that was merely early.
MISSING_REVIEW_SCORE = -1
MISSING_DELIVERY_DELAY_DAYS = -9999

DOCUMENTS_FILENAME = "documents.parquet"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data"

# Orders are one document each. These names are the only doc_type values
# the indexer and the future retrieval layer should see.
DOC_TYPES = ("order", "review", "category_summary", "state_summary")


def project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Ingestion settings.

    The default embedding model is multilingual because review text is
    Brazilian Portuguese. Swap `embedding_model` or `embedding_provider`
    and recreate the collection with `--reset`; mixed vector spaces are refused.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    chroma_dir: Path = Path("data/chroma")

    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    batch_size: int = 500
    # Window of deterministic chunk ids deleted on re-index so a shorter
    # document does not leave orphan chunks behind. Order text will not
    # approach 20 chunks of `chunk_size` characters.
    max_extra_chunks: int = 20
    collection_name: str = "olist_ecommerce"

    enable_review_documents: bool = True
    enable_aggregate_documents: bool = True

    random_seed: int = 42
    log_level: str = "INFO"
    dataset_version: str = "brazilian-ecommerce-olist"

    missing_review_score: int = Field(default=MISSING_REVIEW_SCORE)
    missing_delivery_delay_days: int = Field(default=MISSING_DELIVERY_DELAY_DAYS)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_extra_chunks < 0:
            raise ValueError("max_extra_chunks must be >= 0")
        if not self.collection_name.strip():
            raise ValueError("collection_name must not be empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must not be empty")
        return self

    def resolve_path(self, path: Path) -> Path:
        """Resolve `path` against the repository root when it is relative."""
        if path.is_absolute():
            return path
        return project_root() / path


def configure_logging(level: str) -> None:
    """Configure process-wide logging once from a script entry point."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
