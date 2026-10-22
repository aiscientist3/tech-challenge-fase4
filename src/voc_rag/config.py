from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "olist_reviews"
    embedding_model: str = "intfloat/multilingual-e5-small"
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    upsert_batch_size: int = 256
    embedding_batch_size: int = 64


def get_settings() -> Settings:
    return Settings()
