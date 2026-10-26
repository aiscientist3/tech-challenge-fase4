"""Ingestion package for the Olist RAG project.

Importing this package disables Chroma telemetry and tokenizer thread warnings
before either library is loaded. Those settings must be in place at import time.
"""

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
