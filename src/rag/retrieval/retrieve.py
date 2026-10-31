"""Embed a question and read the nearest chunks from the persistent collection.

The embedding function has to be the one that indexed the collection. The
indexer stores those vectors directly, so this module embeds the question
itself and passes the vector to Chroma instead of asking the collection to
embed it with a different function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb
from langchain_core.embeddings import Embeddings

from rag.config import MISSING_REVIEW_SCORE, Settings
from rag.ingestion.indexer import (
    CollectionConfigError,
    chroma_client,
    collection_names,
)
from rag.retrieval.filters import RetrievalFilters, to_chroma_where
from rag.retrieval.interpret import interpret_question, merge_filters
from rag.retrieval.select import (
    RetrievalResult,
    RetrievedDocument,
    SelectionPolicy,
    select_documents,
)

logger = logging.getLogger(__name__)


class CollectionNotFoundError(Exception):
    """Raised when the configured Chroma directory has no collection to query."""


@dataclass(frozen=True, slots=True)
class PreparedQuestion:
    """The user text and the cleaned text that will be embedded."""

    original_question: str
    normalized_question: str


def prepare_question(question: str) -> PreparedQuestion:
    """Trim and collapse whitespace. The original text stays intact.

    Score and state filters are not decided here. `interpret_question` reads
    the normalized text for that.
    """
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("The question must not be empty.")
    return PreparedQuestion(original_question=question, normalized_question=normalized)


def retrieve(
    question: str,
    settings: Settings,
    embeddings: Embeddings,
    *,
    filters: RetrievalFilters | None = None,
    policy: SelectionPolicy | None = None,
) -> RetrievalResult:
    """Embed one question and return the reviews closest to it.

    The question is prepared, embedded with the same model that indexed the
    collection, and compared with the stored vectors. Filters named in the
    question (a low score, a customer state) are applied inside that query
    unless the caller already set the same field. Selection then keeps the
    nearest eligible chunks. `insufficient_evidence` is true when fewer than
    `policy.min_keep` chunks remain. No generative model is called.
    """
    prepared = prepare_question(question)
    active_filters = merge_filters(filters, interpret_question(prepared.normalized_question))
    active_policy = policy if policy is not None else SelectionPolicy()
    collection = open_collection(settings)
    candidates = search_collection(
        collection,
        embeddings,
        prepared.normalized_question,
        filters=active_filters,
        candidate_k=active_policy.candidate_k,
    )
    selected, dropped = select_documents(candidates, active_policy)
    insufficient = len(selected) < active_policy.min_keep
    logger.info(
        "Retrieved %s candidates, selected %s, insufficient=%s, filters=%s.",
        len(candidates),
        len(selected),
        insufficient,
        active_filters.describe(),
    )
    return RetrievalResult(
        original_question=prepared.original_question,
        question=prepared.normalized_question,
        filters=active_filters,
        selected=selected,
        dropped=dropped,
        insufficient_evidence=insufficient,
        candidate_count=len(candidates),
    )


def open_collection(settings: Settings) -> chromadb.Collection:
    """Open the ingested collection and refuse a different embedding model."""
    client = chroma_client(settings)
    name = settings.collection_name
    if name not in collection_names(client):
        directory = settings.resolve_path(settings.chroma_dir)
        raise CollectionNotFoundError(
            f"Collection '{name}' does not exist in {directory}. "
            "Ingest the CSV files before running retrieval."
        )
    collection = client.get_collection(name)
    stored_model = dict(collection.metadata or {}).get("embedding_model")
    if stored_model != settings.embedding_model:
        raise CollectionConfigError(
            f"The collection '{name}' was built with embedding model '{stored_model}', "
            f"but the current configuration uses '{settings.embedding_model}'. "
            "Querying with a different model does not produce a meaningful ranking."
        )
    return collection


def search_collection(
    collection: chromadb.Collection,
    embeddings: Embeddings,
    question: str,
    *,
    filters: RetrievalFilters | None,
    candidate_k: int,
) -> list[RetrievedDocument]:
    """Return up to `candidate_k` nearest chunks that pass `filters`."""
    where = to_chroma_where(filters)
    matched = collection.get(where=where, include=[]) if where is not None else collection.get(include=[])
    available = len(matched.get("ids", []))
    if available == 0 or candidate_k < 1:
        return []
    n_results = min(candidate_k, available)
    vector = [float(value) for value in embeddings.embed_query(question)]
    query: dict[str, object] = {
        "query_embeddings": [vector],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query["where"] = where
    found = collection.query(**query)
    return _rows(found)


def _rows(found: dict[str, object]) -> list[RetrievedDocument]:
    ids = _column(found, "ids")
    documents = _column(found, "documents")
    metadatas = _column(found, "metadatas")
    distances = _column(found, "distances")
    rows: list[RetrievedDocument] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        text = documents[index] if index < len(documents) and isinstance(documents[index], str) else ""
        distance = distances[index] if index < len(distances) else None
        if chunk_id is None or distance is None:
            continue
        try:
            distance_value = float(distance)
        except (TypeError, ValueError):
            continue
        rows.append(
            RetrievedDocument(
                id=str(chunk_id),
                text=text,
                distance=distance_value,
                doc_type=str(metadata.get("doc_type", "")),
                entity_id=str(metadata.get("entity_id", "")),
                order_id=str(metadata.get("order_id", "")),
                review_score=_as_int(metadata.get("review_score"), MISSING_REVIEW_SCORE),
                product_category=str(metadata.get("product_category", "")),
                customer_state=str(metadata.get("customer_state", "")),
                purchase_year=_as_int(metadata.get("purchase_year"), 0),
                purchase_month=_as_int(metadata.get("purchase_month"), 0),
            )
        )
    return rows


def _column(found: dict[str, object], key: str) -> list[object]:
    batch = _as_list(found.get(key))
    if not batch:
        return []
    return _as_list(batch[0]) or []


def _as_list(value: object) -> list[object] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    to_list = getattr(value, "tolist", None)
    if not callable(to_list):
        return None
    converted = to_list()
    if isinstance(converted, list):
        return converted
    return None


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    # Numpy scalars expose the Python value through .item().
    item = getattr(value, "item", None)
    if not callable(item):
        return default
    try:
        return int(item())
    except (TypeError, ValueError):
        return default
