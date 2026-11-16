"""Embed a question and read the nearest chunks from the persistent collection.

The embedding function has to be the one that indexed the collection. The
indexer stores those vectors directly, so this module embeds the question
itself and passes the vector to Chroma instead of asking the collection to
embed it with a different function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import chromadb
from langchain_core.embeddings import Embeddings

from rag.config import MISSING_REVIEW_SCORE, Settings
from rag.ingestion.indexer import (
    CollectionConfigError,
    chroma_client,
    collection_names,
)
from rag.retrieval.filters import RetrievalFilters, to_chroma_where
from rag.retrieval.group import RegionTally, example_indexes, group_regions, tally_text
from rag.retrieval.interpret import contrasts_high_and_low, interpret_question, merge_filters, region_question
from rag.retrieval.select import (
    DroppedDocument,
    RetrievalResult,
    RetrievedDocument,
    SelectionPolicy,
    select_documents,
)

logger = logging.getLogger(__name__)

# A regional comparison needs more than the usual 40 neighbors, so several
# regions show up in the same vector search before they are grouped by score.
_REGION_CANDIDATES = 80


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
    caller_set_score = filters is not None and (
        filters.review_score_min is not None or filters.review_score_max is not None
    )
    compared_question = region_question(prepared.normalized_question)
    focus = compared_question[0] if compared_question is not None else None
    compare_left = compared_question[1] if compared_question is not None else ()
    compare_right = compared_question[2] if compared_question is not None else ()
    score_groups: tuple[RegionTally, ...] = ()
    if contrasts_high_and_low(prepared.normalized_question) and not caller_set_score:
        selected, dropped, candidate_count = _both_score_bands(
            collection,
            embeddings,
            prepared.normalized_question,
            active_filters,
            active_policy,
        )
    else:
        search_filters = active_filters
        search_policy = active_policy
        if focus in {"low", "high"} and not caller_set_score:
            search_filters = replace(active_filters, review_score_min=None, review_score_max=None)
            wider = max(active_policy.candidate_k, _REGION_CANDIDATES)
            search_policy = replace(active_policy, candidate_k=wider)
        elif focus == "count":
            wider = max(active_policy.candidate_k, _REGION_CANDIDATES)
            search_policy = replace(active_policy, candidate_k=wider)
        candidates = search_collection(
            collection,
            embeddings,
            prepared.normalized_question,
            filters=search_filters,
            candidate_k=search_policy.candidate_k,
        )
        if focus in {"low", "high"}:
            score_groups = group_regions([(item.customer_state, item.review_score) for item in candidates])
            pool = _in_score_band(candidates, active_filters)
        elif focus == "count":
            score_groups = group_regions([(item.customer_state, item.review_score) for item in candidates])
            pool = candidates
        else:
            pool = candidates
        if focus is not None and tally_text(score_groups, focus, compare_left, compare_right):
            near = [item for item in pool if item.distance <= active_policy.max_distance]
            indexes = example_indexes(
                [item.customer_state for item in near],
                score_groups,
                focus,
                compare_left,
                compare_right,
                [item.order_id for item in near],
            )
            if indexes:
                selected = [near[index] for index in indexes]
                if focus == "count" and compare_left and compare_right:
                    score_groups = group_regions(
                        [(item.customer_state, item.review_score) for item in selected]
                    )
                dropped = []
                candidates = near
            else:
                selected, dropped = select_documents(
                    pool,
                    active_policy,
                    question=prepared.normalized_question,
                )
                candidates = pool
        else:
            selected, dropped = select_documents(
                pool,
                active_policy,
                question=prepared.normalized_question,
            )
            candidates = pool
        candidate_count = len(candidates)
    compared = focus is not None and bool(
        tally_text(score_groups, focus, compare_left, compare_right)
    )
    insufficient = len(selected) < active_policy.min_keep and not compared
    logger.info(
        "Retrieved %s candidates, selected %s, insufficient=%s, filters=%s, relevance=%.2f.",
        candidate_count,
        len(selected),
        insufficient,
        active_filters.describe(),
        selected[0].relevance if selected else 0.0,
    )
    return RetrievalResult(
        original_question=prepared.original_question,
        question=prepared.normalized_question,
        filters=active_filters,
        selected=selected,
        dropped=dropped,
        insufficient_evidence=insufficient,
        candidate_count=candidate_count,
        score_groups=score_groups,
        score_focus=focus or "",
        compare_left=compare_left,
        compare_right=compare_right,
    )


def _in_score_band(
    candidates: list[RetrievedDocument],
    filters: RetrievalFilters,
) -> list[RetrievedDocument]:
    """Keep the score band the question asked for, after the wider search was grouped."""
    low = filters.review_score_min
    high = filters.review_score_max
    if low is None and high is None:
        return candidates
    kept: list[RetrievedDocument] = []
    for document in candidates:
        if low is not None and document.review_score < low:
            continue
        if high is not None and document.review_score > high:
            continue
        kept.append(document)
    return kept


def _both_score_bands(
    collection: chromadb.Collection,
    embeddings: Embeddings,
    question: str,
    filters: RetrievalFilters,
    policy: SelectionPolicy,
) -> tuple[list[RetrievedDocument], list[DroppedDocument], int]:
    """Read low scores and high scores separately, then keep both sides.

    A question that asks why some notes are high and others are low has no
    single score filter. Each band is searched on its own so one side cannot
    fill the whole list.
    """
    half = SelectionPolicy(
        candidate_k=policy.candidate_k,
        k=max(1, policy.k // 2),
        max_distance=policy.max_distance,
        min_keep=1,
    )
    selected: list[RetrievedDocument] = []
    dropped: list[DroppedDocument] = []
    seen = 0
    for low, high in ((1, 3), (4, 5)):
        band = replace(filters, review_score_min=low, review_score_max=high)
        candidates = search_collection(
            collection,
            embeddings,
            question,
            filters=band,
            candidate_k=half.candidate_k,
        )
        kept, rejected = select_documents(candidates, half, question=question)
        selected.extend(kept)
        dropped.extend(rejected)
        seen += len(candidates)
    return selected, dropped, seen


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
