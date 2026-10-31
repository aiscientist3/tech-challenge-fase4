"""Choose which retrieved chunks are evidence.

Similarity search returns a ranked neighborhood. Selection then removes a
chunk when its cosine distance is above the cutoff, when a closer chunk of
the same order was already kept, or when the result list is already full.
Nothing in this module reads the question or calls a language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag.retrieval.filters import RetrievalFilters

DropReason = Literal["distance", "duplicate_chunk", "duplicate_order", "beyond_k"]


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Limits applied after the vector search.

    `max_distance` is a Chroma cosine distance: lower means closer. The
    default 0.65 sits above the relevant hits measured in the sample
    collection (about 0.39 to 0.53) and below a loose neighborhood.
    """

    candidate_k: int = 20
    k: int = 10
    max_distance: float = 0.65
    min_keep: int = 2

    def __post_init__(self) -> None:
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be >= 1.")
        if self.k < 1 or self.k > self.candidate_k:
            raise ValueError("k must be between 1 and candidate_k.")
        if self.min_keep < 1 or self.min_keep > self.k:
            raise ValueError("min_keep must be between 1 and k.")
        if self.max_distance <= 0:
            raise ValueError("max_distance must be > 0.")


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """One chunk returned by the vector search, before or after selection."""

    id: str
    text: str
    distance: float
    doc_type: str
    entity_id: str
    order_id: str
    review_score: int
    product_category: str
    customer_state: str
    purchase_year: int
    purchase_month: int


@dataclass(frozen=True, slots=True)
class DroppedDocument:
    """A candidate that selection refused, with the reason."""

    document: RetrievedDocument
    reason: DropReason


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Selected evidence for one question, plus what was left out.

    `original_question` is the text the user typed, kept for the later prompt.
    `question` is the normalized text that was embedded.
    """

    original_question: str
    question: str
    filters: RetrievalFilters
    selected: list[RetrievedDocument]
    dropped: list[DroppedDocument]
    insufficient_evidence: bool
    candidate_count: int


def select_documents(
    candidates: list[RetrievedDocument],
    policy: SelectionPolicy,
) -> tuple[list[RetrievedDocument], list[DroppedDocument]]:
    """Keep the closest eligible chunks and record why the others were dropped.

    Candidates are ordered by distance, then by id, so the same input always
    yields the same selection. An empty `order_id` (category and state
    summaries) is not treated as one shared order.
    """
    ordered = sorted(candidates, key=lambda document: (document.distance, document.id))
    selected: list[RetrievedDocument] = []
    dropped: list[DroppedDocument] = []
    seen_entities: set[tuple[str, str]] = set()
    seen_orders: set[str] = set()
    for document in ordered:
        reason = _drop_reason(document, policy, selected, seen_entities, seen_orders)
        if reason is not None:
            dropped.append(DroppedDocument(document, reason))
            continue
        selected.append(document)
        seen_entities.add((document.doc_type, document.entity_id))
        if document.order_id:
            seen_orders.add(document.order_id)
    return selected, dropped


def _drop_reason(
    document: RetrievedDocument,
    policy: SelectionPolicy,
    selected: list[RetrievedDocument],
    seen_entities: set[tuple[str, str]],
    seen_orders: set[str],
) -> DropReason | None:
    if document.distance > policy.max_distance:
        return "distance"
    if (document.doc_type, document.entity_id) in seen_entities:
        return "duplicate_chunk"
    if document.order_id and document.order_id in seen_orders:
        return "duplicate_order"
    if len(selected) >= policy.k:
        return "beyond_k"
    return None
