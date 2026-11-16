"""Choose which retrieved chunks are evidence.

Similarity search returns a ranked neighborhood. When the question is known,
a cross-encoder reads the question with each comment and the chunks are
reordered by vector similarity plus that score. Selection then removes a
chunk that is too far, duplicated, or past the result limit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.group import RegionTally
from rag.retrieval.hybrid import hybrid_score

DropReason = Literal["distance", "duplicate_chunk", "duplicate_order", "beyond_k"]


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Limits applied after the vector search.

    `max_distance` is a Chroma cosine distance: lower means closer. The
    default 0.75 keeps a direct question, whose hits sit around 0.56 to
    0.63, and a comparison, whose nearest reviews sit just above 0.65.
    """

    candidate_k: int = 40
    k: int = 10
    max_distance: float = 0.75
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
    relevance: float = 0.0


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
    score_groups: tuple[RegionTally, ...] = ()
    score_focus: str = ""
    compare_left: tuple[str, ...] = ()
    compare_right: tuple[str, ...] = ()


def select_documents(
    candidates: list[RetrievedDocument],
    policy: SelectionPolicy,
    *,
    question: str | None = None,
    score_passages: Callable[[str, list[str]], list[float]] | None = None,
) -> tuple[list[RetrievedDocument], list[DroppedDocument]]:
    """Keep the best eligible chunks and record why the others were dropped.

    With a question, the order is the hybrid score: vector similarity plus
    how well the comment answers the question. Without a question, the order
    stays the vector distance. An empty `order_id` is not one shared order.
    """
    scored = _with_relevance(candidates, question, score_passages)
    ordered = sorted(scored, key=_rank_key(question))
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


def _with_relevance(
    candidates: list[RetrievedDocument],
    question: str | None,
    score_passages: Callable[[str, list[str]], list[float]] | None,
) -> list[RetrievedDocument]:
    if not question or not question.strip() or not candidates:
        return candidates
    scorer = score_passages if score_passages is not None else _default_scorer()
    scores = scorer(question, [document.text for document in candidates])
    if len(scores) != len(candidates):
        raise ValueError("The relevance model returned one score per comment.")
    return [
        replace(document, relevance=float(score))
        for document, score in zip(candidates, scores, strict=True)
    ]


def _default_scorer() -> Callable[[str, list[str]], list[float]]:
    from rag.retrieval.relevance import score_passages

    return score_passages


def _rank_key(question: str | None):
    def key(document: RetrievedDocument) -> tuple[float, float, str]:
        if not question or not question.strip():
            return (document.distance, document.distance, document.id)
        return (-hybrid_score(document.distance, document.relevance), document.distance, document.id)

    return key


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
