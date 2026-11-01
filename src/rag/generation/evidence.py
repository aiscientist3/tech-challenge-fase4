"""Turn a retrieval payload into the reviews the prompt is allowed to use.

A review without an id, without text, or without a score from 1 to 5 is
dropped. Fewer than `min_reviews` usable comments means the generator must
not ask a model to answer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Evidence:
    """One review that can be shown to the model."""

    review_id: str
    text: str
    review_score: int


def usable_reviews(payload: dict[str, object], *, min_reviews: int) -> tuple[str, list[Evidence]] | None:
    """Return the original question and the usable reviews.

    `None` means the payload cannot support an answer: the retrieval already
    marked the evidence as insufficient, the question is blank, or too few
    reviews survived validation.
    """
    if min_reviews < 1:
        raise ValueError("min_reviews must be >= 1.")
    if payload.get("insufficient_evidence") is True:
        return None
    question = _question(payload)
    if question is None:
        return None
    reviews = _reviews(payload.get("retrieved_reviews"))
    if len(reviews) < min_reviews:
        return None
    return question, reviews


def _question(payload: dict[str, object]) -> str | None:
    query = payload.get("query")
    if not isinstance(query, dict):
        return None
    original = query.get("original_question")
    if not isinstance(original, str):
        return None
    text = original.strip()
    return text or None


def _reviews(raw: object) -> list[Evidence]:
    if not isinstance(raw, list):
        return []
    reviews: list[Evidence] = []
    seen: set[str] = set()
    for item in raw:
        evidence = _one(item)
        if evidence is None or evidence.review_id in seen:
            continue
        seen.add(evidence.review_id)
        reviews.append(evidence)
    return reviews


def _one(item: object) -> Evidence | None:
    if not isinstance(item, dict):
        return None
    review_id = item.get("review_id")
    text = item.get("text")
    score = _score(item.get("review_score"))
    if not isinstance(review_id, str) or not review_id.strip():
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    if score is None:
        return None
    return Evidence(review_id=review_id.strip(), text=text.strip(), review_score=score)


def _score(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 1 or value > 5:
        return None
    return value
