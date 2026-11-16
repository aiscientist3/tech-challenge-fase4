"""Format retrieved reviews as the evidence block of the prompt.

The score and the comment are included. The customer state is included only
when the question is about a place. The review number stays out, so the
answer does not point at one card. Order, category, date, and similarity
stay out of the prompt.
"""

from __future__ import annotations

from rag.generation.evidence import Evidence
from rag.retrieval.group import region_name


def build_context(
    reviews: list[Evidence],
    *,
    max_reviews: int,
    max_chars: int,
    include_state: bool = False,
) -> str:
    """Join up to `max_reviews` comments, stopping before `max_chars`."""
    if max_reviews < 1:
        raise ValueError("max_reviews must be >= 1.")
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1.")
    blocks: list[str] = []
    used = 0
    for review in reviews:
        if len(blocks) >= max_reviews:
            break
        place = f", {review.customer_state}" if include_state and review.customer_state else ""
        block = f"Nota {review.review_score}/5{place}\n{review.text}"
        separator = 2 if blocks else 0
        if blocks and used + separator + len(block) > max_chars:
            break
        if not blocks and len(block) > max_chars:
            block = block[: max_chars - 3].rstrip() + "..."
        blocks.append(block)
        used += separator + len(block)
    return "\n\n".join(blocks)


def build_grouped_context(
    reviews: list[Evidence],
    *,
    left: tuple[str, ...],
    right: tuple[str, ...],
    max_chars: int,
) -> str:
    """Name the region group when it changes. Reviews stay unnumbered."""
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1.")
    blocks: list[str] = []
    used = 0
    previous = ""
    for review in reviews:
        label = _group_label(review.customer_state, left, right)
        place = f", {review.customer_state}" if review.customer_state else ""
        body = f"Nota {review.review_score}/5{place}\n{review.text}"
        if label and label != previous:
            body = f"{label}\n{body}"
            previous = label
        separator = 2 if blocks else 0
        if blocks and used + separator + len(body) > max_chars:
            break
        blocks.append(body)
        used += separator + len(body)
    return "\n\n".join(blocks)


def _group_label(state: str, left: tuple[str, ...], right: tuple[str, ...]) -> str:
    region = region_name(state)
    if region is None:
        return "Outras"
    if left and right:
        if region in left:
            return _join(left)
        if region in right:
            return _join(right)
    return region


def _join(names: tuple[str, ...]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " e " + names[-1]
