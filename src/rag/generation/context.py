"""Format retrieved reviews as the evidence block of the prompt.

Only the review id, the score, and the original text are included. Order,
category, state, date, and similarity stay out of the prompt.
"""

from __future__ import annotations

from rag.generation.evidence import Evidence


def build_context(
    reviews: list[Evidence],
    *,
    max_reviews: int,
    max_chars: int,
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
        block = f"[{review.review_id}] nota {review.review_score}/5\n{review.text}"
        separator = 2 if blocks else 0
        if blocks and used + separator + len(block) > max_chars:
            break
        if not blocks and len(block) > max_chars:
            block = block[: max_chars - 3].rstrip() + "..."
        blocks.append(block)
        used += separator + len(block)
    return "\n\n".join(blocks)
