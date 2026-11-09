"""Readable answer built only from retrieved reviews.

This is a study stand-in for the language-model call. It quotes the comments
the retrieval already selected and refuses when those comments are too few.
"""

from __future__ import annotations

from rag.generation.evidence import usable_reviews
from rag.generation.prompt import REFUSAL

_COMMENT_MARK = "Review comment:"


def draft_answer(
    payload: dict[str, object],
    *,
    min_reviews: int = 2,
    max_reviews: int = 4,
    excerpt_chars: int = 320,
) -> str:
    """Return a Portuguese answer, or the refusal when evidence is too thin."""
    if max_reviews < 1:
        raise ValueError("max_reviews must be >= 1.")
    if excerpt_chars < 40:
        raise ValueError("excerpt_chars must be >= 40.")
    usable = usable_reviews(payload, min_reviews=min_reviews)
    if usable is None:
        return REFUSAL
    question, _reviews = usable
    shown = _showable(payload, limit=max_reviews, excerpt_chars=excerpt_chars)
    if len(shown) < min_reviews:
        return REFUSAL
    lines = [
        f"Pergunta\n{question}",
        "",
        "Resposta",
        (
            f"Encontrei {len(shown)} avaliações próximas dessa pergunta. "
            "O texto abaixo é o que elas dizem. Nenhum modelo reescreveu isso."
        ),
        "",
    ]
    for index, review in enumerate(shown, 1):
        lines.append(
            f"[{index}] nota {review['score']}/5 · pedido {review['order_id']} · "
            f"similaridade {review['similarity']}"
        )
        lines.append(review["excerpt"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _showable(
    payload: dict[str, object],
    *,
    limit: int,
    excerpt_chars: int,
) -> list[dict[str, object]]:
    raw = payload.get("retrieved_reviews")
    if not isinstance(raw, list):
        return []
    shown: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in raw:
        if len(shown) >= limit:
            break
        review = _one(item, excerpt_chars=excerpt_chars)
        if review is None or review["review_id"] in seen:
            continue
        seen.add(str(review["review_id"]))
        shown.append(review)
    return shown


def _one(item: object, *, excerpt_chars: int) -> dict[str, object] | None:
    if not isinstance(item, dict):
        return None
    review_id = item.get("review_id")
    text = item.get("text")
    score = item.get("review_score")
    if not isinstance(review_id, str) or not review_id.strip():
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    if isinstance(score, bool) or not isinstance(score, int) or score < 1 or score > 5:
        return None
    order_id = item.get("order_id")
    similarity = item.get("similarity")
    return {
        "review_id": review_id.strip(),
        "score": score,
        "order_id": order_id.strip() if isinstance(order_id, str) and order_id.strip() else "sem id",
        "similarity": similarity if isinstance(similarity, (int, float)) and not isinstance(similarity, bool) else "?",
        "excerpt": _excerpt(text, excerpt_chars),
    }


def _excerpt(text: str, limit: int) -> str:
    comment = text
    mark = text.lower().rfind(_COMMENT_MARK.lower())
    if mark >= 0:
        comment = text[mark + len(_COMMENT_MARK) :]
    comment = " ".join(comment.split())
    if len(comment) <= limit:
        return comment
    return comment[: limit - 3].rstrip() + "..."
