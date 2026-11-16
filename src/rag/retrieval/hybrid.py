"""Blend vector similarity with how well the comment answers the question.

The vector search finds the neighborhood. A cross-encoder then reads the
question and the comment. The blend still trusts the vector more: 0.7 for
the distance, 0.3 for that reading.
"""

from __future__ import annotations

VECTOR_WEIGHT = 0.7
RELEVANCE_WEIGHT = 0.3


def hybrid_score(distance: float, relevance: float) -> float:
    """Higher is better. `relevance` is a probability from 0 to 1."""
    similarity = max(0.0, 1.0 - float(distance))
    bounded = min(1.0, max(0.0, float(relevance)))
    return (VECTOR_WEIGHT * similarity) + (RELEVANCE_WEIGHT * bounded)
