"""Read review score and customer state from a normalized question.

The rules look for explicit phrases. A question about delivery delay stays
unfiltered on score, because the word "entrega" does not say whether the
reviews are negative. An explicit filter passed by the caller wins over
anything found here.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TypeVar

from rag.retrieval.filters import RetrievalFilters

_T = TypeVar("_T")

# Longer names come first so "mato grosso do sul" is not read as "mato grosso".
_STATE_NAMES: tuple[tuple[str, str], ...] = (
    ("mato grosso do sul", "MS"),
    ("rio grande do norte", "RN"),
    ("rio grande do sul", "RS"),
    ("distrito federal", "DF"),
    ("minas gerais", "MG"),
    ("rio de janeiro", "RJ"),
    ("santa catarina", "SC"),
    ("espirito santo", "ES"),
    ("mato grosso", "MT"),
    ("sao paulo", "SP"),
    ("pernambuco", "PE"),
    ("tocantins", "TO"),
    ("alagoas", "AL"),
    ("amazonas", "AM"),
    ("maranhao", "MA"),
    ("paraiba", "PB"),
    ("parana", "PR"),
    ("rondonia", "RO"),
    ("roraima", "RR"),
    ("sergipe", "SE"),
    ("goias", "GO"),
    ("bahia", "BA"),
    ("ceara", "CE"),
    ("piaui", "PI"),
    ("amapa", "AP"),
    ("acre", "AC"),
)

_PARA_STATE = re.compile(r"\b(?:estado\s+(?:do|de)\s+para|no\s+para)\b")
_SERGIPE_SIGLA = re.compile(r"\b(?:em|no|na|estado(?:\s+de)?)\s+se\b")
_UF_SIGLA = re.compile(
    r"\b(ac|al|am|ap|ba|ce|df|es|go|ma|mg|ms|mt|pb|pe|pi|pr|rj|rn|ro|rr|rs|sc|sp|to)\b"
)
_NEGATIVE_SCORE = re.compile(
    r"\bnotas?\s+baixas?\b"
    r"|\bavaliacoes?\s+negativas?\b"
    r"|\binsatisf\w*\b"
    r"|\breclamac\w*\b"
)
_POSITIVE_SCORE = re.compile(
    r"\bnotas?\s+altas?\b"
    r"|\bavaliacoes?\s+positivas?\b"
    r"|\belogios?\b|\belogiam\b|\belogiad\w*\b"
    r"|\bsatisf(?:acao|eit[oa]s?)\b"
)


def interpret_question(normalized_question: str) -> RetrievalFilters:
    """Return score and state filters mentioned in the question."""
    folded = _fold(normalized_question)
    minimum, maximum = _score_bounds(folded)
    return RetrievalFilters(
        review_score_min=minimum,
        review_score_max=maximum,
        customer_state=_customer_state(folded),
    )


def merge_filters(
    explicit: RetrievalFilters | None,
    extracted: RetrievalFilters,
) -> RetrievalFilters:
    """Keep an explicit value when the caller set it. Fill the rest from the question."""
    chosen = explicit if explicit is not None else RetrievalFilters()
    return RetrievalFilters(
        doc_type=_prefer(chosen.doc_type, extracted.doc_type),
        review_score_min=_prefer(chosen.review_score_min, extracted.review_score_min),
        review_score_max=_prefer(chosen.review_score_max, extracted.review_score_max),
        product_category=_prefer(chosen.product_category, extracted.product_category),
        purchase_year=_prefer(chosen.purchase_year, extracted.purchase_year),
        customer_state=_prefer(chosen.customer_state, extracted.customer_state),
    )


def _score_bounds(folded: str) -> tuple[int | None, int | None]:
    negative = _NEGATIVE_SCORE.search(folded) is not None
    positive = _POSITIVE_SCORE.search(folded) is not None
    if negative and not positive:
        return 1, 3
    if positive and not negative:
        return 4, 5
    return None, None


def _customer_state(folded: str) -> str | None:
    for name, uf in _STATE_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", folded):
            return uf
    para = _PARA_STATE.search(folded)
    if para is not None:
        return "PA"
    sergipe = _SERGIPE_SIGLA.search(folded)
    if sergipe is not None:
        return "SE"
    sigla = _UF_SIGLA.search(folded)
    if sigla is not None:
        return sigla.group(1).upper()
    return None


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_accents.casefold()


def _prefer(explicit: _T | None, extracted: _T | None) -> _T | None:
    return extracted if explicit is None else explicit
