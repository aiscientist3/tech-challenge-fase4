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
    r"|(?<!sem )\bproblemas?\b"
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
    state = _customer_state(folded)
    return RetrievalFilters(
        review_score_min=minimum,
        review_score_max=maximum,
        customer_state=state,
        customer_states=None if state is not None else _region_states(folded),
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
        customer_states=_prefer(chosen.customer_states, extracted.customer_states),
    )


_REGION_LABELS: tuple[tuple[str, str], ...] = (
    ("centro-oeste", "Centro-Oeste"),
    ("centro oeste", "Centro-Oeste"),
    ("nordeste", "Nordeste"),
    ("sudeste", "Sudeste"),
    ("norte", "Norte"),
    ("sul", "Sul"),
)


def region_question(normalized_question: str) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    """Read a regional comparison.

    The result is `(focus, left, right)`. `focus` is `low`, `high`, or `count`.
    `left` and `right` are the two sides when the question says one has more
    than the other. A question that only names a score band leaves both empty.
    """
    folded = _fold(normalized_question)
    compares = (
        re.search(r"\b(mais|maior|qual)\b", folded) is not None
        or "tem a ver" in folded
    )
    named = _region_labels(folded)
    mentions_region = re.search(r"\bregio", folded) is not None or len(named) >= 2
    if not compares or not mentions_region:
        return None
    low = re.search(r"\bnotas?\s+baixas?\b|\bavaliacoes?\s+negativas?\b", folded) is not None
    high = re.search(r"\bnotas?\s+altas?\b|\bavaliacoes?\s+positivas?\b", folded) is not None
    if low and not high:
        focus = "low"
    elif high and not low:
        focus = "high"
    else:
        focus = "count"
    left, right = _comparison_sides(folded)
    if focus == "count" and not left and len(named) < 2 and re.search(r"\bregio", folded) is None:
        return None
    return focus, left, right


def asks_region_by_score(normalized_question: str) -> str | None:
    """Return `low` or `high` when the question asks which region has more of that score."""
    found = region_question(normalized_question)
    if found is None or found[0] == "count":
        return None
    return found[0]


def _region_labels(folded: str) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    seen: set[str] = set()
    for name, label in _REGION_LABELS:
        match = re.search(rf"\b{re.escape(name)}\b", folded)
        if match is None or label in seen:
            continue
        seen.add(label)
        found.append((match.start(), label))
    found.sort()
    return tuple(label for _start, label in found)


def _comparison_sides(folded: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The side said to have more, then the side it is compared with."""
    mais = re.search(r"\bmais\b", folded)
    if mais is None:
        return (), ()
    after = folded[mais.end() :]
    than = re.search(r"\b(?:do que|que)\b", after)
    if than is None:
        return (), ()
    left = _region_labels(folded[: mais.start()])
    right = _region_labels(after[than.end() :])
    if not left or not right:
        return (), ()
    return left, right


def contrasts_high_and_low(normalized_question: str) -> bool:
    """True when the question names both a high score and a low score.

    One score filter would hide one of the two sides, so the search has to
    read each band on its own.
    """
    folded = _fold(normalized_question)
    negative = _NEGATIVE_SCORE.search(folded) is not None
    positive = _POSITIVE_SCORE.search(folded) is not None
    return negative and positive


def _score_bounds(folded: str) -> tuple[int | None, int | None]:
    negative = _NEGATIVE_SCORE.search(folded) is not None
    positive = _POSITIVE_SCORE.search(folded) is not None
    if negative and not positive:
        return 1, 3
    if positive and not negative:
        return 4, 5
    return None, None


_REGIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("centro-oeste", ("DF", "GO", "MS", "MT")),
    ("centro oeste", ("DF", "GO", "MS", "MT")),
    ("nordeste", ("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE")),
    ("sudeste", ("ES", "MG", "RJ", "SP")),
    ("norte", ("AC", "AM", "AP", "PA", "RO", "RR", "TO")),
    ("sul", ("PR", "RS", "SC")),
)


def _region_states(folded: str) -> tuple[str, ...] | None:
    """Every region named in the question. The first match must not hide the others."""
    found: list[str] = []
    seen: set[str] = set()
    for name, states in _REGIONS:
        if re.search(rf"\b{re.escape(name)}\b", folded) is None:
            continue
        for uf in states:
            if uf in seen:
                continue
            seen.add(uf)
            found.append(uf)
    return tuple(found) or None


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
