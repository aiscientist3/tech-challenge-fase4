"""Metadata filters for a retrieval query.

The field names match the metadata written during ingestion and the filters
the chat UI already sends. A filter narrows the collection before similarity
is computed. Reading those values out of the question text happens in
`interpret.py`; this module only stores and validates them.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import DOC_TYPES

# Olist stores the customer UF in uppercase. These are the only values a
# state filter can match.
CUSTOMER_STATES = frozenset(
    {
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
        "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
        "SE", "SP", "TO",
    }
)


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Optional restrictions. An unset field does not filter that metadata."""

    doc_type: str | None = None
    review_score_min: int | None = None
    review_score_max: int | None = None
    product_category: str | None = None
    purchase_year: int | None = None
    customer_state: str | None = None

    def __post_init__(self) -> None:
        if self.doc_type is not None and self.doc_type not in DOC_TYPES:
            allowed = ", ".join(DOC_TYPES)
            raise ValueError(f"doc_type must be one of: {allowed}.")
        _check_score("review_score_min", self.review_score_min)
        _check_score("review_score_max", self.review_score_max)
        if (
            self.review_score_min is not None
            and self.review_score_max is not None
            and self.review_score_min > self.review_score_max
        ):
            raise ValueError("review_score_min cannot be greater than review_score_max.")
        if self.product_category is not None and not self.product_category.strip():
            raise ValueError("product_category must not be empty when it is set.")
        if self.purchase_year is not None and self.purchase_year < 1:
            raise ValueError("purchase_year must be a positive year when it is set.")
        if self.customer_state is not None:
            uf = self.customer_state.strip().upper()
            if uf not in CUSTOMER_STATES:
                raise ValueError("customer_state must be a Brazilian UF.")
            object.__setattr__(self, "customer_state", uf)

    def describe(self) -> str:
        """Return a short label of the active filters, or `nenhum`."""
        parts: list[str] = []
        if self.doc_type is not None:
            parts.append(f"doc_type={self.doc_type}")
        if self.review_score_min is not None:
            parts.append(f"review_score>={self.review_score_min}")
        if self.review_score_max is not None:
            parts.append(f"review_score<={self.review_score_max}")
        if self.product_category is not None:
            parts.append(f"product_category={self.product_category}")
        if self.purchase_year is not None:
            parts.append(f"purchase_year={self.purchase_year}")
        if self.customer_state is not None:
            parts.append(f"customer_state={self.customer_state}")
        return ", ".join(parts) if parts else "nenhum"


def to_chroma_where(filters: RetrievalFilters | None) -> dict[str, object] | None:
    """Build a Chroma `where` document. `None` means the whole collection."""
    if filters is None:
        return None
    clauses: list[dict[str, object]] = []
    if filters.doc_type is not None:
        clauses.append({"doc_type": filters.doc_type})
    if filters.product_category is not None:
        clauses.append({"product_category": filters.product_category})
    if filters.purchase_year is not None:
        clauses.append({"purchase_year": int(filters.purchase_year)})
    if filters.customer_state is not None:
        clauses.append({"customer_state": filters.customer_state})
    # This Chroma build accepts one operator per field. A closed range becomes
    # $in so "notas 1 a 3" is a single clause, which is what the query supports.
    if filters.review_score_min is not None and filters.review_score_max is not None:
        scores = list(range(int(filters.review_score_min), int(filters.review_score_max) + 1))
        clauses.append({"review_score": {"$in": scores}})
    elif filters.review_score_min is not None:
        clauses.append({"review_score": {"$gte": int(filters.review_score_min)}})
    elif filters.review_score_max is not None:
        clauses.append({"review_score": {"$lte": int(filters.review_score_max)}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _check_score(name: str, value: int | None) -> None:
    if value is None:
        return
    if value < 1 or value > 5:
        raise ValueError(f"{name} must be an integer from 1 to 5.")
