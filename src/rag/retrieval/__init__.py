"""Search the ingested collection with any user question.

The flow is: prepare the question, read score and state filters from it,
embed it with the indexing model, retrieve the nearest chunks, and drop those
that are too far, duplicated, or past the result limit. It does not call a
generative model.
"""

from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.payload import to_payload
from rag.retrieval.retrieve import CollectionNotFoundError, prepare_question, retrieve
from rag.retrieval.select import RetrievalResult, SelectionPolicy

__all__ = [
    "CollectionNotFoundError",
    "RetrievalFilters",
    "RetrievalResult",
    "SelectionPolicy",
    "prepare_question",
    "retrieve",
    "to_payload",
]
