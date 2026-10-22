"""Embeddings do E5 multilíngue.

O modelo compara vetores só quando o documento usa o prefixo ``passage: ``
e a pergunta usa ``query: ``. Esses prefixos não entram no texto gravado
na coleção.
"""

from functools import lru_cache

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


@lru_cache(maxsize=1)
def get_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _as_rows(vectors) -> list[list[float]]:
    return [[float(value) for value in row] for row in vectors]


def embed_passages(
    texts: list[str],
    model_name: str,
    batch_size: int = 64,
    model=None,
) -> list[list[float]]:
    encoder = model or get_model(model_name)
    prefixed = [PASSAGE_PREFIX + text for text in texts]
    vectors = encoder.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return _as_rows(vectors)


def embed_query(text: str, model_name: str, model=None) -> list[float]:
    encoder = model or get_model(model_name)
    vectors = encoder.encode(
        [QUERY_PREFIX + text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return _as_rows(vectors)[0]
