"""Read the question and each comment together.

Vector search only finds a neighborhood. This model, from the
sentence-transformers library, scores whether that comment answers that
question. The same call works for any phrasing. There is no list of synonyms.
"""

from __future__ import annotations

import math
import os
import threading

from rag.generation.evidence import customer_text

MODEL_NAME = os.environ.get("RELEVANCE_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

_lock = threading.Lock()
_encoder: object | None = None


class RelevanceError(Exception):
    """The library that reads the question could not score the comments."""


def load_relevance() -> None:
    """Download and load the model once. The first call needs a network."""
    _encoder_or_load()


def score_passages(question: str, texts: list[str]) -> list[float]:
    """Return one probability per text. Higher means the comment answers the question."""
    return _score_with(_encoder_or_load(), question, texts)


def _score_with(encoder: object, question: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    pairs = [(question, customer_text(text)) for text in texts]
    predict = getattr(encoder, "predict", None)
    if not callable(predict):
        raise RelevanceError("O modelo que lê a pergunta não está pronto.")
    with _lock:
        raw = predict(pairs, show_progress_bar=False, batch_size=8, activation_fn=_identity)
    values = _as_floats(raw)
    if len(values) != len(texts):
        raise RelevanceError("O modelo que lê a pergunta devolveu uma quantidade diferente de notas.")
    return [_sigmoid(value) for value in values]


def _encoder_or_load() -> object:
    global _encoder
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is not None:
            return _encoder
        try:
            from sentence_transformers import CrossEncoder

            _encoder = CrossEncoder(MODEL_NAME, device="cpu", max_length=192)
        except Exception as exc:
            raise RelevanceError(
                "Não foi possível carregar o modelo que lê a pergunta. "
                "A primeira execução precisa de internet para baixá-lo."
            ) from exc
        return _encoder


def _as_floats(raw: object) -> list[float]:
    if isinstance(raw, bool):
        raise RelevanceError("O modelo que lê a pergunta devolveu uma nota inválida.")
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, (list, tuple)):
        return [_float_item(item) for item in raw]
    to_list = getattr(raw, "tolist", None)
    if callable(to_list):
        converted = to_list()
        if isinstance(converted, list):
            return [_float_item(item) for item in converted]
        if isinstance(converted, (int, float)) and not isinstance(converted, bool):
            return [float(converted)]
    raise RelevanceError("O modelo que lê a pergunta devolveu uma nota inválida.")


def _float_item(item: object) -> float:
    if isinstance(item, bool):
        raise RelevanceError("O modelo que lê a pergunta devolveu uma nota inválida.")
    if isinstance(item, (list, tuple)):
        if len(item) != 1:
            raise RelevanceError("O modelo que lê a pergunta devolveu uma nota inválida.")
        return _float_item(item[0])
    try:
        return float(item)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RelevanceError("O modelo que lê a pergunta devolveu uma nota inválida.") from exc


def _identity(scores: object) -> object:
    """Keep the raw logits. This module turns them into probabilities."""
    return scores


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)
