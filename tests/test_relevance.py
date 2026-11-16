"""The cross-encoder call, without downloading the model."""

from __future__ import annotations

import pytest

from rag.retrieval.relevance import RelevanceError, _score_with


class _Encoder:
    def __init__(self, raw: object) -> None:
        self.raw = raw
        self.pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs: list[tuple[str, str]], **_kwargs: object) -> object:
        self.pairs = pairs
        return self.raw


def test_scores_are_probabilities_and_only_the_comment_is_read() -> None:
    encoder = _Encoder([0.0, 2.0])
    scaffold = "Review score: 1/5. Review title: Atraso. Review comment: Não chegou o pedido."
    scores = _score_with(encoder, "atraso na entrega?", ["A entrega foi rápida.", scaffold])
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(0.880797, abs=1e-4)
    assert encoder.pairs is not None
    assert encoder.pairs[1][0] == "atraso na entrega?"
    assert encoder.pairs[1][1] == "Atraso.\nNão chegou o pedido."
    assert "Review score" not in encoder.pairs[1][1]


def test_a_short_score_list_is_refused() -> None:
    with pytest.raises(RelevanceError, match="quantidade"):
        _score_with(_Encoder([0.1]), "pergunta", ["um", "dois"])


def test_an_empty_candidate_list_does_not_call_the_model() -> None:
    encoder = _Encoder([1.0])
    assert _score_with(encoder, "pergunta", []) == []
    assert encoder.pairs is None
