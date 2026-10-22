from voc_rag.embeddings import PASSAGE_PREFIX, QUERY_PREFIX, embed_passages, embed_query


class _FakeEncoder:
    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.seen.append(list(texts))
        return [[float(index), 1.0] for index, _text in enumerate(texts)]


def test_passage_prefix_is_applied_only_to_the_vector():
    encoder = _FakeEncoder()
    vectors = embed_passages(["Nota: 2/5"], model_name="ignored", model=encoder)

    assert encoder.seen == [[f"{PASSAGE_PREFIX}Nota: 2/5"]]
    assert vectors == [[0.0, 1.0]]


def test_query_prefix_is_applied_to_the_question():
    encoder = _FakeEncoder()
    vector = embed_query("atraso na entrega", model_name="ignored", model=encoder)

    assert encoder.seen == [[f"{QUERY_PREFIX}atraso na entrega"]]
    assert vector == [0.0, 1.0]
