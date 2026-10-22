from pathlib import Path

from voc_rag.ingestion.prepare import METADATA_KEYS, prepare_documents


def test_metadata_uses_only_chroma_scalars(raw_dir: Path):
    documents = prepare_documents(raw_dir).documents
    assert documents

    for document in documents:
        payload = document.metadata()
        assert list(payload) == list(METADATA_KEYS)
        assert payload["order_id"] == document.order_id
        assert payload["review_score"] == document.review_score
        assert type(payload["review_score"]) is int
        assert type(payload["delivery_delay_days"]) is int
        assert type(payload["late_delivery"]) is bool
        for key, value in payload.items():
            assert value is not None, key
            assert isinstance(value, (str, int, bool)), key
            assert not isinstance(value, (list, dict)), key


def test_review_ids_stay_stable(raw_dir: Path):
    first = [document.review_id for document in prepare_documents(raw_dir).documents]
    second = [document.review_id for document in prepare_documents(raw_dir).documents]
    assert first == second
    assert len(first) == len(set(first))
