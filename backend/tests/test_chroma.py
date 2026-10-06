"""Tests for the vector store wrapper: batching, retry and safe swap."""

import pytest

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.database.chroma import (
    COLLECTION_NAME,
    STAGING_COLLECTION_NAME,
    ChromaDatabase,
)


class RecordingCollection:
    """Stands in for a Chroma collection and can fail on demand."""

    def __init__(self, failures: int = 0):
        self.failures = failures
        self.batch_sizes: list[int] = []

    def add(self, **payload) -> None:
        self.batch_sizes.append(len(payload["ids"]))

        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("transient write failure")


@pytest.fixture()
def db(tmp_path) -> ChromaDatabase:
    return ChromaDatabase(path=str(tmp_path / "chroma"))


def _payload(count: int) -> dict:
    return {
        "ids": [f"doc_1_{i}" for i in range(count)],
        "documents": [f"chunk {i}" for i in range(count)],
        "embeddings": [[0.1, 0.2] for _ in range(count)],
        "metadatas": [{"page": 1, "chunk": i} for i in range(count)],
    }


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


def test_add_documents_splits_into_batches(db, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_WRITE_BATCH_SIZE", 2)

    recording = RecordingCollection()

    db.add_documents(collection=recording, **_payload(5))

    assert recording.batch_sizes == [2, 2, 1]


def test_add_documents_writes_single_batch_when_it_fits(db, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_WRITE_BATCH_SIZE", 500)

    recording = RecordingCollection()

    db.add_documents(collection=recording, **_payload(3))

    assert recording.batch_sizes == [3]


def test_all_four_lists_stay_aligned(db, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_WRITE_BATCH_SIZE", 2)

    seen: list[tuple] = []

    class Inspecting(RecordingCollection):
        def add(self, **payload):
            seen.append(
                (
                    len(payload["ids"]),
                    len(payload["documents"]),
                    len(payload["embeddings"]),
                    len(payload["metadatas"]),
                )
            )

    db.add_documents(collection=Inspecting(), **_payload(5))

    assert seen == [(2, 2, 2, 2), (2, 2, 2, 2), (1, 1, 1, 1)]


# --------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------


def test_transient_write_failure_is_retried(db, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_WRITE_MAX_RETRIES", 3)
    monkeypatch.setattr(settings, "CHROMA_WRITE_RETRY_BASE_SECONDS", 0.0)

    recording = RecordingCollection(failures=2)

    db.add_documents(collection=recording, **_payload(2))

    assert len(recording.batch_sizes) == 3


def test_write_failure_after_all_attempts_raises(db, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_WRITE_MAX_RETRIES", 3)
    monkeypatch.setattr(settings, "CHROMA_WRITE_RETRY_BASE_SECONDS", 0.0)

    recording = RecordingCollection(failures=99)

    with pytest.raises(RetrievalError):
        db.add_documents(collection=recording, **_payload(2))

    assert len(recording.batch_sizes) == 3


# --------------------------------------------------------------------------
# Non-destructive replacement (B1)
# --------------------------------------------------------------------------


def test_replace_documents_indexes_content(db):
    db.replace_documents(
        ids=["doc_1_1"],
        documents=["the first document"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"document_id": "doc"}],
    )

    assert db.count() == 1


def test_replace_documents_swaps_in_new_content(db):
    db.replace_documents(
        ids=["a_1_1"],
        documents=["first"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"document_id": "a"}],
    )

    db.replace_documents(
        ids=["b_1_1"],
        documents=["second"],
        embeddings=[[0.9, 0.9]],
        metadatas=[{"document_id": "b"}],
    )

    assert db.count() == 1

    result = db.search([0.9, 0.9], n_results=1)

    assert result["documents"][0] == ["second"]


def test_failed_replace_keeps_the_previous_index(db, monkeypatch):
    db.replace_documents(
        ids=["a_1_1"],
        documents=["first"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"document_id": "a"}],
    )

    def explode(**kwargs):
        raise RetrievalError("Could not write to the document index.")

    monkeypatch.setattr(db, "add_documents", explode)

    with pytest.raises(RetrievalError):
        db.replace_documents(
            ids=["b_1_1"],
            documents=["second"],
            embeddings=[[0.9, 0.9]],
            metadatas=[{"document_id": "b"}],
        )

    # The document that was already indexed is still there and queryable.
    assert db.count() == 1

    result = db.search([0.1, 0.2], n_results=1)

    assert result["documents"][0] == ["first"]


def test_failed_replace_drops_the_staging_collection(db, monkeypatch):
    def explode(**kwargs):
        raise RetrievalError("Could not write to the document index.")

    monkeypatch.setattr(db, "add_documents", explode)

    with pytest.raises(RetrievalError):
        db.replace_documents(
            ids=["b_1_1"],
            documents=["second"],
            embeddings=[[0.9, 0.9]],
            metadatas=[{"document_id": "b"}],
        )

    assert STAGING_COLLECTION_NAME not in db._names()
    assert COLLECTION_NAME in db._names()


def test_replace_cleans_up_previous_collection(db):
    for name in ("a", "b"):
        db.replace_documents(
            ids=[f"{name}_1_1"],
            documents=[name],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"document_id": name}],
        )

    assert db._names() == {COLLECTION_NAME}


# --------------------------------------------------------------------------
# Reads and error wrapping
# --------------------------------------------------------------------------


def test_has_document_matches_only_the_same_document(db):
    db.add_documents(
        ids=["a_1_1"],
        documents=["first"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"document_id": "a"}],
    )

    assert db.has_document("a") is True
    assert db.has_document("b") is False


def test_indexed_embedding_models_reports_what_was_stored(db):
    db.add_documents(
        ids=["a_1_1"],
        documents=["first"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"document_id": "a", "embedding_model": "model-x"}],
    )

    assert db.indexed_embedding_models() == {"model-x"}


def test_indexed_embedding_models_is_empty_for_an_empty_index(db):
    assert db.indexed_embedding_models() == set()


def test_document_profile_returns_text_and_metadata(db):
    db.add_documents(
        ids=["a_1_0"],
        documents=["Document profile:\nTitle: Principles of Information Security"],
        embeddings=[[0.1, 0.2]],
        metadatas=[
            {
                "document_id": "a",
                "page": 1,
                "chunk": 0,
                "kind": "document_summary",
            }
        ],
    )

    profile = db.document_profile()

    assert profile is not None
    assert "Title: Principles of Information Security" in profile["text"]
    assert profile["metadata"]["kind"] == "document_summary"


def test_document_profile_is_none_without_a_profile_chunk(db):
    db.add_documents(
        ids=["a_1_1"],
        documents=["ordinary content"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"document_id": "a", "page": 1, "chunk": 1, "kind": "content"}],
    )

    assert db.document_profile() is None


def test_document_profile_is_none_for_an_empty_index(db):
    assert db.document_profile() is None


def test_document_profile_wraps_collection_errors(db, monkeypatch):
    class BrokenCollection:
        def get(self, *args, **kwargs):
            raise RuntimeError("index is down")

    monkeypatch.setattr(db, "collection", BrokenCollection())

    with pytest.raises(RetrievalError):
        db.document_profile()


def test_reset_empties_the_index(db):
    db.add_documents(
        ids=["a_1_1"],
        documents=["first"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"document_id": "a"}],
    )

    db.reset()

    assert db.count() == 0


@pytest.mark.parametrize(
    "method, args",
    [
        ("count", ()),
        ("has_document", ("a",)),
        ("search", ([0.1, 0.2],)),
        ("indexed_embedding_models", ()),
    ],
)
def test_collection_errors_are_wrapped_as_retrieval_errors(
    db,
    monkeypatch,
    method,
    args,
):
    class BrokenCollection:
        def count(self):
            raise RuntimeError("index is down")

        def get(self, *a, **k):
            raise RuntimeError("index is down")

        def query(self, *a, **k):
            raise RuntimeError("index is down")

    monkeypatch.setattr(db, "collection", BrokenCollection())

    with pytest.raises(RetrievalError):
        getattr(db, method)(*args)


def test_add_documents_wraps_collection_errors(db, monkeypatch):
    monkeypatch.setattr(settings, "CHROMA_WRITE_MAX_RETRIES", 1)

    with pytest.raises(RetrievalError):
        db.add_documents(collection=RecordingCollection(failures=99), **_payload(1))
