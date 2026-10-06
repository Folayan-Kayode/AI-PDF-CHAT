"""Tests for the document registry."""

import pytest

from app.database.registry import DocumentRegistry


@pytest.fixture()
def registry(tmp_path) -> DocumentRegistry:
    return DocumentRegistry(path=tmp_path / "registry.sqlite3")


def _record(**overrides):
    values = {
        "document_id": "doc-a",
        "filename": "a.pdf",
        "pages": 10,
        "chunks": 25,
        "embedding_model": "gemini-embedding-2",
        "schema_version": 2,
    }
    values.update(overrides)
    return values


def test_recorded_document_can_be_read_back(registry):
    registry.record(**_record())

    record = registry.get("doc-a")

    assert record is not None
    assert record["filename"] == "a.pdf"
    assert record["pages"] == 10
    assert record["chunks"] == 25
    assert record["embedding_model"] == "gemini-embedding-2"
    assert record["schema_version"] == 2
    assert record["created_at"]


def test_unknown_document_is_none(registry):
    assert registry.get("missing") is None


def test_recording_the_same_document_twice_updates_it(registry):
    registry.record(**_record())
    registry.record(**_record(chunks=30))

    assert registry.get("doc-a")["chunks"] == 30
    assert len(registry.list()) == 1


def test_replace_with_keeps_only_the_new_document(registry):
    registry.record(**_record())

    registry.replace_with(**_record(document_id="doc-b", filename="b.pdf"))

    assert [row["document_id"] for row in registry.list()] == ["doc-b"]


def test_documents_can_be_forgotten(registry):
    registry.record(**_record())

    registry.forget("doc-a")

    assert registry.get("doc-a") is None


def test_list_is_newest_first(registry):
    registry.record(**_record(document_id="doc-a", filename="a.pdf"))

    registry.record(**_record(document_id="doc-b", filename="b.pdf"))

    ids = [row["document_id"] for row in registry.list()]

    assert set(ids) == {"doc-a", "doc-b"}


def test_registry_survives_reopening(tmp_path):
    path = tmp_path / "registry.sqlite3"

    DocumentRegistry(path=path).record(**_record())

    assert DocumentRegistry(path=path).get("doc-a") is not None
