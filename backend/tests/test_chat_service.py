"""Tests for per-document chat scoping (E1)."""

import pytest

from app.core.exceptions import DocumentNotFoundError
from app.services import chat_service


class FakeRegistry:
    def __init__(self, records):
        self.records = records

    def list(self):
        return self.records

    def get(self, document_id):
        return next(
            (record for record in self.records if record["document_id"] == document_id),
            None,
        )


class FakeDatabase:
    def __init__(self, indexed):
        self.indexed = indexed

    def has_document(self, document_id):
        return document_id in self.indexed


def test_chat_uses_the_newest_document_when_none_is_given(monkeypatch):
    registry = FakeRegistry([{"document_id": "newest"}, {"document_id": "older"}])

    monkeypatch.setattr(chat_service, "get_registry", lambda: registry)

    captured: dict[str, str | None] = {}

    class FakeRetriever:
        def __init__(self, document_id=None):
            captured["document_id"] = document_id

    class FakePipeline:
        def __init__(self, retriever=None, generator=None):
            self.retriever = retriever

        def ask(self, question):
            return {"answer": "ok"}

    monkeypatch.setattr(chat_service, "Retriever", FakeRetriever)
    monkeypatch.setattr(chat_service, "RAGPipeline", FakePipeline)

    result = chat_service.ChatService.chat("q")

    assert result == {"answer": "ok"}
    assert captured["document_id"] == "newest"


def test_chat_scopes_to_the_requested_document(monkeypatch):
    registry = FakeRegistry([{"document_id": "newest"}, {"document_id": "older"}])

    monkeypatch.setattr(chat_service, "get_registry", lambda: registry)

    captured: dict[str, str | None] = {}

    class FakeRetriever:
        def __init__(self, document_id=None):
            captured["document_id"] = document_id

    class FakePipeline:
        def __init__(self, retriever=None, generator=None):
            self.retriever = retriever

        def ask(self, question):
            return {"answer": "ok"}

    monkeypatch.setattr(chat_service, "Retriever", FakeRetriever)
    monkeypatch.setattr(chat_service, "RAGPipeline", FakePipeline)

    chat_service.ChatService.chat("q", document_id="older")

    assert captured["document_id"] == "older"


def test_chat_rejects_an_unknown_document(monkeypatch):
    monkeypatch.setattr(chat_service, "get_registry", lambda: FakeRegistry([]))
    monkeypatch.setattr(
        chat_service,
        "get_database",
        lambda: FakeDatabase(indexed=set()),
    )

    with pytest.raises(DocumentNotFoundError):
        chat_service.ChatService.chat("q", document_id="missing")


def test_chat_accepts_a_document_known_only_to_the_index(monkeypatch):
    """A lost registry must not turn a queryable document into a 404."""
    monkeypatch.setattr(chat_service, "get_registry", lambda: FakeRegistry([]))
    monkeypatch.setattr(
        chat_service,
        "get_database",
        lambda: FakeDatabase(indexed={"orphan"}),
    )

    captured: dict[str, str | None] = {}

    class FakeRetriever:
        def __init__(self, document_id=None):
            captured["document_id"] = document_id

    class FakePipeline:
        def __init__(self, retriever=None, generator=None):
            self.retriever = retriever

        def ask(self, question):
            return {"answer": "ok"}

    monkeypatch.setattr(chat_service, "Retriever", FakeRetriever)
    monkeypatch.setattr(chat_service, "RAGPipeline", FakePipeline)

    chat_service.ChatService.chat("q", document_id="orphan")

    assert captured["document_id"] == "orphan"


def test_chat_uses_an_explicit_pipeline_unchanged():
    class FakePipeline:
        def ask(self, question):
            return {"answer": "from-pipeline"}

    # No registry or database access should happen on this path.
    result = chat_service.ChatService.chat("q", pipeline=FakePipeline())

    assert result == {"answer": "from-pipeline"}
