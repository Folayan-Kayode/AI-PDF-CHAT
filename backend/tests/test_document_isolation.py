"""
Two documents in one store must not contaminate each other.

Retrieval and the profile are scoped by document_id, and this is the test that
proves it. Without scoping, the profile of one document is injected into every
prompt, so a second document silently answers from the first one's metadata.
"""

import pytest

from app.core.config import settings
from app.database.chroma import ChromaDatabase
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = vector

    def embed_query(self, query: str) -> list[float]:
        return self.vector


class FakeGenerator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "answer [p.1]"


@pytest.fixture()
def store(tmp_path) -> ChromaDatabase:
    """Two documents, orthogonal vectors, each with its own profile."""
    database = ChromaDatabase(path=str(tmp_path / "chroma"))

    database.add_documents(
        ids=["doc-a_1_0", "doc-a_1_1", "doc-b_1_0", "doc-b_1_1"],
        documents=[
            "Profile of alpha",
            "Alpha content about turbines",
            "Profile of beta",
            "Beta content about orchards",
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        metadatas=[
            {
                "page": 0,
                "chunk": 0,
                "kind": "document_summary",
                "document_id": "doc-a",
                "embedding_model": settings.EMBEDDING_MODEL,
                "schema_version": settings.EMBEDDING_SCHEMA_VERSION,
            },
            {
                "page": 1,
                "chunk": 1,
                "kind": "content",
                "document_id": "doc-a",
                "embedding_model": settings.EMBEDDING_MODEL,
                "schema_version": settings.EMBEDDING_SCHEMA_VERSION,
            },
            {
                "page": 0,
                "chunk": 0,
                "kind": "document_summary",
                "document_id": "doc-b",
                "embedding_model": settings.EMBEDDING_MODEL,
                "schema_version": settings.EMBEDDING_SCHEMA_VERSION,
            },
            {
                "page": 1,
                "chunk": 1,
                "kind": "content",
                "document_id": "doc-b",
                "embedding_model": settings.EMBEDDING_MODEL,
                "schema_version": settings.EMBEDDING_SCHEMA_VERSION,
            },
        ],
    )

    return database


def test_search_scoped_to_a_document_returns_only_that_document(store):
    # The query points at doc-a, but we ask for doc-b: the filter must win.
    results = store.search([1.0, 0.0, 0.0], n_results=10, document_id="doc-b")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    assert documents, "doc-b has content to return"
    assert {meta["document_id"] for meta in metadatas} == {"doc-b"}
    assert "Beta content about orchards" in documents
    assert all("alpha" not in document.lower() for document in documents)
    assert "Profile of beta" in documents


def test_unscoped_search_sees_both_documents(store):
    results = store.search([1.0, 0.0, 0.0], n_results=10)

    found = {meta["document_id"] for meta in results["metadatas"][0]}

    assert found == {"doc-a", "doc-b"}


def test_profile_for_a_document_is_never_another_documents_profile(store):
    assert "Profile of alpha" in store.document_profile("doc-a")["text"]
    assert "Profile of beta" in store.document_profile("doc-b")["text"]

    profile_a = store.document_profile("doc-a")

    assert "beta" not in profile_a["text"].lower()


def test_retriever_scoped_to_a_document_returns_only_its_chunks(store):
    retriever = Retriever(
        embedding_model=FakeEmbedder([1.0, 0.0, 0.0]),
        database=store,
        document_id="doc-b",
    )

    result = retriever.retrieve("anything")

    assert result["documents"]
    assert all("beta" in document.lower() for document in result["documents"])


def test_retriever_returns_the_scoped_document_profile(store):
    retriever = Retriever(
        embedding_model=FakeEmbedder([1.0, 0.0, 0.0]),
        database=store,
        document_id="doc-b",
    )

    assert "Profile of beta" in retriever.document_profile()["text"]


def test_an_answer_scoped_to_a_document_cites_only_that_document(store):
    generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=Retriever(
            embedding_model=FakeEmbedder([1.0, 0.0, 0.0]),
            database=store,
            document_id="doc-a",
        ),
        generator=generator,
    )

    result = pipeline.ask("anything")

    assert "Profile of alpha" in generator.prompts[0]
    assert "Profile of beta" not in generator.prompts[0]
    assert all(source["document_id"] == "doc-a" for source in result["sources"])


def test_ingestion_records_each_document_in_the_registry(tmp_path, monkeypatch):
    """The registry is what makes listing and selection possible later."""
    from app.database.registry import DocumentRegistry

    registry = DocumentRegistry(path=tmp_path / "registry.sqlite3")

    registry.record(
        document_id="doc-a",
        filename="a.pdf",
        pages=1,
        chunks=2,
        embedding_model=settings.EMBEDDING_MODEL,
        schema_version=settings.EMBEDDING_SCHEMA_VERSION,
    )

    registry.record(
        document_id="doc-b",
        filename="b.pdf",
        pages=3,
        chunks=9,
        embedding_model=settings.EMBEDDING_MODEL,
        schema_version=settings.EMBEDDING_SCHEMA_VERSION,
    )

    assert {row["document_id"] for row in registry.list()} == {"doc-a", "doc-b"}
