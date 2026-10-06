"""Tests for the retriever: shape, top-k and the embedding-model guard."""

from typing import Any

from app.core.config import settings
from app.rag.retriever import Retriever


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2, 0.3]


class FakeDatabase:
    """Returns canned results and records how it was queried."""

    def __init__(
        self,
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        distances: list[float] | None = None,
        models: set[str] | None = None,
        raw: dict[str, Any] | None = None,
    ):
        self.documents = documents if documents is not None else ["chunk"]
        self.metadatas = metadatas if metadatas is not None else [{"page": 1, "chunk": 1}]
        self.distances = distances if distances is not None else [0.42]
        self.models = models if models is not None else set()
        self.raw = raw
        self.searches: list[int] = []

    def indexed_embedding_models(self, sample_size: int = 200) -> set[str]:
        return self.models

    def search(self, embedding, n_results: int = 5) -> dict[str, Any]:
        self.searches.append(n_results)

        if self.raw is not None:
            return self.raw

        return {
            "documents": [self.documents],
            "metadatas": [self.metadatas],
            "distances": [self.distances],
        }


def _retriever(database: FakeDatabase, embedder: FakeEmbedder | None = None) -> Retriever:
    return Retriever(
        embedding_model=embedder or FakeEmbedder(),
        database=database,
    )


def test_retrieve_returns_documents_metadata_and_distances():
    database = FakeDatabase(
        documents=["a", "b"],
        metadatas=[{"page": 1, "chunk": 1}, {"page": 2, "chunk": 1}],
        distances=[0.1, 0.2],
    )

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == ["a", "b"]
    assert result["metadata"] == [{"page": 1, "chunk": 1}, {"page": 2, "chunk": 1}]
    assert result["distances"] == [0.1, 0.2]


def test_top_k_defaults_to_the_configured_value():
    database = FakeDatabase()

    _retriever(database).retrieve("what?")

    assert database.searches == [settings.RETRIEVAL_TOP_K]


def test_explicit_top_k_is_passed_through():
    database = FakeDatabase()

    _retriever(database).retrieve("what?", n_results=2)

    assert database.searches == [2]


def test_empty_results_have_a_stable_shape():
    database = FakeDatabase(
        raw={"documents": [[]], "metadatas": [[]], "distances": [[]]},
    )

    assert _retriever(database).retrieve("what?") == {
        "documents": [],
        "metadata": [],
        "distances": [],
    }


def test_missing_keys_do_not_crash():
    database = FakeDatabase(raw={})

    assert _retriever(database).retrieve("what?") == {
        "documents": [],
        "metadata": [],
        "distances": [],
    }


def test_question_is_embedded():
    embedder = FakeEmbedder()

    _retriever(FakeDatabase(), embedder).retrieve("a specific question")

    assert embedder.queries == ["a specific question"]


# --------------------------------------------------------------------------
# Embedding-model fingerprint guard (B10)
# --------------------------------------------------------------------------


def test_matching_embedding_model_queries_normally():
    database = FakeDatabase(models={settings.EMBEDDING_MODEL})

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == ["chunk"]
    assert database.searches == [settings.RETRIEVAL_TOP_K]


def test_mismatched_embedding_model_returns_nothing():
    database = FakeDatabase(models={"some-other-model"})

    result = _retriever(database).retrieve("what?")

    assert result == {"documents": [], "metadata": [], "distances": []}
    assert database.searches == [], "a foreign index must not be queried"


def test_unfingerprinted_index_is_still_usable():
    database = FakeDatabase(models=set())

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == ["chunk"]
