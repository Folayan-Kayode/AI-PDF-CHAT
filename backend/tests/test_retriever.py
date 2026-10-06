"""Tests for retrieval: shape, top-k, threshold, rewriting and reranking."""

from typing import Any

from app.core.config import settings
from app.rag.retriever import Retriever


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2, 0.3]


def _result(
    documents: list[str] | None = None,
    metadatas: list[dict[str, Any]] | None = None,
    distances: list[float] | None = None,
) -> dict[str, Any]:
    """A Chroma-shaped query result."""
    return {
        "documents": [documents if documents is not None else ["chunk"]],
        "metadatas": [metadatas if metadatas is not None else [{"page": 1, "chunk": 1}]],
        "distances": [distances if distances is not None else [0.42]],
    }


class FakeDatabase:
    """Returns one canned result per search, then repeats the last one."""

    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        models: set[str] | None = None,
        profile: dict[str, Any] | None = None,
    ):
        self.results = results if results is not None else [_result()]
        self.models = models if models is not None else set()
        self.profile = profile
        self.searches: list[int] = []

    def indexed_embedding_models(self, sample_size: int = 200) -> set[str]:
        return self.models

    def document_profile(self) -> dict[str, Any] | None:
        return self.profile

    def search(self, embedding, n_results: int = 5) -> dict[str, Any]:
        self.searches.append(n_results)

        index = len(self.searches) - 1

        if index < len(self.results):
            return self.results[index]

        return _result()


class FakeRewriter:
    def __init__(self, rewritten: str | None = None):
        self.rewritten = rewritten
        self.calls: list[str] = []

    def rewrite(self, question: str) -> str | None:
        self.calls.append(question)
        return self.rewritten


class FakeReranker:
    def __init__(self, order: list[int] | None = None):
        self.order = order
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, question: str, passages: list[str]) -> list[int] | None:
        self.calls.append((question, passages))
        return self.order


def _retriever(
    database: FakeDatabase,
    embedder: FakeEmbedder | None = None,
    rewriter: FakeRewriter | None = None,
    reranker: FakeReranker | None = None,
) -> Retriever:
    return Retriever(
        embedding_model=embedder or FakeEmbedder(),
        database=database,
        query_rewriter=rewriter or FakeRewriter(),
        reranker=reranker or FakeReranker(),
    )


# --------------------------------------------------------------------------
# Basic shape
# --------------------------------------------------------------------------


def test_retrieve_returns_documents_metadata_and_distances():
    database = FakeDatabase(
        results=[
            _result(
                documents=["a", "b"],
                metadatas=[{"page": 1, "chunk": 1}, {"page": 2, "chunk": 1}],
                distances=[0.1, 0.2],
            )
        ]
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
    database = FakeDatabase(results=[_result(documents=[], metadatas=[], distances=[])])

    assert _retriever(database).retrieve("what?") == {
        "documents": [],
        "metadata": [],
        "distances": [],
    }


def test_missing_keys_do_not_crash():
    database = FakeDatabase(results=[{}])

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
# Embedding-model fingerprint guard
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

    assert _retriever(database).retrieve("what?")["documents"] == ["chunk"]


# --------------------------------------------------------------------------
# Distance threshold
# --------------------------------------------------------------------------


def test_weak_matches_are_dropped(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 0.5)

    database = FakeDatabase(
        results=[
            _result(
                documents=["close", "medium", "far"],
                metadatas=[
                    {"page": 1, "chunk": 1},
                    {"page": 2, "chunk": 1},
                    {"page": 3, "chunk": 1},
                ],
                distances=[0.2, 0.45, 0.8],
            )
        ]
    )

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == ["close", "medium"]
    assert result["distances"] == [0.2, 0.45]


def test_everything_weak_returns_no_context(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 0.5)

    database = FakeDatabase(results=[_result(distances=[0.9])])

    assert _retriever(database).retrieve("what?")["documents"] == []


def test_threshold_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 10.0)

    database = FakeDatabase(results=[_result(distances=[0.95])])

    assert _retriever(database).retrieve("what?")["documents"] == ["chunk"]


# --------------------------------------------------------------------------
# Multi-query rewriting
# --------------------------------------------------------------------------


def test_rewritten_query_adds_a_second_search():
    database = FakeDatabase(results=[_result(), _result()])

    embedder = FakeEmbedder()

    _retriever(
        database,
        embedder,
        rewriter=FakeRewriter("better search terms"),
    ).retrieve("what is the title?")

    assert embedder.queries == ["what is the title?", "better search terms"]
    assert len(database.searches) == 2


def test_results_from_both_queries_are_merged_and_deduped():
    database = FakeDatabase(
        results=[
            _result(
                documents=["alpha", "beta"],
                metadatas=[
                    {"page": 1, "chunk": 1, "document_id": "d"},
                    {"page": 2, "chunk": 1, "document_id": "d"},
                ],
                distances=[0.5, 0.6],
            ),
            _result(
                documents=["beta", "gamma"],
                metadatas=[
                    {"page": 2, "chunk": 1, "document_id": "d"},
                    {"page": 3, "chunk": 1, "document_id": "d"},
                ],
                distances=[0.2, 0.7],
            ),
        ]
    )

    result = _retriever(database, rewriter=FakeRewriter("better")).retrieve("q")

    # beta appears in both and keeps its best distance.
    assert result["documents"] == ["beta", "alpha", "gamma"]
    assert result["distances"] == [0.2, 0.5, 0.7]


def test_rewrite_identical_to_the_question_is_ignored():
    database = FakeDatabase()

    embedder = FakeEmbedder()

    _retriever(database, embedder, rewriter=FakeRewriter("q")).retrieve("q")

    assert embedder.queries == ["q"]
    assert len(database.searches) == 1


def test_missing_rewrite_keeps_the_original_query():
    database = FakeDatabase()

    embedder = FakeEmbedder()

    _retriever(database, embedder, rewriter=FakeRewriter(None)).retrieve("q")

    assert embedder.queries == ["q"]


# --------------------------------------------------------------------------
# Reranking
# --------------------------------------------------------------------------


def test_rerank_reorders_when_retrieval_is_weak(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)

    database = FakeDatabase(
        results=[
            _result(
                documents=["noise", "noise2", "answer"],
                metadatas=[
                    {"page": 1, "chunk": 1},
                    {"page": 2, "chunk": 1},
                    {"page": 3, "chunk": 1},
                ],
                distances=[0.6, 0.62, 0.7],
            )
        ]
    )

    reranker = FakeReranker(order=[2])

    result = _retriever(database, reranker=reranker).retrieve("q", n_results=1)

    assert len(reranker.calls) == 1
    assert reranker.calls[0][0] == "q"
    assert result["documents"] == ["answer"]


def test_rerank_keeps_displaced_candidates_after_the_ranked_ones(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)

    documents = ["a", "b", "c", "d", "e"]

    database = FakeDatabase(
        results=[
            _result(
                documents=documents,
                metadatas=[{"page": index, "chunk": 1} for index in range(len(documents))],
                distances=[0.6] * len(documents),
            )
        ]
    )

    # Promote "d" (index 3) to the front; the rest keep their order behind it.
    result = _retriever(database, reranker=FakeReranker(order=[3])).retrieve("q", n_results=3)

    assert result["documents"] == ["d", "a", "b"]


def test_rerank_is_skipped_when_retrieval_is_confident(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_SKIP_DISTANCE", 0.35)

    database = FakeDatabase(
        results=[
            _result(
                documents=["a", "b", "c"],
                distances=[0.2, 0.3, 0.4],
            )
        ]
    )

    reranker = FakeReranker(order=[2])

    _retriever(database, reranker=reranker).retrieve("q")

    assert reranker.calls == []


def test_rerank_is_skipped_when_there_is_nothing_to_reorder(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)

    database = FakeDatabase()

    reranker = FakeReranker(order=[0])

    _retriever(database, reranker=reranker).retrieve("q")

    assert reranker.calls == []


def test_unavailable_rerank_keeps_the_retrieval_order(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)

    database = FakeDatabase(
        results=[
            _result(
                documents=["a", "b", "c"],
                metadatas=[
                    {"page": 1, "chunk": 1},
                    {"page": 2, "chunk": 1},
                    {"page": 3, "chunk": 1},
                ],
                distances=[0.6, 0.61, 0.62],
            )
        ]
    )

    result = _retriever(database, reranker=FakeReranker(order=None)).retrieve("q")

    assert result["documents"] == ["a", "b", "c"]


def test_more_candidates_are_fetched_when_reranking(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATES", 20)

    database = FakeDatabase()

    _retriever(database).retrieve("q", n_results=5)

    assert database.searches == [20]


def test_only_top_k_are_returned_after_reranking(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)

    database = FakeDatabase(
        results=[
            _result(
                documents=[f"d{index}" for index in range(10)],
                metadatas=[{"page": index, "chunk": 1} for index in range(10)],
                distances=[0.6] * 10,
            )
        ]
    )

    result = _retriever(database, reranker=FakeReranker(order=[9, 8])).retrieve("q", n_results=2)

    assert result["documents"] == ["d9", "d8"]


# --------------------------------------------------------------------------
# Document profile passthrough
# --------------------------------------------------------------------------


def test_document_profile_is_returned_from_the_store():
    profile = {"text": "Title: X", "metadata": {"page": 1, "chunk": 0}}

    database = FakeDatabase(profile=profile)

    assert _retriever(database).document_profile() == profile


def test_document_profile_is_none_when_not_indexed():
    assert _retriever(FakeDatabase()).document_profile() is None


def test_document_profile_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "DOCUMENT_PROFILE_IN_CONTEXT", False)

    database = FakeDatabase(profile={"text": "Title: X", "metadata": {}})

    assert _retriever(database).document_profile() is None
