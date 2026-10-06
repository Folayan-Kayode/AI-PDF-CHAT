"""Tests for retrieval: shape, relative selection, scoping and reranking."""

from statistics import median
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
        versions: set[int] | None = None,
        space_matches: bool = True,
        profile: dict[str, Any] | None = None,
    ):
        self.results = results if results is not None else [_result()]
        self.models = models if models is not None else set()
        self.versions = versions if versions is not None else set()
        self._space_matches = space_matches
        self.profile = profile
        self.profile_scopes: list[str | None] = []
        self.searches: list[dict[str, Any]] = []

    def indexed_fingerprint(self, sample_size: int = 200) -> dict[str, set]:
        return {"embedding_models": self.models, "schema_versions": self.versions}

    def indexed_embedding_models(self, sample_size: int = 200) -> set[str]:
        return self.models

    def indexed_schema_versions(self, sample_size: int = 200) -> set[int]:
        return self.versions

    def space_matches(self) -> bool:
        return self._space_matches

    def document_profile(self, document_id: str | None = None):
        self.profile_scopes.append(document_id)
        return self.profile

    def search(self, embedding, n_results: int = 5, document_id=None) -> dict[str, Any]:
        self.searches.append({"n_results": n_results, "document_id": document_id})

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
    document_id: str | None = None,
) -> Retriever:
    return Retriever(
        embedding_model=embedder or FakeEmbedder(),
        database=database,
        query_rewriter=rewriter or FakeRewriter(),
        reranker=reranker or FakeReranker(),
        document_id=document_id,
    )


# --------------------------------------------------------------------------
# Basic shape and telemetry
# --------------------------------------------------------------------------


def test_retrieve_returns_documents_metadata_distances_and_telemetry():
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
    assert result["considered_candidates"] == 2
    assert result["best_distance"] == 0.1


def test_top_k_defaults_to_the_configured_value():
    database = FakeDatabase()

    _retriever(database).retrieve("what?")

    assert database.searches[0]["n_results"] == settings.RETRIEVAL_TOP_K


def test_explicit_top_k_is_passed_through():
    database = FakeDatabase()

    _retriever(database).retrieve("what?", n_results=2)

    assert database.searches[0]["n_results"] == 2


def test_empty_results_have_a_stable_shape():
    database = FakeDatabase(results=[_result(documents=[], metadatas=[], distances=[])])

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == []
    assert result["metadata"] == []
    assert result["distances"] == []
    assert result["considered_candidates"] == 0
    assert result["best_distance"] is None


def test_missing_keys_do_not_crash():
    database = FakeDatabase(results=[{}])

    assert _retriever(database).retrieve("what?")["documents"] == []


def test_question_is_embedded():
    embedder = FakeEmbedder()

    _retriever(FakeDatabase(), embedder).retrieve("a specific question")

    assert embedder.queries == ["a specific question"]


# --------------------------------------------------------------------------
# Index fingerprint: model, schema version and vector space
# --------------------------------------------------------------------------


def test_matching_fingerprint_queries_normally():
    database = FakeDatabase(models={settings.EMBEDDING_MODEL})

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == ["chunk"]
    assert len(database.searches) == 1


def test_mismatched_embedding_model_returns_nothing():
    database = FakeDatabase(models={"some-other-model"})

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == []
    assert database.searches == [], "a foreign index must not be queried"


def test_mismatched_schema_version_returns_nothing():
    database = FakeDatabase(versions={settings.EMBEDDING_SCHEMA_VERSION + 1})

    assert _retriever(database).retrieve("what?")["documents"] == []
    assert database.searches == []


def test_mismatched_vector_space_returns_nothing():
    database = FakeDatabase(space_matches=False)

    assert _retriever(database).retrieve("what?")["documents"] == []
    assert database.searches == [], "distances from another space are meaningless"


def test_unfingerprinted_index_is_still_usable():
    database = FakeDatabase(models=set(), versions=set())

    assert _retriever(database).retrieve("what?")["documents"] == ["chunk"]


def test_index_status_reports_every_check():
    status = _retriever(FakeDatabase(models={"other"})).index_status()

    assert status["embedding_model_match"] is False
    assert status["schema_version_match"] is True
    assert status["vector_space_match"] is True
    assert status["indexed_embedding_models"] == ["other"]


# --------------------------------------------------------------------------
# Relative selection (the fix for the empty-context bug)
# --------------------------------------------------------------------------


def test_selection_is_relative_to_the_best_match(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_RELATIVE_MARGIN", 1.2)
    monkeypatch.setattr(settings, "RETRIEVAL_ABSOLUTE_SLACK", 0.0)
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 2.0)

    database = FakeDatabase(
        results=[
            _result(
                documents=["best", "near", "far"],
                metadatas=[
                    {"page": 1, "chunk": 1},
                    {"page": 2, "chunk": 1},
                    {"page": 3, "chunk": 1},
                ],
                distances=[0.50, 0.55, 0.70],
            )
        ]
    )

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == ["best", "near"]
    assert result["best_distance"] == 0.50


def test_a_broad_question_is_not_emptied_by_an_absolute_cut(monkeypatch):
    # The regression this whole change exists for: every candidate is beyond
    # the old absolute value, but they are all close to the best one.
    monkeypatch.setattr(settings, "RETRIEVAL_RELATIVE_MARGIN", 1.15)
    monkeypatch.setattr(settings, "RETRIEVAL_ABSOLUTE_SLACK", 0.10)
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 1.5)

    database = FakeDatabase(
        results=[
            _result(
                documents=["a", "b", "c"],
                metadatas=[
                    {"page": 1, "chunk": 1},
                    {"page": 2, "chunk": 1},
                    {"page": 3, "chunk": 1},
                ],
                distances=[0.79, 0.85, 0.90],
            )
        ]
    )

    result = _retriever(database).retrieve("what is this document about?")

    assert result["documents"], "a broad question must still reach the prompt"
    assert result["best_distance"] == 0.79


def test_nothing_survives_the_noise_floor_but_the_best_is_kept_anyway(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_RELATIVE_MARGIN", 1.15)
    monkeypatch.setattr(settings, "RETRIEVAL_ABSOLUTE_SLACK", 0.10)
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 0.50)

    database = FakeDatabase(
        results=[
            _result(
                documents=["a", "b"],
                metadatas=[{"page": 1, "chunk": 1}, {"page": 2, "chunk": 1}],
                distances=[0.90, 0.95],
            )
        ]
    )

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == ["a", "b"], (
        "an empty context makes the answer profile-only and ungrounded; "
        "keeping the best candidates is the lesser evil"
    )


def test_noise_floor_caps_a_generous_relative_margin(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_RELATIVE_MARGIN", 5.0)
    monkeypatch.setattr(settings, "RETRIEVAL_ABSOLUTE_SLACK", 5.0)
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 0.60)

    database = FakeDatabase(
        results=[
            _result(
                documents=["a", "b"],
                metadatas=[{"page": 1, "chunk": 1}, {"page": 2, "chunk": 1}],
                distances=[0.50, 0.58],
            )
        ]
    )

    result = _retriever(database).retrieve("what?")

    assert result["documents"] == ["a", "b"]


# --------------------------------------------------------------------------
# Multi-query rewriting
# --------------------------------------------------------------------------


def test_rewritten_query_adds_a_second_search():
    database = FakeDatabase(results=[_result(), _result()])

    embedder = FakeEmbedder()

    _retriever(database, embedder, rewriter=FakeRewriter("better terms")).retrieve(
        "what is the title?"
    )

    assert embedder.queries == ["what is the title?", "better terms"]
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


def test_rerank_runs_when_the_candidate_set_is_flat(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_SKIP_RATIO", 0.6)

    database = FakeDatabase(
        results=[
            _result(
                documents=["a", "b", "c", "d"],
                metadatas=[{"page": i, "chunk": 1} for i in range(4)],
                distances=[0.70, 0.71, 0.72, 0.73],
            )
        ]
    )

    reranker = FakeReranker(order=[3])

    result = _retriever(database, reranker=reranker).retrieve("q", n_results=2)

    assert len(reranker.calls) == 1
    assert result["documents"] == ["d", "a"]


def test_rerank_is_skipped_when_the_best_stands_out(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_SKIP_RATIO", 0.6)

    # Best is 0.20 against a median of 0.75: clearly the right passage, so the
    # extra call would be wasted.
    database = FakeDatabase(
        results=[
            _result(
                documents=["a", "b", "c", "d"],
                metadatas=[{"page": i, "chunk": 1} for i in range(4)],
                distances=[0.20, 0.74, 0.75, 0.76],
            )
        ]
    )

    reranker = FakeReranker(order=[3])

    _retriever(database, reranker=reranker).retrieve("q")

    assert median([0.20, 0.74, 0.75, 0.76]) == 0.745
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
                metadatas=[{"page": i, "chunk": 1} for i in range(3)],
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

    assert database.searches[0]["n_results"] == 20


def test_only_top_k_are_returned_after_reranking(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_SKIP_RATIO", 0.0)

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
# Document scoping
# --------------------------------------------------------------------------


def test_search_is_scoped_to_the_document():
    database = FakeDatabase()

    _retriever(database, document_id="document-a").retrieve("q")

    assert database.searches[0]["document_id"] == "document-a"


def test_search_is_unscoped_when_no_document_is_selected():
    database = FakeDatabase()

    _retriever(database).retrieve("q")

    assert database.searches[0]["document_id"] is None


def test_profile_is_requested_for_the_selected_document():
    database = FakeDatabase(profile={"text": "Title: A", "metadata": {}})

    _retriever(database, document_id="document-a").document_profile()

    assert database.profile_scopes == ["document-a"]


def test_profile_is_returned_from_the_store():
    profile = {"text": "Title: X", "metadata": {"page": 0, "chunk": 0}}

    assert _retriever(FakeDatabase(profile=profile)).document_profile() == profile


def test_profile_is_none_when_not_indexed():
    assert _retriever(FakeDatabase()).document_profile() is None


def test_profile_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "DOCUMENT_PROFILE_IN_CONTEXT", False)

    database = FakeDatabase(profile={"text": "Title: X", "metadata": {}})

    assert _retriever(database).document_profile() is None
    assert database.profile_scopes == [], "no lookup should happen at all"
