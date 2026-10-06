"""
The retrieval invariants that must never regress.

The failure this file exists for: an absolute distance threshold removed every
candidate for broad questions, so the answer was generated from the document
profile alone while every existing metric (hit@5, accuracy, cost) still looked
healthy. These tests are offline and deterministic, and the plan treats
deleting one as a release blocker.
"""

import logging

from app.core.config import settings
from app.database.chroma import ChromaDatabase
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever


class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return [0.0, 1.0, 0.0]


class IdenticalEmbedder:
    """Returns a vector identical to the stored one: distance 0."""

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FarDatabase:
    """A store whose candidates are all beyond any sane cut."""

    def __init__(self, distances):
        self.distances = distances
        self.searches = 0

    def indexed_fingerprint(self, sample_size=200):
        return {"embedding_models": set(), "schema_versions": set()}

    def space_matches(self):
        return True

    def document_profile(self, document_id=None):
        return None

    def search(self, embedding, n_results=5, document_id=None):
        self.searches += 1

        return {
            "documents": [[f"chunk {index}" for index in range(len(self.distances))]],
            "metadatas": [
                [{"page": index + 1, "chunk": 1} for index in range(len(self.distances))]
            ],
            "distances": [self.distances],
        }


class FakeGenerator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "an answer"


def _store(tmp_path, documents):
    """A real Chroma store with the given {document_id: vector} entries."""
    database = ChromaDatabase(path=str(tmp_path / "chroma"))

    database.add_documents(
        ids=[f"{document_id}_1_1" for document_id in documents],
        documents=[f"content of {document_id}" for document_id in documents],
        embeddings=list(documents.values()),
        metadatas=[
            {
                "page": 1,
                "chunk": 1,
                "document_id": document_id,
                "embedding_model": settings.EMBEDDING_MODEL,
                "schema_version": settings.EMBEDDING_SCHEMA_VERSION,
            }
            for document_id in documents
        ],
    )

    return database


# --------------------------------------------------------------------------
# The single assertion that would have caught the shipped regression
# --------------------------------------------------------------------------


def test_no_query_against_a_non_empty_index_produces_empty_context(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Zero passages plus a confident answer is the failure mode."""
    # The query is orthogonal to what is stored: cosine distance 1.0.
    database = _store(tmp_path, {"doc-a": [1.0, 0.0, 0.0]})

    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 0.5)
    monkeypatch.setattr(settings, "RETRIEVAL_RELATIVE_MARGIN", 1.0)
    monkeypatch.setattr(settings, "RETRIEVAL_ABSOLUTE_SLACK", 0.0)

    # FakeEmbedder returns [0, 1, 0], orthogonal to the stored [1, 0, 0].
    retriever = Retriever(embedding_model=FakeEmbedder(), database=database)

    with caplog.at_level(logging.WARNING):
        result = retriever.retrieve("something entirely unrelated")

    assert database.count() > 0, "the index must be non-empty for this test"
    assert result["documents"], (
        "a non-empty index must always send at least one passage; an empty "
        "context is answered from the profile alone and reads as grounded"
    )
    assert "no candidate passed the distance filter" in caplog.text


def test_never_empty_invariant_holds_with_a_fake_store(monkeypatch, caplog):
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 0.01)
    monkeypatch.setattr(settings, "RETRIEVAL_RELATIVE_MARGIN", 1.0)
    monkeypatch.setattr(settings, "RETRIEVAL_ABSOLUTE_SLACK", 0.0)

    retriever = Retriever(
        embedding_model=FakeEmbedder(),
        database=FarDatabase([0.9, 0.95, 0.99]),
    )

    with caplog.at_level(logging.WARNING):
        result = retriever.retrieve("anything")

    assert len(result["documents"]) == 3
    assert "sending the best" in caplog.text


def test_best_k_is_capped_by_top_k_when_nothing_passes(monkeypatch):
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_DISTANCE", 0.01)
    monkeypatch.setattr(settings, "RETRIEVAL_RELATIVE_MARGIN", 1.0)
    monkeypatch.setattr(settings, "RETRIEVAL_ABSOLUTE_SLACK", 0.0)

    retriever = Retriever(
        embedding_model=FakeEmbedder(),
        database=FarDatabase([0.9, 0.91, 0.92, 0.93, 0.94, 0.95]),
    )

    assert len(retriever.retrieve("anything")["documents"]) <= settings.RETRIEVAL_TOP_K


def test_telemetry_exposes_how_much_context_was_supplied():
    retriever = Retriever(
        embedding_model=FakeEmbedder(),
        database=FarDatabase([0.2, 0.3]),
    )

    result = retriever.retrieve("anything")

    assert result["considered_candidates"] == 2
    assert result["best_distance"] == 0.2
    assert result["candidate_distances"] == [0.2, 0.3]


def test_an_empty_selection_still_reports_what_was_considered():
    """
    "Nothing was found" and "everything was discarded" are different failures.

    The second is the one that shipped, so an empty result must not report a
    bare zero and hide the candidates that were discarded.
    """

    class DropEverything(Retriever):
        def _select(self, ordered, top_k):
            return []

    retriever = DropEverything(
        embedding_model=FakeEmbedder(),
        database=FarDatabase([0.2, 0.4, 0.6]),
    )

    result = retriever.retrieve("anything")

    assert result["documents"] == []
    assert result["considered_candidates"] == 3
    assert result["candidate_distances"] == [0.2, 0.4, 0.6]
    assert result["best_distance"] == 0.2
    assert result["cut_distance"] is not None


# --------------------------------------------------------------------------
# The context budget cannot empty the prompt either
# --------------------------------------------------------------------------


def test_context_budget_never_removes_everything(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONTEXT_CHARS", 1)
    monkeypatch.setattr(settings, "MAX_CONTEXT_TOKENS", 1)

    retriever = Retriever(
        embedding_model=FakeEmbedder(),
        database=FarDatabase([0.1, 0.2]),
    )

    generator = FakeGenerator()

    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    pipeline.ask("anything")

    prompt = generator.prompts[0]
    body = prompt.split("\n<document>\n", 1)[1].split("\n</document>", 1)[0]

    assert body.strip(), "at least one passage must always reach the prompt"


def test_an_empty_profile_alone_still_abstains(tmp_path):
    # The inverse of the invariant: with nothing at all, abstaining is correct.
    database = ChromaDatabase(path=str(tmp_path / "chroma"))

    retriever = Retriever(embedding_model=FakeEmbedder(), database=database)

    generator = FakeGenerator()

    result = RAGPipeline(retriever=retriever, generator=generator).ask("anything")

    assert result["sources"] == []
    assert result["abstained"] is True
    assert generator.prompts == [], "nothing to answer from, so no model call"
