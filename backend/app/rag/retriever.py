"""
Retrieval over the vector store.

Selection is *relative* to the best candidate for each query. An absolute
distance cut cannot work across documents: a broad question such as "what is
this document about?" sits far from every chunk in a way a specific question
does not, so a constant threshold deletes exactly the questions that need
help. A constant survives only as a noise floor.
"""

import logging
from dataclasses import dataclass
from statistics import median
from typing import Any

from app.core.config import settings
from app.database.chroma import ChromaDatabase, get_database
from app.rag.embeddings import EmbeddingModel, get_embedding_model
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import Reranker

logger = logging.getLogger(__name__)

_EMPTY_RESULT: dict[str, Any] = {
    "documents": [],
    "metadata": [],
    "distances": [],
    "considered_candidates": 0,
    "best_distance": None,
}


@dataclass
class Candidate:
    """One retrieved chunk with its distance from the query."""

    document: str
    metadata: dict[str, Any]
    distance: float

    @property
    def key(self) -> tuple:
        """Identity used to de-duplicate across multiple queries."""
        metadata = self.metadata or {}

        identity = (
            metadata.get("document_id"),
            metadata.get("page_start", metadata.get("page")),
            metadata.get("chunk"),
        )

        if any(part is not None for part in identity):
            return identity

        return (self.document,)


class Retriever:
    """Embeds a question and returns the closest chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        database: ChromaDatabase | None = None,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        document_id: str | None = None,
    ) -> None:
        self.embedding_model = embedding_model or get_embedding_model()

        self.database = database or get_database()

        self.query_rewriter = query_rewriter or QueryRewriter()

        self.reranker = reranker or Reranker()

        self.document_id = document_id

    # ------------------------------------------------------------------
    # Index health
    # ------------------------------------------------------------------

    def index_status(self) -> dict[str, Any]:
        """
        Whether the index can be trusted for this configuration.

        Three independent things must agree: the embedding model, the schema
        version (which is bumped when the vector space changes) and the
        collection's actual distance space. An unindexed collection reports
        True for all of them.
        """
        fingerprint = self.database.indexed_fingerprint()

        models = fingerprint["embedding_models"]
        versions = fingerprint["schema_versions"]

        return {
            "embedding_model_match": not models or settings.EMBEDDING_MODEL in models,
            "schema_version_match": (not versions or settings.EMBEDDING_SCHEMA_VERSION in versions),
            "vector_space_match": self.database.space_matches(),
            "indexed_embedding_models": sorted(models),
            "indexed_schema_versions": sorted(versions),
            "configured_schema_version": settings.EMBEDDING_SCHEMA_VERSION,
            "configured_vector_space": settings.CHROMA_SPACE,
        }

    def index_usable(self) -> bool:
        """Whether the index matches the configured model and space."""
        status = self.index_status()

        return bool(
            status["embedding_model_match"]
            and status["schema_version_match"]
            and status["vector_space_match"]
        )

    # Backwards-compatible alias: this guard used to be about the model only.
    def indexed_embedding_model_matches(self) -> bool:
        return bool(self.index_status()["embedding_model_match"])

    # ------------------------------------------------------------------
    # Document profile
    # ------------------------------------------------------------------

    def document_profile(self) -> dict[str, Any] | None:
        """
        The indexed profile of the current document, if there is one.

        It is returned directly rather than searched for: a metadata question
        ("what is the title of this book?") does not embed close to the
        profile text, even though the profile is what answers it.
        """
        if not settings.DOCUMENT_PROFILE_IN_CONTEXT:
            return None

        return self.database.document_profile(self.document_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
    ) -> dict[str, Any]:
        """Return documents, metadata, distances and retrieval telemetry."""
        if not self.index_usable():
            logger.warning(
                "the index does not match the configured model or vector "
                "space; treating it as empty. Re-upload the document. %s",
                self.index_status(),
            )

            return dict(_EMPTY_RESULT)

        top_k = n_results or settings.RETRIEVAL_TOP_K

        candidates, considered = self._gather_candidates(question, top_k)

        if not candidates:
            return dict(_EMPTY_RESULT)

        if self._should_rerank(candidates, top_k):
            candidates = self._apply_rerank(question, candidates)

        selected = candidates[:top_k]

        return {
            "documents": [candidate.document for candidate in selected],
            "metadata": [candidate.metadata for candidate in selected],
            "distances": [candidate.distance for candidate in selected],
            "considered_candidates": considered,
            "best_distance": selected[0].distance if selected else None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _gather_candidates(
        self,
        question: str,
        top_k: int,
    ) -> tuple[list[Candidate], int]:
        """
        Search with the question and a rewritten variant, then merge.

        Searching twice keeps the original wording in play, so a poor rewrite
        cannot lose a chunk the original query would have found.

        Returns the selected candidates and how many were considered before
        selection, which is what makes a silent retrieval failure visible.
        """
        queries = [question]

        rewritten = self.query_rewriter.rewrite(question)

        if rewritten and rewritten.strip().lower() != question.strip().lower():
            queries.append(rewritten)

            logger.info("retrieving with a rewritten query: %s", rewritten)

        candidate_k = (
            max(top_k, settings.RETRIEVAL_CANDIDATES) if settings.RERANK_ENABLED else top_k
        )

        merged: dict[tuple, Candidate] = {}

        for query in queries:
            embedding = self.embedding_model.embed_query(query)

            results = self.database.search(
                embedding=embedding,
                n_results=candidate_k,
                document_id=self.document_id,
            )

            for candidate in _to_candidates(results):
                existing = merged.get(candidate.key)

                if existing is None or candidate.distance < existing.distance:
                    merged[candidate.key] = candidate

        ordered = sorted(merged.values(), key=lambda candidate: candidate.distance)

        return self._select(ordered, top_k), len(ordered)

    @staticmethod
    def _cut(best_distance: float) -> float:
        """
        The largest distance still counted as relevant for this query.

        Relative to the best match, with both a multiplicative margin and an
        additive slack (a pure multiplier is too tight when the best distance
        is small), then capped by the noise floor.
        """
        margin = best_distance * settings.RETRIEVAL_RELATIVE_MARGIN
        slack = best_distance + settings.RETRIEVAL_ABSOLUTE_SLACK

        return min(max(margin, slack), settings.RETRIEVAL_MAX_DISTANCE)

    def _select(
        self,
        ordered: list[Candidate],
        top_k: int,
    ) -> list[Candidate]:
        """
        Apply the relative cut, but never return nothing.

        Chroma returning candidates and the pipeline sending none is the
        worst outcome available: the answer is then generated from the profile
        alone and reads as confidently grounded. If the filter removes
        everything, keep the best few and say so.
        """
        if not ordered:
            return []

        best = ordered[0].distance
        cut = self._cut(best)

        kept = [candidate for candidate in ordered if candidate.distance <= cut]

        if not kept:
            logger.warning(
                "no candidate passed the distance filter (best %.3f, cut %.3f); "
                "sending the best %d anyway so the answer is grounded in the "
                "document rather than the profile alone",
                best,
                cut,
                min(top_k, len(ordered)),
            )

            return ordered[: max(1, top_k)]

        return kept

    @staticmethod
    def _should_rerank(candidates: list[Candidate], top_k: int) -> bool:
        """
        Rerank only when it can change the outcome and is worth the call.

        "Already confident" is a relative idea: it means the best match stands
        out from the rest of *this query's* candidates, not that it beats a
        constant that happened to suit one corpus.
        """
        if not settings.RERANK_ENABLED or len(candidates) <= top_k:
            return False

        typical = median(candidate.distance for candidate in candidates)

        if typical <= 0:
            return False

        return candidates[0].distance > typical * settings.RERANK_SKIP_RATIO

    def _apply_rerank(
        self,
        question: str,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        order = self.reranker.rerank(
            question,
            [candidate.document for candidate in candidates],
        )

        if not order:
            return candidates

        ranked = [candidates[index] for index in order]

        ranked_indices = set(order)

        ranked.extend(
            candidate for index, candidate in enumerate(candidates) if index not in ranked_indices
        )

        return ranked


def _to_candidates(results: dict[str, Any]) -> list[Candidate]:
    """Flatten a Chroma query result into candidates."""
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    candidates: list[Candidate] = []

    # Chroma returns aligned lists; strict=False keeps this tolerant of a
    # short or missing list rather than raising mid-request.
    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        if not document:
            continue

        candidates.append(
            Candidate(
                document=document,
                metadata=metadata or {},
                distance=float(distance) if distance is not None else float("inf"),
            )
        )

    return candidates
