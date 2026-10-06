"""Retrieval over the vector store."""

import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.database.chroma import ChromaDatabase, get_database
from app.rag.embeddings import EmbeddingModel, get_embedding_model
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import Reranker

logger = logging.getLogger(__name__)

_EMPTY_RESULT: dict[str, list[Any]] = {
    "documents": [],
    "metadata": [],
    "distances": [],
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
            metadata.get("page"),
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
    ) -> None:
        self.embedding_model = embedding_model or get_embedding_model()

        self.database = database or get_database()

        self.query_rewriter = query_rewriter or QueryRewriter()

        self.reranker = reranker or Reranker()

    def indexed_embedding_model_matches(self) -> bool:
        """
        Whether the index was built with the configured embedding model.

        Query vectors from one model are meaningless against an index built
        by another, and the failure is silent: Chroma simply returns
        nearest neighbours in an incompatible space. An unindexed or
        unfingerprinted collection reports True so callers are unaffected.
        """
        indexed_models = self.database.indexed_embedding_models()

        if not indexed_models:
            return True

        return settings.EMBEDDING_MODEL in indexed_models

    def document_profile(self) -> dict[str, Any] | None:
        """
        The indexed profile of the document, if there is one.

        It is returned directly rather than searched for: a metadata question
        ("what is the title of this book?") does not embed close to the
        profile text, even though the profile is what answers it.
        """
        if not settings.DOCUMENT_PROFILE_IN_CONTEXT:
            return None

        return self.database.document_profile()

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
    ) -> dict[str, Any]:
        """Return documents, metadata and distances for the closest chunks."""
        if not self.indexed_embedding_model_matches():
            logger.warning(
                "the index was built with a different embedding model than "
                "%s; treating the index as empty",
                settings.EMBEDDING_MODEL,
            )

            return dict(_EMPTY_RESULT)

        top_k = n_results or settings.RETRIEVAL_TOP_K

        candidates = self._gather_candidates(question, top_k)

        if not candidates:
            return dict(_EMPTY_RESULT)

        if self._should_rerank(candidates, top_k):
            candidates = self._apply_rerank(question, candidates)

        selected = candidates[:top_k]

        return {
            "documents": [candidate.document for candidate in selected],
            "metadata": [candidate.metadata for candidate in selected],
            "distances": [candidate.distance for candidate in selected],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _gather_candidates(self, question: str, top_k: int) -> list[Candidate]:
        """
        Search with the question and a rewritten variant, then merge.

        Searching twice keeps the original wording in play, so a poor
        rewrite cannot lose a chunk the original query would have found.
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
            )

            for candidate in _to_candidates(results):
                existing = merged.get(candidate.key)

                if existing is None or candidate.distance < existing.distance:
                    merged[candidate.key] = candidate

        ordered = sorted(merged.values(), key=lambda candidate: candidate.distance)

        kept = [
            candidate
            for candidate in ordered
            if candidate.distance <= settings.RETRIEVAL_MAX_DISTANCE
        ]

        if not kept and ordered:
            logger.info(
                "all %s candidates were weaker than RETRIEVAL_MAX_DISTANCE=%.2f",
                len(ordered),
                settings.RETRIEVAL_MAX_DISTANCE,
            )

        return kept

    @staticmethod
    def _should_rerank(candidates: list[Candidate], top_k: int) -> bool:
        """
        Rerank only when it can change the outcome and is worth the call.

        Already-confident retrieval (a close best match) is left alone, as is
        a candidate list no larger than the number of results wanted.
        """
        if not settings.RERANK_ENABLED or len(candidates) <= top_k:
            return False

        return candidates[0].distance > settings.RERANK_SKIP_DISTANCE

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
