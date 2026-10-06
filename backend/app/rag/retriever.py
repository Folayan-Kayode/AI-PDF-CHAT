"""Retrieval over the vector store."""

import logging
from typing import Any

from app.core.config import settings
from app.database.chroma import ChromaDatabase, get_database
from app.rag.embeddings import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)


class Retriever:
    """Embeds a question and returns the closest chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        database: ChromaDatabase | None = None,
    ) -> None:
        self.embedding_model = embedding_model or get_embedding_model()

        self.database = database or get_database()

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

            return {
                "documents": [],
                "metadata": [],
                "distances": [],
            }

        top_k = n_results or settings.RETRIEVAL_TOP_K

        query_embedding = self.embedding_model.embed_query(question)

        results = self.database.search(
            embedding=query_embedding,
            n_results=top_k,
        )

        return {
            "documents": (results.get("documents") or [[]])[0],
            "metadata": (results.get("metadatas") or [[]])[0],
            "distances": (results.get("distances") or [[]])[0],
        }
