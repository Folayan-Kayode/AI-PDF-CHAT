"""Retrieval over the vector store."""

from typing import Any

from app.core.config import settings
from app.database.chroma import ChromaDatabase, get_database
from app.rag.embeddings import EmbeddingModel, get_embedding_model


class Retriever:
    """Embeds a question and returns the closest chunks."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        database: ChromaDatabase | None = None,
    ) -> None:
        self.embedding_model = embedding_model or get_embedding_model()

        self.database = database or get_database()

    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
    ) -> dict[str, Any]:
        """Return documents, metadata and distances for the closest chunks."""
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
