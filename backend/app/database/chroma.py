"""Vector store access."""

from functools import lru_cache
from typing import Any

from chromadb import PersistentClient

from app.core.config import settings
from app.core.exceptions import RetrievalError

COLLECTION_NAME = "pdf_documents"


class ChromaDatabase:
    """
    Thin wrapper around a persistent Chroma collection.

    The collection is created once and is NOT wiped on construction.
    Call reset() explicitly when a fresh ingest should replace the
    existing index (single-document mode for now).
    """

    def __init__(self, path: str | None = None) -> None:
        self.client = PersistentClient(
            path=path or settings.CHROMA_DIRECTORY
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def reset(self) -> None:
        """Drop and recreate the collection. Called by the upload step only."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            # The collection may not exist yet; recreating it is the goal
            # either way.
            pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def has_document(self, document_id: str) -> bool:
        """Return True if any chunk with this document_id is already indexed."""
        try:
            result = self.collection.get(
                where={"document_id": document_id},
                limit=1,
            )
        except Exception as exc:
            raise RetrievalError(
                "Could not read the document index."
            ) from exc

        return bool(result.get("ids"))

    def count(self) -> int:
        """Number of chunks currently indexed."""
        try:
            return self.collection.count()
        except Exception as exc:
            raise RetrievalError(
                "Could not read the document index."
            ) from exc

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add chunks to the collection."""
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise RetrievalError(
                "Could not write to the document index."
            ) from exc

    def search(
        self,
        embedding: list[float],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Return the closest chunks to the given embedding."""
        try:
            return self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
            )
        except Exception as exc:
            raise RetrievalError(
                "Could not query the document index."
            ) from exc


@lru_cache(maxsize=1)
def get_database() -> ChromaDatabase:
    """
    Process-wide database handle.

    Opening a PersistentClient per request re-reads the index every time, so
    one shared instance is used for the life of the process. Tests can call
    get_database.cache_clear().
    """
    return ChromaDatabase()
