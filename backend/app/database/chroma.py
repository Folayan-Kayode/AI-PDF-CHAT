"""Vector store access."""

import logging
import time
from functools import lru_cache
from typing import Any

from chromadb import PersistentClient

from app.core.config import settings
from app.core.exceptions import RetrievalError

logger = logging.getLogger(__name__)

COLLECTION_NAME = "pdf_documents"

STAGING_COLLECTION_NAME = "pdf_documents__staging"

PREVIOUS_COLLECTION_NAME = "pdf_documents__previous"

# Metadata kind for the document profile chunk (see app/rag/summarizer.py).
DOCUMENT_SUMMARY_KIND = "document_summary"


class ChromaDatabase:
    """
    Thin wrapper around a persistent Chroma collection.

    The collection is created once and is NOT wiped on construction. New
    content is written through replace_documents(), which stages the write in
    a separate collection and swaps it in, so a failed write cannot destroy
    the document that is currently indexed.
    """

    def __init__(self, path: str | None = None) -> None:
        self.client = PersistentClient(path=path or settings.CHROMA_DIRECTORY)

        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def _drop(self, name: str) -> None:
        """Delete a collection if it exists."""
        try:
            self.client.delete_collection(name)
        except Exception:
            # Delete-if-present; a missing collection is not an error here.
            pass

    def _names(self) -> set[str]:
        return {collection.name for collection in self.client.list_collections()}

    def reset(self) -> None:
        """Drop and recreate the live collection (used by tests and admin)."""
        self._drop(COLLECTION_NAME)

        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def has_document(self, document_id: str) -> bool:
        """Return True if any chunk with this document_id is already indexed."""
        try:
            result = self.collection.get(
                where={"document_id": document_id},
                limit=1,
            )
        except Exception as exc:
            raise RetrievalError("Could not read the document index.") from exc

        return bool(result.get("ids"))

    def count(self) -> int:
        """Number of chunks currently indexed."""
        try:
            return self.collection.count()
        except Exception as exc:
            raise RetrievalError("Could not read the document index.") from exc

    def indexed_embedding_models(self, sample_size: int = 200) -> set[str]:
        """
        Embedding model names recorded in the index.

        Vectors from a different model live in an incompatible space, so the
        caller compares this against the configured model before trusting
        retrieved chunks.
        """
        try:
            result = self.collection.get(
                limit=sample_size,
                include=["metadatas"],
            )
        except Exception as exc:
            raise RetrievalError("Could not read the document index.") from exc

        metadatas = result.get("metadatas") or []

        return {
            metadata["embedding_model"]
            for metadata in metadatas
            if metadata and metadata.get("embedding_model")
        }

    def document_profile(self) -> dict[str, Any] | None:
        """
        The document profile chunk, if one is indexed.

        The profile is document-level metadata (title, author, publisher), so
        it is supplied to the model directly instead of competing in the
        vector ranking: a question like "what is the title of this book?" does
        not embed close to the profile, even though the profile answers it.
        """
        try:
            result = self.collection.get(
                where={"kind": DOCUMENT_SUMMARY_KIND},
                limit=1,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise RetrievalError("Could not read the document index.") from exc

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        if not documents:
            return None

        return {
            "text": documents[0],
            "metadata": metadatas[0] if metadatas else {},
        }

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        collection: Any = None,
    ) -> None:
        """
        Add chunks in bounded batches, retrying transient failures.

        A single unbounded ``collection.add`` can exceed what one
        SQLite-backed transaction will commit, and a single transient error
        would lose the whole document, so the write is chunked and retried.
        All four lists are sliced together so they cannot drift apart.
        """
        target = collection if collection is not None else self.collection

        batch_size = max(1, settings.CHROMA_WRITE_BATCH_SIZE)
        attempts = max(1, settings.CHROMA_WRITE_MAX_RETRIES)
        base_delay = max(0.0, settings.CHROMA_WRITE_RETRY_BASE_SECONDS)

        for start in range(0, len(ids), batch_size):
            end = start + batch_size

            payload = {
                "ids": ids[start:end],
                "documents": documents[start:end],
                "embeddings": embeddings[start:end],
                "metadatas": metadatas[start:end],
            }

            self._add_batch(
                target,
                payload,
                attempts=attempts,
                base_delay=base_delay,
                start=start,
                end=end,
            )

    @staticmethod
    def _add_batch(
        collection: Any,
        payload: dict[str, Any],
        attempts: int,
        base_delay: float,
        start: int,
        end: int,
    ) -> None:
        for attempt in range(1, attempts + 1):
            try:
                collection.add(**payload)
                return

            except Exception as exc:
                if attempt >= attempts:
                    raise RetrievalError("Could not write to the document index.") from exc

                delay = base_delay * (2 ** (attempt - 1))

                logger.warning(
                    "index write failed for chunks %s-%s (attempt %s/%s): %s; retrying in %.1fs",
                    start,
                    end,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )

                if delay:
                    time.sleep(delay)

    def replace_documents(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Replace the indexed document with new content.

        The new chunks are written to a staging collection first, and only
        swapped in once every batch has been committed. If anything fails the
        staging collection is dropped and the previously indexed document
        stays queryable, so a bad upload cannot leave the user with nothing.
        """
        self._drop(STAGING_COLLECTION_NAME)

        staging = self.client.get_or_create_collection(name=STAGING_COLLECTION_NAME)

        try:
            self.add_documents(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                collection=staging,
            )
        except RetrievalError:
            self._drop(STAGING_COLLECTION_NAME)
            raise

        had_previous = COLLECTION_NAME in self._names()

        try:
            if had_previous:
                # Move the live collection aside first so the live name is
                # never left empty, then promote the staging collection.
                self.client.get_collection(COLLECTION_NAME).modify(name=PREVIOUS_COLLECTION_NAME)

            staging.modify(name=COLLECTION_NAME)

        except Exception as exc:
            logger.exception("index swap failed; attempting to roll back")

            try:
                if PREVIOUS_COLLECTION_NAME in self._names():
                    self.client.get_collection(PREVIOUS_COLLECTION_NAME).modify(
                        name=COLLECTION_NAME
                    )
            except Exception:
                logger.exception("index swap rollback failed")

            self._drop(STAGING_COLLECTION_NAME)

            raise RetrievalError("Could not activate the new document index.") from exc

        self._drop(PREVIOUS_COLLECTION_NAME)

        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

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
            raise RetrievalError("Could not query the document index.") from exc


@lru_cache(maxsize=1)
def get_database() -> ChromaDatabase:
    """
    Process-wide database handle.

    Opening a PersistentClient per request re-reads the index every time, so
    one shared instance is used for the life of the process. Tests can call
    get_database.cache_clear().
    """
    return ChromaDatabase()
