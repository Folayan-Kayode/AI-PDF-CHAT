from chromadb import PersistentClient

from app.core.config import settings

COLLECTION_NAME = "pdf_documents"


class ChromaDatabase:
    """
    Thin wrapper around a persistent Chroma collection.

    The collection is created once and is NOT wiped on construction.
    Call reset() explicitly when a fresh ingest should replace the
    existing index (single-document mode for now).
    """

    def __init__(self, path: str = None):
        self.client = PersistentClient(
            path=path or settings.CHROMA_DIRECTORY
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def reset(self):
        """Drop and recreate the collection. Called by the upload step only."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def has_document(self, document_id: str) -> bool:
        """Return True if any chunk with this document_id is already indexed."""
        result = self.collection.get(
            where={"document_id": document_id},
            limit=1,
        )

        return bool(result.get("ids"))

    def count(self) -> int:
        return self.collection.count()

    def add_documents(
        self,
        ids,
        documents,
        embeddings,
        metadatas,
    ):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, embedding, n_results=5):
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
