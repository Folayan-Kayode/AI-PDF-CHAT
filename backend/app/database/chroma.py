from chromadb import PersistentClient


class ChromaDatabase:

    def __init__(self, path="chroma_db"):
        self.client = PersistentClient(path=path)

        try:
            self.client.delete_collection("pdf_documents")
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name="pdf_documents"
        )

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