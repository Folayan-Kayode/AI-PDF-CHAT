from app.database.chroma import ChromaDatabase
from app.rag.embeddings import EmbeddingModel


class Retriever:

    def __init__(self):

        self.embedding_model = EmbeddingModel()

        self.database = ChromaDatabase()

    def retrieve(
        self,
        question: str,
        n_results: int = 5,
    ):

        query_embedding = self.embedding_model.embed_query(question)

        results = self.database.search(
            embedding=query_embedding,
            n_results=n_results,
        )

        return {
    "documents": results["documents"][0],
    "metadata": results["metadatas"][0],
    "distances": results["distances"][0]
}