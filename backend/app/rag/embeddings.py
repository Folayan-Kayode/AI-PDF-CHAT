from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings


class EmbeddingModel:

    def __init__(self):

        self.model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    def embed_documents(self, texts):

        return self.model.embed_documents(texts)

    def embed_query(self, query):

        return self.model.embed_query(query)