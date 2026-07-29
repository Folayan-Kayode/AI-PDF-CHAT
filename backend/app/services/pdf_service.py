from app.database.chroma import ChromaDatabase
from app.rag.embeddings import EmbeddingModel
from app.rag.loader import PDFLoader
from app.rag.splitter import TextSplitter


class PDFService:

    @staticmethod
    def process(pdf_path):

        loader = PDFLoader(pdf_path)
        pages = loader.load()

        splitter = TextSplitter()
        chunks = splitter.split_pages(pages)

        embedding_model = EmbeddingModel()

        texts = []

        ids = []

        metadatas = []

        for i, chunk in enumerate(chunks):

            texts.append(chunk["text"])

            ids.append(f"chunk_{i}")

            metadatas.append(
                {
                    "page": chunk["page"],
                    "chunk": chunk["chunk"],
                }
            )

        embeddings = embedding_model.embed_documents(texts)

        database = ChromaDatabase()

        database.add_documents(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {
            "pages": pages,
            "chunks": chunks,
        }