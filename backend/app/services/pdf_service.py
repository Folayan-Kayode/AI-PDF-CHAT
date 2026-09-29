import hashlib

from app.core.exceptions import PDFProcessingError
from app.database.chroma import ChromaDatabase
from app.rag.embeddings import EmbeddingModel
from app.rag.loader import PDFLoader
from app.rag.splitter import TextSplitter


class PDFService:

    @staticmethod
    def file_hash(pdf_path) -> str:
        """Stable content hash used for chunk IDs and duplicate detection."""
        digest = hashlib.sha256()

        with open(pdf_path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()

    @staticmethod
    def process(pdf_path):
        document_id = PDFService.file_hash(pdf_path)

        database = ChromaDatabase()

        if database.has_document(document_id):
            return {
                "pages": [],
                "chunks": [],
                "document_id": document_id,
                "duplicate": True,
            }

        loader = PDFLoader(pdf_path)
        pages = loader.load()

        splitter = TextSplitter()
        chunks = splitter.split_pages(pages)

        if not chunks:
            raise PDFProcessingError(
                "No usable text chunks could be created from this PDF."
            )

        embedding_model = EmbeddingModel()

        texts = []
        ids = []
        metadatas = []

        for chunk in chunks:
            texts.append(chunk["text"])

            ids.append(
                f"{document_id}_{chunk['page']}_{chunk['chunk']}"
            )

            metadatas.append(
                {
                    "page": chunk["page"],
                    "chunk": chunk["chunk"],
                    "document_id": document_id,
                }
            )

        embeddings = embedding_model.embed_documents(texts)

        database.reset()

        database.add_documents(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {
            "pages": pages,
            "chunks": chunks,
            "document_id": document_id,
            "duplicate": False,
        }
