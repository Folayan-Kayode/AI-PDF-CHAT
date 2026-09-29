"""End-to-end PDF ingestion."""

import hashlib
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import DocumentTooLargeError, PDFProcessingError
from app.database.chroma import ChromaDatabase, get_database
from app.rag.embeddings import EmbeddingModel, get_embedding_model
from app.rag.loader import PDFLoader
from app.rag.splitter import TextSplitter


class PDFService:
    """Loads, chunks, embeds and indexes an uploaded document."""

    @staticmethod
    def file_hash(pdf_path: str | Path) -> str:
        """Stable content hash used for chunk IDs and duplicate detection."""
        digest = hashlib.sha256()

        with open(pdf_path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)

        return digest.hexdigest()

    @classmethod
    def process(
        cls,
        pdf_path: str | Path,
        database: ChromaDatabase | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a PDF into the vector store.

        The document replaces the current index (single-document mode).
        Returns page/chunk counts, or duplicate=True without re-embedding
        when the same content has already been ingested.
        """
        document_id = cls.file_hash(pdf_path)

        database = database or get_database()
        embedding_model = embedding_model or get_embedding_model()

        if database.has_document(document_id):
            return {
                "pages": [],
                "chunks": [],
                "document_id": document_id,
                "duplicate": True,
            }

        pages = PDFLoader(pdf_path).load()

        chunks = TextSplitter().split_pages(pages)

        if not chunks:
            raise PDFProcessingError(
                "No usable text chunks could be created from this PDF."
            )

        if len(chunks) > settings.MAX_CHUNKS_PER_DOCUMENT:
            raise DocumentTooLargeError(
                f"This document produced {len(chunks)} chunks, which exceeds "
                f"the {settings.MAX_CHUNKS_PER_DOCUMENT}-chunk limit."
            )

        texts = [chunk["text"] for chunk in chunks]

        ids = [
            f"{document_id}_{chunk['page']}_{chunk['chunk']}"
            for chunk in chunks
        ]

        metadatas = [
            {
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "document_id": document_id,
            }
            for chunk in chunks
        ]

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
