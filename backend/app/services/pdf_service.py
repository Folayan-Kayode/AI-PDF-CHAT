"""End-to-end PDF ingestion."""

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    DocumentTooLargeError,
    IngestionInProgressError,
    PDFProcessingError,
    RetrievalError,
)
from app.database.chroma import DOCUMENT_SUMMARY_KIND, ChromaDatabase, get_database
from app.rag.citations import PROFILE_PAGE
from app.rag.embeddings import EmbeddingModel, get_embedding_model
from app.rag.loader import PDFLoader
from app.rag.splitter import TextSplitter
from app.rag.summarizer import DocumentSummarizer

logger = logging.getLogger(__name__)

# Ingestion replaces a single-document index, so two of them running at once
# would interleave their resets and writes and produce a mixed index. This
# guard is process-wide: the application assumes a single uvicorn worker
# (see README).
_INGESTION_LOCK = threading.Lock()


def _profile_chunk(profile: str) -> dict[str, Any]:
    """The chunk that lets a document answer questions about itself.

    Page 0 marks it as not being a real page of the document, and the answer
    prompt asks for it to be cited as ``[document]``.
    """
    return {
        "text": f"Document profile:\n{profile}",
        "page": PROFILE_PAGE,
        "chunk": 0,
        "kind": DOCUMENT_SUMMARY_KIND,
    }


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
        summarizer: DocumentSummarizer | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a PDF into the vector store.

        Serialised against other ingestions; a concurrent call is rejected
        with a 409 rather than queued, because ingestion can take minutes.
        The previous index survives any failure, so a bad upload never costs
        the user the document they already had.
        """
        database = database or get_database()
        embedding_model = embedding_model or get_embedding_model()

        timeout = max(0.0, settings.INGESTION_LOCK_TIMEOUT_SECONDS)

        if not _INGESTION_LOCK.acquire(timeout=timeout):
            raise IngestionInProgressError(
                "Another document is currently being ingested. Wait for it to finish and try again."
            )

        try:
            return cls._ingest(pdf_path, database, embedding_model, summarizer)
        finally:
            _INGESTION_LOCK.release()

    @classmethod
    def _ingest(
        cls,
        pdf_path: str | Path,
        database: ChromaDatabase,
        embedding_model: EmbeddingModel,
        summarizer: DocumentSummarizer | None = None,
    ) -> dict[str, Any]:
        document_id = cls.file_hash(pdf_path)

        if database.has_document(document_id):
            return {
                "pages": [],
                "chunks": [],
                "document_id": document_id,
                "duplicate": True,
                "index_replaced": False,
            }

        pages = PDFLoader(pdf_path).load()

        chunks = TextSplitter().split_pages(pages)

        profile = (summarizer or DocumentSummarizer()).summarize(pages)

        if profile:
            # Prepended so it is the first chunk of the document; chunk 0
            # keeps it distinct from the splitter's 1-based numbering.
            chunks = [_profile_chunk(profile), *chunks]

        if not chunks:
            raise PDFProcessingError("No usable text chunks could be created from this PDF.")

        if len(chunks) > settings.MAX_CHUNKS_PER_DOCUMENT:
            raise DocumentTooLargeError(
                f"This document produced {len(chunks)} chunks, which exceeds "
                f"the {settings.MAX_CHUNKS_PER_DOCUMENT}-chunk limit."
            )

        texts = [chunk["text"] for chunk in chunks]

        ids = [f"{document_id}_{chunk['page']}_{chunk['chunk']}" for chunk in chunks]

        # The embedding fingerprint lets the retriever detect an index built
        # in a different vector space instead of returning meaningless hits.
        metadatas = [
            {
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "kind": chunk.get("kind", "content"),
                "document_id": document_id,
                "embedding_model": settings.EMBEDDING_MODEL,
                "schema_version": settings.EMBEDDING_SCHEMA_VERSION,
            }
            for chunk in chunks
        ]

        embeddings = embedding_model.embed_documents(texts)

        try:
            database.replace_documents(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        except RetrievalError as exc:
            raise RetrievalError(
                f"{exc.detail} The previously indexed document is unchanged.",
                service="vector_store",
            ) from exc

        logger.info(
            "indexed document_id=%s chunks=%s",
            document_id,
            len(chunks),
        )

        return {
            "pages": pages,
            "chunks": chunks,
            "document_id": document_id,
            "duplicate": False,
            "index_replaced": True,
        }
