"""End-to-end PDF ingestion."""

import hashlib
import logging
import math
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import (
    DocumentTooLargeError,
    IngestBudgetExceededError,
    IngestionInProgressError,
    PDFProcessingError,
    RetrievalError,
)
from app.database.chroma import DOCUMENT_SUMMARY_KIND, ChromaDatabase, get_database
from app.database.registry import get_registry
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

#: The registry metric the daily ingest budget is spent against.
_EMBEDDING_BATCH_METRIC = "embedding_batches"


def _utc_day(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m-%d")


def _seconds_until_utc_midnight(now: datetime | None = None) -> float:
    """Seconds until the daily budget window resets."""
    now = now or datetime.now(UTC)

    tomorrow = (now + timedelta(days=1)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    return max(1.0, (tomorrow - now).total_seconds())


def _profile_chunk(profile: str) -> dict[str, Any]:
    """The chunk that lets a document answer questions about itself.

    Page 0 marks it as not being a real page of the document, and the answer
    prompt asks for it to be cited as ``[document]``.
    """
    return {
        "text": f"Document profile:\n{profile}",
        "page": PROFILE_PAGE,
        "page_start": PROFILE_PAGE,
        "page_end": PROFILE_PAGE,
        "chunk": 0,
        "kind": DOCUMENT_SUMMARY_KIND,
    }


def _record_document(**record: Any) -> None:
    """
    Remember what the index was built from.

    A registry failure must not fail an ingest that otherwise succeeded: the
    index is authoritative and the registry is how /ready describes it.
    """
    try:
        get_registry().replace_with(**record)
    except Exception:
        logger.exception("could not record the document in the registry")


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

    @staticmethod
    def _enforce_ingest_budget(batches: int) -> None:
        """
        Refuse an ingest that would exceed the daily embedding budget.

        Checked before embedding starts, so the caller gets a clear 429 rather
        than a provider error part-way through a long ingest. A budget of 0
        disables the ceiling.
        """
        budget = settings.INGEST_DAILY_EMBEDDING_BATCH_BUDGET

        if budget <= 0:
            return

        try:
            used = get_registry().usage(_utc_day(), _EMBEDDING_BATCH_METRIC)
        except Exception:
            # The budget is a guardrail, not the product: if it cannot be read,
            # allow the ingest rather than break the feature.
            logger.exception("could not read the ingest budget; allowing the ingest")

            return

        if used + batches > budget:
            raise IngestBudgetExceededError(
                f"Daily embedding budget reached ({used}/{budget} batches used). "
                "Try again after 00:00 UTC, or raise "
                "INGEST_DAILY_EMBEDDING_BATCH_BUDGET.",
                _seconds_until_utc_midnight(),
            )

    @staticmethod
    def _charge_ingest_budget(batches: int) -> None:
        """Record embedding batches against today's budget after they succeed."""
        if settings.INGEST_DAILY_EMBEDDING_BATCH_BUDGET <= 0:
            return

        try:
            get_registry().add_usage(_utc_day(), _EMBEDDING_BATCH_METRIC, batches)
        except Exception:
            logger.exception("could not record the ingest budget usage")

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

        document = PDFLoader(pdf_path).load_document()

        pages = document["pages"]

        # Whole-document splitting, so chunks are not cut at page boundaries.
        chunks = TextSplitter().split_document(pages)

        profile = (summarizer or DocumentSummarizer()).profile(
            pages,
            outline=document["outline"],
            metadata=document["metadata"],
        )

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

        ids = [f"{document_id}_{chunk['page_start']}_{chunk['chunk']}" for chunk in chunks]

        # The embedding fingerprint lets the retriever detect an index built
        # in a different vector space instead of returning meaningless hits.
        metadatas = [
            {
                # `page` is the citation anchor; the span is what the chunk
                # actually covers, which may be several pages.
                "page": chunk["page_start"],
                "page_start": chunk["page_start"],
                "page_end": chunk.get("page_end", chunk["page_start"]),
                "chunk": chunk["chunk"],
                "kind": chunk.get("kind", "content"),
                "document_id": document_id,
                "embedding_model": settings.EMBEDDING_MODEL,
                "schema_version": settings.EMBEDDING_SCHEMA_VERSION,
            }
            for chunk in chunks
        ]

        batches = math.ceil(len(texts) / max(1, settings.EMBEDDING_BATCH_SIZE))

        # Show the commitment before it is made: batching and model calls are
        # the only real cost in this path.
        logger.info(
            "ingesting document_id=%s chunks=%s embedding_batches=%s daily_budget=%s",
            document_id,
            len(texts),
            batches,
            settings.INGEST_DAILY_EMBEDDING_BATCH_BUDGET or "unlimited",
        )

        cls._enforce_ingest_budget(batches)

        embeddings = embedding_model.embed_documents(texts)

        cls._charge_ingest_budget(batches)

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

        _record_document(
            document_id=document_id,
            filename=Path(pdf_path).name,
            pages=len(pages),
            chunks=len(chunks),
            embedding_model=settings.EMBEDDING_MODEL,
            schema_version=settings.EMBEDDING_SCHEMA_VERSION,
        )

        return {
            "pages": pages,
            "chunks": chunks,
            "document_id": document_id,
            "duplicate": False,
            "index_replaced": True,
        }
