"""Chat orchestration."""

import logging
from typing import Any

from app.core.exceptions import DocumentNotFoundError
from app.database.chroma import get_database
from app.database.registry import get_registry
from app.rag.pipeline import RAGPipeline, get_pipeline
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class ChatService:
    """Answers questions using the shared RAG pipeline."""

    @staticmethod
    def chat(
        question: str,
        document_id: str | None = None,
        pipeline: RAGPipeline | None = None,
    ) -> dict[str, Any]:
        """
        Return {"answer": str, "sources": [...]} for the question.

        Scoped to one document. When ``document_id`` is omitted the newest
        registered document is used, so single-document clients keep working;
        the UI always sends the selected id. An explicitly supplied pipeline is
        used as-is, which is how the evaluation harness stays single-document.
        """
        if pipeline is not None:
            return pipeline.ask(question)

        if document_id is None:
            document_id = _newest_document_id()

        if document_id is None:
            # Nothing indexed: the pipeline's empty-index path abstains rather
            # than inventing an answer.
            return get_pipeline().ask(question)

        if not _document_exists(document_id):
            raise DocumentNotFoundError(f"Document {document_id!r} was not found.")

        return RAGPipeline(retriever=Retriever(document_id=document_id)).ask(question)


def _newest_document_id() -> str | None:
    """The most recently ingested document, or None when the registry is empty."""
    try:
        records = get_registry().list()
    except Exception:
        logger.exception("could not read the document registry")

        return None

    return records[0]["document_id"] if records else None


def _document_exists(document_id: str) -> bool:
    """
    Whether a document is known to the registry or present in the index.

    Both are checked so a lost registry does not turn a queryable document
    into a 404, while a stale id is still rejected.
    """
    try:
        if get_registry().get(document_id):
            return True
    except Exception:
        logger.exception("could not read the document registry")

    try:
        return get_database().has_document(document_id)
    except Exception:
        logger.exception("could not read the document index")

        return False
