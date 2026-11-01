"""Document listing and deletion."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.database.registry import get_registry
from app.services.pdf_service import PDFService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("/")
def list_documents() -> dict[str, Any]:
    """Every indexed document, newest first."""
    try:
        documents = get_registry().list()
    except Exception as exc:
        logger.exception("could not read the document registry")

        raise HTTPException(
            status_code=503,
            detail="The document registry is unavailable.",
        ) from exc

    return {"documents": documents}


@router.delete("/{document_id}")
def delete_document(document_id: str) -> dict[str, Any]:
    """
    Delete one document: vectors, registry row and uploaded file.

    Returns 404 for an id that is neither registered nor indexed, so deleting
    twice is a clear error rather than a silent success.
    """
    record = PDFService.delete_document(document_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    logger.info(
        "deleted document_id=%s filename=%s",
        document_id,
        record.get("filename"),
    )

    return {
        "deleted": True,
        "document_id": document_id,
        "filename": record.get("filename"),
    }
