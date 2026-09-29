"""Health endpoint."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.database.chroma import get_database

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


# Plain def: reading the index is a blocking call.
@router.get("/health")
def health() -> dict[str, Any]:
    """
    Report readiness.

    Verifies the vector store is actually reachable rather than always
    claiming to be healthy. It deliberately does not call the paid
    generation or embedding APIs.
    """
    try:
        indexed_chunks = get_database().count()

    except Exception as exc:
        logger.error("health check failed: %s", exc)

        raise HTTPException(
            status_code=503,
            detail="The document index is unavailable.",
        ) from exc

    return {
        "status": "healthy",
        "indexed_chunks": indexed_chunks,
        "generation_model": settings.MODEL_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "version": settings.API_VERSION,
    }
