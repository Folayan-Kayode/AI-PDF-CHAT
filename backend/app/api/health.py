"""Health and readiness endpoints."""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.database.chroma import get_database
from app.rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

# Cached result of the optional provider probe, so a caller polling with
# ?deep=true cannot run up a bill.
_deep_check: dict[str, Any] = {"checked_at": 0.0, "result": None}


@router.get("/health")
def health() -> dict[str, Any]:
    """
    Liveness: the process is up and configured.

    Deliberately cheap and independent of the vector store, so a platform
    probe firing every few seconds cannot contend with an ingestion.
    """
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "generation_model": settings.MODEL_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


@router.get("/ready")
def ready(deep: bool = False) -> dict[str, Any]:
    """
    Readiness: the index exists and is usable.

    An empty index reports "degraded" with HTTP 200 - a fresh install is not
    broken, it simply has no document yet - so a platform health check does
    not flap. HTTP 503 is reserved for the vector store being unreachable.
    """
    try:
        database = get_database()

        indexed_chunks = database.count()
        indexed_models = database.indexed_embedding_models()

    except Exception as exc:
        logger.exception("readiness check failed: %s", exc)

        raise HTTPException(
            status_code=503,
            detail="The document index is unavailable.",
        ) from exc

    model_matches = not indexed_models or settings.EMBEDDING_MODEL in indexed_models

    body: dict[str, Any] = {
        "status": "ready" if indexed_chunks and model_matches else "degraded",
        "indexed_chunks": indexed_chunks,
        "indexed_embedding_models": sorted(indexed_models),
        "embedding_model_match": model_matches,
        "generation_model": settings.MODEL_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "version": settings.API_VERSION,
    }

    if deep:
        body["embedding_provider"] = _probe_embedding_provider()

    return body


def _probe_embedding_provider() -> str:
    """
    Probe the embedding provider, cached for HEALTH_DEEP_CACHE_SECONDS.

    Only embeddings are probed: they are on both the ingest and the query
    path, and unlike generation a single embed call costs effectively
    nothing.
    """
    now = time.monotonic()

    cached = _deep_check.get("result")
    checked_at = _deep_check.get("checked_at", 0.0)

    if cached is not None and now - checked_at < settings.HEALTH_DEEP_CACHE_SECONDS:
        return str(cached)

    try:
        get_embedding_model().embed_query("health check")
        result = "ok"

    except Exception as exc:
        logger.warning("embedding provider probe failed: %s", exc)
        result = "error"

    _deep_check["checked_at"] = now
    _deep_check["result"] = result

    return result
