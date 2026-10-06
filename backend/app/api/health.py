"""Health and readiness endpoints."""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.database.chroma import get_database
from app.rag.embeddings import get_embedding_model
from app.rag.retriever import Retriever

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
    Readiness: the index exists, is usable, and is described per document.

    Reports "degraded" (still HTTP 200) when the index is empty or does not
    match the configured embedding settings — a fresh install is not broken,
    it just has nothing usable yet — so a platform health check does not flap.
    HTTP 503 is reserved for the vector store being unreachable.
    """
    try:
        database = get_database()

        indexed_chunks = database.count()
        status = Retriever(database=database).index_status()
        documents = _document_summaries(database)

    except Exception as exc:
        logger.exception("readiness check failed: %s", exc)

        raise HTTPException(
            status_code=503,
            detail="The document index is unavailable.",
        ) from exc

    usable = bool(
        status["embedding_model_match"]
        and status["schema_version_match"]
        and status["vector_space_match"]
        and indexed_chunks
    )

    body: dict[str, Any] = {
        "status": "ready" if usable else "degraded",
        "indexed_chunks": indexed_chunks,
        "documents": documents,
        **status,
        "generation_model": settings.MODEL_NAME,
        "embedding_model": settings.EMBEDDING_MODEL,
        "version": settings.API_VERSION,
    }

    if deep:
        body["embedding_provider"] = _probe_embedding_provider()

    return body


def _document_summaries(database: Any) -> list[dict[str, Any]]:
    """
    One entry per indexed document.

    Read from the index rather than a side table so /ready cannot disagree
    with what is actually stored; the registry is authoritative for the
    filename.
    """
    try:
        metadatas = (
            database.collection.get(
                limit=5000,
                include=["metadatas"],
            ).get("metadatas")
            or []
        )
    except Exception:
        return []

    summaries: dict[str, dict[str, Any]] = {}

    for metadata in metadatas:
        if not metadata:
            continue

        document_id = metadata.get("document_id")

        if not document_id:
            continue

        entry = summaries.setdefault(
            document_id,
            {
                "document_id": document_id,
                "chunks": 0,
                "embedding_model": metadata.get("embedding_model"),
                "schema_version": metadata.get("schema_version"),
            },
        )

        entry["chunks"] += 1

    try:
        from app.database.registry import get_registry

        for document_id, entry in summaries.items():
            record = get_registry().get(document_id)

            if record:
                entry["filename"] = record.get("filename")
                entry["pages"] = record.get("pages")
                entry["created_at"] = record.get("created_at")
    except Exception:
        # The registry is a convenience; the index itself is authoritative.
        pass

    return [summaries[key] for key in sorted(summaries)]


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
