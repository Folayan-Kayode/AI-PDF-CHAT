"""
Runtime topology guards.

Embedded ChromaDB keeps its state in local files, and ingestion is serialised
by a process-wide lock. Two uvicorn workers would each hold a client over the
same directory and could interleave writes or swap collections, corrupting the
index silently. There is no performance reason to run more than one worker for
this application, so the multi-worker case is treated as a misconfiguration and
refused rather than warned about.

Managed platforms (Render, Heroku) set ``WEB_CONCURRENCY`` for you, which is
exactly the accidental case this catches.
"""

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class UnsupportedTopologyError(RuntimeError):
    """The process was asked to start in a configuration the app cannot run."""


def assert_single_worker() -> None:
    """
    Refuse to start with more than one worker unless explicitly acknowledged.

    ``MULTI_WORKER_ACK`` exists for the day Chroma is moved to server mode (or
    the operator otherwise accepts the risk); it is not a convenience switch.
    """
    if settings.WEB_CONCURRENCY > 1 and not settings.MULTI_WORKER_ACK:
        raise UnsupportedTopologyError(
            f"WEB_CONCURRENCY={settings.WEB_CONCURRENCY}, but this app runs embedded "
            "ChromaDB with a process-wide ingestion lock and must use exactly one "
            "worker. Set WEB_CONCURRENCY=1, or set MULTI_WORKER_ACK=true only after "
            "moving Chroma to server mode."
        )


def worker_count() -> int:
    """The configured worker count, reported by /health for observability."""
    return max(1, settings.WEB_CONCURRENCY)
