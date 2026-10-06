"""
A small persistent registry of ingested documents.

Chroma knows the vectors; this knows what they came from. It is what makes
listing, selecting and deleting documents possible, and it lets /ready report
per-document state instead of one global count. A SQLite file is enough: this
is a handful of rows, not a service.
"""

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    pages INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT,
    schema_version INTEGER,
    created_at TEXT NOT NULL
)
"""


class DocumentRegistry:
    """One record per ingested document."""

    _lock = threading.Lock()

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or settings.REGISTRY_PATH)

        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(self.path, check_same_thread=False)

        self._connection.row_factory = sqlite3.Row

        with self._lock:
            self._connection.execute(_SCHEMA)
            self._connection.commit()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record(
        self,
        document_id: str,
        filename: str,
        pages: int,
        chunks: int,
        embedding_model: str,
        schema_version: int,
    ) -> None:
        """Insert or update one document's record."""
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO documents (
                    document_id, filename, pages, chunks,
                    embedding_model, schema_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    filename = excluded.filename,
                    pages = excluded.pages,
                    chunks = excluded.chunks,
                    embedding_model = excluded.embedding_model,
                    schema_version = excluded.schema_version
                """,
                (
                    document_id,
                    filename,
                    pages,
                    chunks,
                    embedding_model,
                    schema_version,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            self._connection.commit()

    def replace_with(self, **record: Any) -> None:
        """
        Record the given document as the only indexed document.

        Ingestion swaps the whole collection, so the previously indexed
        document is no longer in the index and must not be reported as if it
        were. Multi-document ingestion will replace this with an append.
        """
        with self._lock:
            self._connection.execute("DELETE FROM documents")
            self._connection.commit()

        self.record(**record)

    def forget(self, document_id: str) -> None:
        """Drop one document's record."""
        with self._lock:
            self._connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
            self._connection.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, document_id: str) -> dict[str, Any] | None:
        """One document's record, or None."""
        cursor = self._connection.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id,),
        )

        row = cursor.fetchone()

        return dict(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        """Every document record, newest first."""
        cursor = self._connection.execute("SELECT * FROM documents ORDER BY created_at DESC")

        return [dict(row) for row in cursor.fetchall()]


@lru_cache(maxsize=1)
def get_registry() -> DocumentRegistry:
    """Process-wide registry (tests can call cache_clear())."""
    return DocumentRegistry()
