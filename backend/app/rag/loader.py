"""PDF text extraction, outline and metadata."""

from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import settings
from app.core.exceptions import (
    DocumentTooLargeError,
    EmptyPDFError,
    EncryptedPDFError,
    PDFProcessingError,
    ScannedPDFError,
)
from app.utils.helpers import clean_text

# An outline can be thousands of entries long; the profile only needs enough
# to describe the document's shape.
MAX_OUTLINE_ENTRIES = 80


class PDFLoader:
    """Extracts text, bookmarks and metadata from a PDF in one parse."""

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)

    def load_document(self) -> dict[str, Any]:
        """
        Return pages, bookmark outline and PDF metadata.

        One parse, because parsing a large document is the expensive part and
        the outline and metadata come from the same reader.
        """
        try:
            reader = PdfReader(self.pdf_path)

            if reader.is_encrypted:
                self._require_decryptable(reader)

            page_count = len(reader.pages)

            if page_count > settings.MAX_PAGES_PER_DOCUMENT:
                raise DocumentTooLargeError(
                    f"This PDF has {page_count} pages, which exceeds the "
                    f"{settings.MAX_PAGES_PER_DOCUMENT}-page limit."
                )

            pages = [
                {
                    "page": number,
                    "text": self._extract_text(page),
                }
                for number, page in enumerate(reader.pages, start=1)
            ]

            outline = self._outline(reader)

            metadata = self._metadata(reader)

        except PDFProcessingError:
            raise

        except (PdfReadError, OSError, ValueError) as exc:
            raise PDFProcessingError("The file could not be read as a PDF.") from exc

        if not pages:
            raise EmptyPDFError("The PDF contains no pages.")

        if not any(page["text"] for page in pages):
            raise ScannedPDFError(
                "No selectable text was found in this PDF. It may be a "
                "scanned or image-only document; OCR is not yet supported."
            )

        return {
            "pages": pages,
            "outline": outline,
            "metadata": metadata,
        }

    def load(self) -> list[dict[str, Any]]:
        """Return only the pages (kept for callers that need just text)."""
        return self.load_document()["pages"]

    @staticmethod
    def _outline(reader: PdfReader) -> list[dict[str, Any]]:
        """
        Flatten the bookmark tree into (depth, title) entries.

        Free and exact where it exists, which is why the profile prefers it to
        a model guessing at the document's structure from its first pages.
        """
        try:
            items = reader.outline
        except Exception:
            return []

        entries: list[dict[str, Any]] = []

        def walk(nodes: list[Any], depth: int) -> None:
            for node in nodes:
                if len(entries) >= MAX_OUTLINE_ENTRIES:
                    return

                if isinstance(node, list):
                    walk(node, depth + 1)
                    continue

                title = getattr(node, "title", None)

                if not title:
                    continue

                cleaned = " ".join(str(title).split())

                if cleaned:
                    entries.append({"depth": depth, "title": cleaned})

        try:
            walk(items, 0)
        except Exception:
            return entries

        return entries

    @staticmethod
    def _metadata(reader: PdfReader) -> dict[str, str]:
        """Title, author and subject as recorded by the producing tool."""
        raw = reader.metadata or {}

        metadata: dict[str, str] = {}

        for key, name in (
            ("/Title", "title"),
            ("/Author", "author"),
            ("/Subject", "subject"),
        ):
            value = raw.get(key)

            if value and str(value).strip():
                metadata[name] = " ".join(str(value).split())

        return metadata

    @staticmethod
    def _require_decryptable(reader: PdfReader) -> None:
        message = "The PDF is password protected and cannot be read."

        try:
            decrypted = reader.decrypt("")
        except Exception as exc:
            raise EncryptedPDFError(message) from exc

        if not decrypted:
            raise EncryptedPDFError(message)

    @staticmethod
    def _extract_text(page: Any) -> str:
        try:
            return clean_text(page.extract_text() or "")
        except Exception:
            # A single unreadable page should not fail the whole document.
            return ""
