"""PDF text extraction."""

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


class PDFLoader:
    """Extracts cleaned text, page by page, from a PDF."""

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)

    def load(self) -> list[dict[str, Any]]:
        """
        Return [{"page": int, "text": str}, ...] for every page.

        Raises a PDFProcessingError subclass when the document is unusable,
        so the caller can answer with a meaningful status code.
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

        except PDFProcessingError:
            raise

        except (PdfReadError, OSError, ValueError) as exc:
            raise PDFProcessingError(
                "The file could not be read as a PDF."
            ) from exc

        if not pages:
            raise EmptyPDFError("The PDF contains no pages.")

        if not any(page["text"] for page in pages):
            raise ScannedPDFError(
                "No selectable text was found in this PDF. It may be a "
                "scanned or image-only document; OCR is not yet supported."
            )

        return pages

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
