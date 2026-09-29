from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import (
    EmptyPDFError,
    EncryptedPDFError,
    PDFProcessingError,
    ScannedPDFError,
)
from app.utils.helpers import clean_text

class PDFLoader:

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    def load(self):

        try:
            reader = PdfReader(self.pdf_path)

            if reader.is_encrypted:
                try:
                    decrypted = reader.decrypt("")
                except Exception as exc:
                    raise EncryptedPDFError(
                        "The PDF is password protected and cannot be read."
                    ) from exc

                if not decrypted:
                    raise EncryptedPDFError(
                        "The PDF is password protected and cannot be read."
                    )

            pages = []

            for page_number, page in enumerate(reader.pages, start=1):

                try:
                    raw_text = page.extract_text() or ""
                except Exception:
                    raw_text = ""

                pages.append(
                    {
                        "page": page_number,
                        "text": clean_text(raw_text)
                    }
                )

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
