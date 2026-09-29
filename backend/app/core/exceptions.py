class PDFProcessingError(Exception):
    """Base class for PDF ingestion failures that should map to HTTP 400."""


class EmptyPDFError(PDFProcessingError):
    """The PDF contains no pages."""


class EncryptedPDFError(PDFProcessingError):
    """The PDF is password protected and cannot be read."""


class ScannedPDFError(PDFProcessingError):
    """No selectable text could be extracted (likely a scanned or image-only PDF)."""
