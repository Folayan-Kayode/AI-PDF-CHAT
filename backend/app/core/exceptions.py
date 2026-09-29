"""
Error hierarchy for the application.

Everything that should produce a *specific* HTTP response derives from
AppError and carries its own status code. A single handler in main.py maps
these to JSON responses, which keeps the API contract consistent and stops
upstream failures from leaking out as bare "Internal Server Error" text.
"""


class AppError(Exception):
    """Base class for errors that map to a specific HTTP response."""

    status_code = 500

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


# --------------------------------------------------------------------------
# Ingestion / document errors (client's fault -> 4xx)
# --------------------------------------------------------------------------

class PDFProcessingError(AppError):
    """The uploaded PDF could not be turned into usable text."""

    status_code = 400


class EmptyPDFError(PDFProcessingError):
    """The PDF contains no pages."""


class EncryptedPDFError(PDFProcessingError):
    """The PDF is password protected and cannot be read."""


class ScannedPDFError(PDFProcessingError):
    """No selectable text could be extracted (scanned or image-only PDF)."""


class DocumentTooLargeError(PDFProcessingError):
    """The document exceeds a configured ingestion limit."""

    status_code = 413


# --------------------------------------------------------------------------
# Upstream dependency errors (our fault / not the client's -> 5xx)
# --------------------------------------------------------------------------

class UpstreamServiceError(AppError):
    """A dependency we call (LLM or vector store) failed."""

    status_code = 502

    def __init__(self, detail: str, service: str = "upstream"):
        super().__init__(detail)
        self.service = service


class UpstreamUnavailableError(UpstreamServiceError):
    """The dependency is unreachable or returned a server error."""

    status_code = 503


class UpstreamRateLimitError(UpstreamServiceError):
    """The dependency rejected the request due to a quota or rate limit."""

    status_code = 503


class UpstreamTimeoutError(UpstreamServiceError):
    """The dependency did not respond in time."""

    status_code = 504


class RetrievalError(UpstreamServiceError):
    """The vector store could not be queried."""

    status_code = 503
