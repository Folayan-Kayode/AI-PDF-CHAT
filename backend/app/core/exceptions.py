"""
Error hierarchy for the application.

Everything that should produce a *specific* HTTP response derives from
AppError and carries its own status code and optional response headers. A
single handler in main.py maps these to JSON responses, which keeps the API
contract consistent and stops upstream failures from leaking out as bare
"Internal Server Error" text.

Errors also declare whether they are worth retrying, so the embedding and
vector-store retry loops do not have to guess.
"""

from typing import Any


class AppError(Exception):
    """Base class for errors that map to a specific HTTP response."""

    status_code = 500
    retryable = False

    def __init__(
        self,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.headers = dict(headers or {})


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


class IngestionInProgressError(AppError):
    """Another document is already being ingested."""

    status_code = 409


class IngestBudgetExceededError(AppError):
    """
    The daily embedding budget is exhausted.

    Ingestion is the expensive path, so it has a spend ceiling independent of
    the request rate limit. Mapped to 429 with a Retry-After pointing at the
    next UTC day rather than a provider error surfacing mid-ingest.
    """

    status_code = 429

    def __init__(self, detail: str, retry_after_seconds: float) -> None:
        super().__init__(
            detail,
            headers={"Retry-After": str(max(1, int(round(retry_after_seconds))))},
        )


# --------------------------------------------------------------------------
# Upstream dependency errors (not the client's fault -> 5xx / 429)
# --------------------------------------------------------------------------


class UpstreamServiceError(AppError):
    """A dependency we call (LLM or vector store) failed."""

    status_code = 502

    retryable = True

    def __init__(
        self,
        detail: str,
        service: str = "upstream",
        retry_after_seconds: float | None = None,
    ) -> None:
        headers: dict[str, str] = {}

        self.retry_after_seconds = retry_after_seconds

        if retry_after_seconds is not None:
            headers["Retry-After"] = str(max(1, int(round(retry_after_seconds))))

        super().__init__(detail, headers=headers)

        self.service = service


class UpstreamUnavailableError(UpstreamServiceError):
    """The dependency is unreachable or returned a server error."""

    status_code = 503


class UpstreamRateLimitError(UpstreamServiceError):
    """The dependency rejected the request due to a quota or rate limit."""

    status_code = 429


class UpstreamTimeoutError(UpstreamServiceError):
    """The dependency did not respond in time."""

    status_code = 504


class UpstreamRequestError(UpstreamServiceError):
    """The dependency rejected the request itself; retrying will not help."""

    status_code = 502

    retryable = False


class RetrievalError(UpstreamServiceError):
    """The vector store could not be read or written."""

    status_code = 503


def error_headers(exc: AppError) -> dict[str, Any]:
    """Response headers implied by an error (currently just Retry-After)."""
    return dict(exc.headers)
