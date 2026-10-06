"""Embedding access (Google) with batching, pacing and retry."""

import logging
import random
import re
import time
from functools import lru_cache
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.exceptions import (
    UpstreamRateLimitError,
    UpstreamRequestError,
    UpstreamServiceError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)

logger = logging.getLogger(__name__)

_RATE_LIMIT_MESSAGE = (
    "The embedding provider rate limit was reached. "
    "Try again shortly, or upload a smaller document."
)

_TIMEOUT_MESSAGE = "The embedding provider timed out."

_STATUS_PATTERN = re.compile(r"\b(4\d{2}|5\d{2})\s+[A-Z_]{3,}\b")

_STATUS_ATTRIBUTES = ("status_code", "code")

# google-genai reports a cooldown as "Please retry in 38.7s" or a
# RetryInfo detail, both of which reduce to "<number>s" after "retry".
_RETRY_AFTER_PATTERN = re.compile(
    r"retry(?:_delay| in)?[^0-9]{0,12}(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)

# Never sleep longer than this, whatever the provider asks for.
_MAX_RETRY_AFTER_SECONDS = 60.0


def _provider_status(exc: Exception) -> int | None:
    """Best-effort HTTP status code for a provider exception."""
    for attribute in _STATUS_ATTRIBUTES:
        value = getattr(exc, attribute, None)

        if isinstance(value, int) and 400 <= value <= 599:
            return value

    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)

    if isinstance(value, int) and 400 <= value <= 599:
        return value

    message = str(exc)

    match = _STATUS_PATTERN.search(message)

    if match:
        return int(match.group(1))

    lowered = message.lower()

    if "resource_exhausted" in lowered or "quota" in lowered:
        return 429

    if "deadline" in lowered or "timeout" in lowered or "timed out" in lowered:
        return 504

    return None


def retry_after_seconds(exc: Exception) -> float | None:
    """Cooldown the provider asked for, if it told us one."""
    match = _RETRY_AFTER_PATTERN.search(str(exc))

    if not match:
        return None

    return min(float(match.group(1)), _MAX_RETRY_AFTER_SECONDS)


def _map_embedding_error(exc: Exception) -> UpstreamServiceError:
    """
    Translate a provider error into one that maps to a sensible response.

    Non-retryable errors (4xx that are not a rate limit) are distinguished
    from transient ones so the retry loop does not spin on a request that
    will never succeed.
    """
    status = _provider_status(exc)
    cooldown = retry_after_seconds(exc)

    if status == 429:
        return UpstreamRateLimitError(
            _RATE_LIMIT_MESSAGE,
            service="embeddings",
            retry_after_seconds=cooldown,
        )

    if status in (408, 504):
        return UpstreamTimeoutError(
            _TIMEOUT_MESSAGE,
            service="embeddings",
            retry_after_seconds=cooldown,
        )

    if status is not None and status >= 500:
        return UpstreamUnavailableError(
            f"The embedding provider failed with HTTP {status}.",
            service="embeddings",
            retry_after_seconds=cooldown,
        )

    if status is not None:
        return UpstreamRequestError(
            f"The embedding provider rejected the request (HTTP {status}): {exc}",
            service="embeddings",
        )

    return UpstreamUnavailableError(
        f"The embedding provider failed: {exc}",
        service="embeddings",
        retry_after_seconds=cooldown,
    )


class EmbeddingModel:
    """Wraps the Gemini embedding model."""

    def __init__(self) -> None:
        self.model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    @property
    def request_pacing_seconds(self) -> float:
        """Pause between embedding requests, at or above the quota budget."""
        return max(
            settings.EMBEDDING_MIN_DELAY_SECONDS,
            settings.EMBEDDING_BATCH_DELAY_SECONDS,
        )

    def embed_documents(self, texts: list[str]) -> list[Any]:
        """
        Embed texts in small, paced batches, retrying transient failures.

        A single request for a large document trips the provider's
        per-minute quota, and one transient error mid-document should not
        discard the batches that already succeeded.
        """
        vectors: list[Any] = []

        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            vectors.extend(self._embed_batch_with_retry(batch))

            if start + batch_size < len(texts):
                time.sleep(self.request_pacing_seconds)

        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        try:
            return self.model.embed_query(query)
        except Exception as exc:
            raise _map_embedding_error(exc) from exc

    def _embed_batch_with_retry(self, batch: list[str]) -> list[Any]:
        attempts = max(1, settings.EMBEDDING_MAX_RETRIES)
        base_delay = max(0.0, settings.EMBEDDING_RETRY_BASE_SECONDS)

        for attempt in range(1, attempts + 1):
            try:
                return self.model.embed_documents(batch)

            except Exception as exc:
                error = _map_embedding_error(exc)

                if not error.retryable or attempt >= attempts:
                    raise error from exc

                cooldown = error.retry_after_seconds or 0.0

                delay = max(
                    base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay),
                    cooldown,
                )

                logger.warning(
                    "embedding batch failed (attempt %s/%s): %s; retrying in %.1fs",
                    attempt,
                    attempts,
                    exc,
                    delay,
                )

                time.sleep(delay)

        # Unreachable: the loop either returns or raises.
        raise UpstreamUnavailableError(
            "The embedding provider failed.",
            service="embeddings",
        )


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Process-wide embedding model (tests can call cache_clear())."""
    return EmbeddingModel()
