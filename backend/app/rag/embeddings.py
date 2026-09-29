"""Embedding access (Google) with batching and error mapping."""

import time
from functools import lru_cache
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.exceptions import (
    UpstreamRateLimitError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)


def _map_embedding_error(exc: Exception) -> Exception:
    """Translate a provider error into one that maps to a sensible HTTP code."""
    message = str(exc).lower()

    if "resource_exhausted" in message or "429" in message or "quota" in message:
        return UpstreamRateLimitError(
            "The embedding provider rate limit was reached. "
            "Try again shortly, or upload a smaller document.",
            service="embeddings",
        )

    if "deadline" in message or "timeout" in message:
        return UpstreamTimeoutError(
            "The embedding provider timed out.",
            service="embeddings",
        )

    return UpstreamUnavailableError(
        f"The embedding provider failed: {exc}",
        service="embeddings",
    )


class EmbeddingModel:
    """Wraps the Gemini embedding model."""

    def __init__(self) -> None:
        self.model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    def embed_documents(self, texts: list[str]) -> list[Any]:
        """
        Embed texts in small batches, pausing between them.

        Embedding a large document in one call trips the provider's
        per-minute request quota, so the work is spaced out instead.
        """
        vectors: list[Any] = []

        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            vectors.extend(self._embed_batch(batch))

            if start + batch_size < len(texts):
                time.sleep(settings.EMBEDDING_BATCH_DELAY_SECONDS)

        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        try:
            return self.model.embed_query(query)
        except Exception as exc:
            raise _map_embedding_error(exc) from exc

    def _embed_batch(self, batch: list[str]) -> list[Any]:
        try:
            return self.model.embed_documents(batch)
        except Exception as exc:
            raise _map_embedding_error(exc) from exc


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Process-wide embedding model (tests can call cache_clear())."""
    return EmbeddingModel()
