"""Tests for embedding error mapping, retry, batching and pacing."""

import pytest

from app.core.config import settings
from app.core.exceptions import (
    UpstreamRateLimitError,
    UpstreamRequestError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from app.rag.embeddings import (
    EmbeddingModel,
    _map_embedding_error,
    retry_after_seconds,
)


def _model() -> EmbeddingModel:
    """An EmbeddingModel without constructing a real API client."""
    return EmbeddingModel.__new__(EmbeddingModel)


class FlakyModel:
    """Fails a given number of times, then succeeds."""

    def __init__(self, failures: int = 0, message: str = "429 RESOURCE_EXHAUSTED"):
        self.failures = failures
        self.message = message
        self.attempts = 0
        self.batch_sizes: list[int] = []

    def embed_documents(self, batch):
        self.attempts += 1
        self.batch_sizes.append(len(batch))

        if self.attempts <= self.failures:
            raise RuntimeError(self.message)

        return [[0.1, 0.2, 0.3] for _ in batch]


# --------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------


def test_rate_limit_is_mapped_and_retryable():
    mapped = _map_embedding_error(Exception("429 RESOURCE_EXHAUSTED. quota exceeded"))

    assert isinstance(mapped, UpstreamRateLimitError)
    assert mapped.retryable is True


def test_rate_limit_keeps_the_provider_cooldown():
    mapped = _map_embedding_error(Exception("429 RESOURCE_EXHAUSTED ... Please retry in 38.7s."))

    assert mapped.retry_after_seconds == pytest.approx(38.7)


def test_client_error_code_attribute_is_used():
    class ProviderError(Exception):
        def __init__(self, code):
            super().__init__("provider said no")
            self.code = code

    assert isinstance(
        _map_embedding_error(ProviderError(429)),
        UpstreamRateLimitError,
    )
    assert isinstance(
        _map_embedding_error(ProviderError(503)),
        UpstreamUnavailableError,
    )


def test_timeout_is_mapped():
    assert isinstance(
        _map_embedding_error(Exception("504 DEADLINE_EXCEEDED")),
        UpstreamTimeoutError,
    )
    assert isinstance(
        _map_embedding_error(Exception("the request timed out")),
        UpstreamTimeoutError,
    )


def test_server_error_is_mapped_to_unavailable():
    mapped = _map_embedding_error(Exception("503 UNAVAILABLE, service is down"))

    assert isinstance(mapped, UpstreamUnavailableError)
    assert mapped.retryable is True


def test_bad_request_is_not_retryable():
    mapped = _map_embedding_error(Exception("400 INVALID_ARGUMENT bad input"))

    assert isinstance(mapped, UpstreamRequestError)
    assert mapped.retryable is False


def test_unknown_error_is_treated_as_retryable():
    mapped = _map_embedding_error(Exception("connection reset by peer"))

    assert isinstance(mapped, UpstreamUnavailableError)
    assert mapped.retryable is True


def test_retry_after_parsing():
    assert retry_after_seconds(Exception("Please retry in 4.5s.")) == pytest.approx(4.5)
    assert retry_after_seconds(Exception("no cooldown here")) is None


def test_retry_after_is_capped():
    assert retry_after_seconds(Exception("retry in 99999s")) == pytest.approx(60.0)


# --------------------------------------------------------------------------
# Retry, batching and pacing
# --------------------------------------------------------------------------


def test_batch_is_retried_until_it_succeeds(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_MAX_RETRIES", 4)
    monkeypatch.setattr(settings, "EMBEDDING_RETRY_BASE_SECONDS", 0.0)

    model = _model()
    flaky = FlakyModel(failures=2)
    model.model = flaky

    vectors = model.embed_documents(["a", "b"])

    assert len(vectors) == 2
    assert flaky.attempts == 3


def test_batch_raises_after_the_final_attempt(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_MAX_RETRIES", 3)
    monkeypatch.setattr(settings, "EMBEDDING_RETRY_BASE_SECONDS", 0.0)

    model = _model()
    flaky = FlakyModel(failures=99)
    model.model = flaky

    with pytest.raises(UpstreamRateLimitError):
        model.embed_documents(["a"])

    assert flaky.attempts == 3


def test_non_retryable_error_fails_immediately(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_MAX_RETRIES", 4)
    monkeypatch.setattr(settings, "EMBEDDING_RETRY_BASE_SECONDS", 0.0)

    model = _model()
    flaky = FlakyModel(failures=99, message="400 INVALID_ARGUMENT")
    model.model = flaky

    with pytest.raises(UpstreamRequestError):
        model.embed_documents(["a"])

    assert flaky.attempts == 1


def test_documents_are_embedded_in_batches(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(settings, "EMBEDDING_REQUESTS_PER_MINUTE", 100000)

    model = _model()
    flaky = FlakyModel()
    model.model = flaky

    vectors = model.embed_documents(["a", "b", "c", "d", "e"])

    assert flaky.batch_sizes == [2, 2, 1]
    assert len(vectors) == 5


def test_pacing_follows_the_quota_budget(monkeypatch):
    model = _model()

    monkeypatch.setattr(settings, "EMBEDDING_REQUESTS_PER_MINUTE", 60)
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_DELAY_SECONDS", 0.0)

    assert model.request_pacing_seconds == pytest.approx(1.0)


def test_configured_delay_wins_when_it_is_slower(monkeypatch):
    model = _model()

    monkeypatch.setattr(settings, "EMBEDDING_REQUESTS_PER_MINUTE", 100000)
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_DELAY_SECONDS", 5.0)

    assert model.request_pacing_seconds == pytest.approx(5.0)


def test_embed_query_wraps_provider_errors():
    model = _model()

    class BrokenModel:
        def embed_query(self, query):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    model.model = BrokenModel()

    with pytest.raises(UpstreamRateLimitError):
        model.embed_query("hello")


def test_embed_query_is_retried(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_MAX_RETRIES", 4)

    class FlakyQueryModel:
        def __init__(self):
            self.attempts = 0

        def embed_query(self, query):
            self.attempts += 1

            if self.attempts <= 2:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")

            return [0.1, 0.2]

    model = _model()
    flaky = FlakyQueryModel()
    model.model = flaky

    assert model.embed_query("q") == [0.1, 0.2]
    assert flaky.attempts == 3


def test_embed_query_gives_up_after_the_final_attempt(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_MAX_RETRIES", 2)

    class BrokenQueryModel:
        def __init__(self):
            self.attempts = 0

        def embed_query(self, query):
            self.attempts += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    model = _model()
    broken = BrokenQueryModel()
    model.model = broken

    with pytest.raises(UpstreamRateLimitError):
        model.embed_query("q")

    assert broken.attempts == 2
