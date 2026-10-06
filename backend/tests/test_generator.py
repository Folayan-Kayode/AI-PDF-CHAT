"""Tests for the generation client's error mapping and response handling."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from app.core.config import settings
from app.core.exceptions import (
    UpstreamRateLimitError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from app.rag.generator import DeepSeekGenerator


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")

    return httpx.Response(
        status_code,
        request=request,
        json={"error": {"message": "provider said no"}},
    )


class FakeCompletions:
    def __init__(self, error: Exception | None = None, content: str | None = "ok"):
        self.error = error
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        if self.content is None:
            return SimpleNamespace(choices=[])

        message = SimpleNamespace(content=self.content)

        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _generator(error: Exception | None = None, content: str | None = "ok"):
    completions = FakeCompletions(error, content)

    generator = DeepSeekGenerator.__new__(DeepSeekGenerator)
    generator.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    return generator, completions


# --------------------------------------------------------------------------
# Successful responses
# --------------------------------------------------------------------------


def test_generate_returns_the_message_content():
    generator, _ = _generator(content="the answer")

    assert generator.generate("a prompt") == "the answer"


def test_generate_sends_the_configured_model_and_prompt():
    generator, completions = _generator()

    generator.generate("a prompt")

    call = completions.calls[0]

    assert call["model"] == settings.MODEL_NAME
    assert call["messages"] == [{"role": "user", "content": "a prompt"}]


def test_generate_tolerates_a_null_content():
    generator, _ = _generator(content=None)

    # No choices at all is an empty answer, not a crash.
    assert generator.generate("a prompt") == ""


# --------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------


def test_rate_limit_maps_to_429():
    error = openai.RateLimitError(
        "rate limited",
        response=_response(429),
        body=None,
    )

    generator, _ = _generator(error=error)

    with pytest.raises(UpstreamRateLimitError):
        generator.generate("a prompt")


def test_timeout_maps_to_504():
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")

    generator, _ = _generator(error=openai.APITimeoutError(request=request))

    with pytest.raises(UpstreamTimeoutError):
        generator.generate("a prompt")


def test_connection_error_maps_to_503():
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")

    generator, _ = _generator(error=openai.APIConnectionError(request=request))

    with pytest.raises(UpstreamUnavailableError):
        generator.generate("a prompt")


def test_server_error_maps_to_503_and_names_the_status():
    error = openai.APIStatusError(
        "server error",
        response=_response(500),
        body=None,
    )

    generator, _ = _generator(error=error)

    with pytest.raises(UpstreamUnavailableError) as excinfo:
        generator.generate("a prompt")

    assert "500" in str(excinfo.value)
