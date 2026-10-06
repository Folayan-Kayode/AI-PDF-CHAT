"""Shared DeepSeek chat client and one-shot completion helper."""

from functools import lru_cache

import openai
from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import (
    UpstreamRateLimitError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)


@lru_cache(maxsize=1)
def get_chat_client() -> OpenAI:
    """Process-wide chat client (tests can call cache_clear())."""
    return OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.GENERATION_TIMEOUT_SECONDS,
    )


def complete(prompt: str, client: OpenAI | None = None) -> str:
    """
    Run a single-prompt completion and return the text.

    Provider failures are translated into UpstreamServiceError subclasses so
    every caller maps to the same HTTP contract.
    """
    client = client or get_chat_client()

    try:
        response = client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
    except openai.RateLimitError as exc:
        raise UpstreamRateLimitError(
            "The generation provider rate limit was reached. Try again shortly.",
            service="generation",
        ) from exc
    except openai.APITimeoutError as exc:
        raise UpstreamTimeoutError(
            "The generation provider timed out.",
            service="generation",
        ) from exc
    except openai.APIConnectionError as exc:
        raise UpstreamUnavailableError(
            "Could not reach the generation provider.",
            service="generation",
        ) from exc
    except openai.APIStatusError as exc:
        raise UpstreamUnavailableError(
            f"The generation provider returned HTTP {exc.status_code}.",
            service="generation",
        ) from exc

    if not response.choices:
        return ""

    return response.choices[0].message.content or ""
