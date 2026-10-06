"""Answer generation via DeepSeek (OpenAI-compatible API)."""

from functools import lru_cache

import openai
from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import (
    UpstreamRateLimitError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)


class DeepSeekGenerator:
    """
    Generation via DeepSeek's OpenAI-compatible chat completions API.

    Embeddings remain on Google; only the generator is switched here.
    """

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            timeout=settings.GENERATION_TIMEOUT_SECONDS,
        )

    def generate(self, prompt: str) -> str:
        """Return the model's answer for the given prompt."""
        try:
            response = self.client.chat.completions.create(
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


@lru_cache(maxsize=1)
def get_generator() -> DeepSeekGenerator:
    """Process-wide generator (tests can call cache_clear())."""
    return DeepSeekGenerator()
