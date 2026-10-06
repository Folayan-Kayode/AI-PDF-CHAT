"""Answer generation via DeepSeek (OpenAI-compatible API)."""

from functools import lru_cache

from openai import OpenAI

from app.rag.llm import complete, get_chat_client


class DeepSeekGenerator:
    """
    Generation via DeepSeek's OpenAI-compatible chat completions API.

    Embeddings remain on Google; only the generator is switched here.
    """

    def __init__(self) -> None:
        self.client: OpenAI = get_chat_client()

    def generate(self, prompt: str) -> str:
        """Return the model's answer for the given prompt."""
        return complete(prompt, client=self.client)


@lru_cache(maxsize=1)
def get_generator() -> DeepSeekGenerator:
    """Process-wide generator (tests can call cache_clear())."""
    return DeepSeekGenerator()
