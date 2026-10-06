"""Reorder retrieved chunks by how likely they are to answer the question."""

import logging
import re

from openai import OpenAI

from app.core.config import settings
from app.rag.llm import complete, get_chat_client

logger = logging.getLogger(__name__)

RERANK_PROMPT = """You rank search results for a question-answering system.

Return the numbers of the passages that are relevant to the question, most
relevant first, as a comma-separated list. Use each number at most once.
If none are relevant, reply with NONE.

Question: {question}

Passages:
{passages}

Ranking:"""


class Reranker:
    """
    Reorders candidates using the model as a cross-encoder.

    A cross-encoder reads the question and the passage together, which
    separates passages that an embedding-distance ranking confuses (repeated
    boilerplate, for example). This implementation only reorders -- it never
    discards -- so a bad ranking cannot turn an answerable question into an
    abstention; weak matches are the distance threshold's job.
    """

    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or get_chat_client()

    def rerank(self, question: str, passages: list[str]) -> list[int] | None:
        """
        Return indices ordered by relevance, or None when unavailable.

        The returned list may be partial; callers append the remaining
        indices in their original order.
        """
        if not settings.RERANK_ENABLED or len(passages) < 2:
            return None

        prompt = RERANK_PROMPT.format(
            question=question,
            passages=_format_passages(passages),
        )

        try:
            raw = complete(prompt, client=self.client)
        except Exception as exc:
            logger.warning("rerank failed, keeping retrieval order: %s", exc)
            return None

        order = _parse_order(raw, len(passages))

        if not order:
            logger.warning("rerank returned no usable ranking; keeping retrieval order")
            return None

        return order


def _format_passages(passages: list[str]) -> str:
    limit = max(1, settings.RERANK_SNIPPET_CHARS)

    lines = [
        f"{index}. {' '.join(passage.split())[:limit]}"
        for index, passage in enumerate(passages, start=1)
    ]

    return "\n".join(lines)


def _parse_order(raw: str, count: int) -> list[int]:
    """Extract a 1-based, in-range, de-duplicated ordering from the reply."""
    order: list[int] = []

    for token in re.findall(r"\d+", raw or ""):
        index = int(token) - 1

        if 0 <= index < count and index not in order:
            order.append(index)

    return order
