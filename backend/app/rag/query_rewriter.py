"""Rewrite a question into search terms that match the document's wording."""

import logging

from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import UpstreamServiceError
from app.rag.llm import complete, get_chat_client

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You turn a user's question into a search query for a \
document search engine.

Rules:
- Output ONLY the search query. No explanation, no quotes, no answer.
- Use the terms most likely to appear in the document, including likely
  names, headings and synonyms.
- Keep it under {max_chars} characters.
- If the question is already a good query, repeat it unchanged.

Question: {question}
Search query:"""


class QueryRewriter:
    """
    Produces an alternative query for retrieval.

    A question like "what is the title of this book?" does not embed close to
    the title page text, so retrieval misses it. A rewritten query using the
    wording a document would actually use recovers those cases. Rewriting is
    an enhancement only: any failure falls back to the original question.
    """

    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or get_chat_client()

    def rewrite(self, question: str) -> str | None:
        """Return an alternative query, or None when unavailable."""
        if not settings.QUERY_REWRITE_ENABLED:
            return None

        prompt = REWRITE_PROMPT.format(
            max_chars=settings.QUERY_REWRITE_MAX_CHARS,
            question=question,
        )

        try:
            raw = complete(prompt, client=self.client)
        except UpstreamServiceError as exc:
            logger.warning(
                "query rewrite failed, falling back to the original question: %s",
                exc,
            )
            return None
        except Exception as exc:
            logger.warning(
                "query rewrite failed unexpectedly, using the original question: %s",
                exc,
            )
            return None

        return _clean_query(raw)


def _clean_query(raw: str) -> str | None:
    """Normalise the model's output into a usable query."""
    text = " ".join((raw or "").split()).strip().strip("\"'").strip()

    if not text:
        return None

    # A model that answers instead of rewriting tends to run long; a long
    # "query" embeds poorly, so it is dropped rather than used.
    if len(text) > settings.QUERY_REWRITE_MAX_CHARS * 3:
        return None

    return text[: settings.QUERY_REWRITE_MAX_CHARS]
