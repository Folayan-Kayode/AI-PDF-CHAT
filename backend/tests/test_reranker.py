"""Tests for LLM-based reranking."""

from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.rag.reranker import Reranker


class FakeClient:
    """Minimal stand-in for the OpenAI client."""

    def __init__(self, content: str | None, error: Exception | None = None):
        self.content = content
        self.error = error
        self.prompts: list[str] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])

        if self.error is not None:
            raise self.error

        message = SimpleNamespace(content=self.content)

        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


PASSAGES = ["alpha", "beta", "gamma", "delta"]


@pytest.fixture(autouse=True)
def _enable_reranking(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)


def test_rerank_returns_a_one_based_order_as_zero_based_indices():
    client = FakeClient("3, 1, 4")

    assert Reranker(client=client).rerank("q", PASSAGES) == [2, 0, 3]


def test_rerank_prompt_numbers_every_passage():
    client = FakeClient("1")

    Reranker(client=client).rerank("the question", PASSAGES)

    prompt = client.prompts[0]

    assert "the question" in prompt
    assert "1. alpha" in prompt
    assert "4. delta" in prompt


def test_out_of_range_and_repeated_numbers_are_ignored():
    client = FakeClient("9, 2, 2, 0, 3")

    assert Reranker(client=client).rerank("q", PASSAGES) == [1, 2]


def test_none_reply_means_no_ranking():
    client = FakeClient("NONE")

    assert Reranker(client=client).rerank("q", PASSAGES) is None


def test_unparseable_reply_means_no_ranking():
    client = FakeClient("I cannot help with that.")

    assert Reranker(client=client).rerank("q", PASSAGES) is None


def test_rerank_failure_is_swallowed():
    client = FakeClient(None, error=RuntimeError("provider down"))

    # Reranking only reorders, so a failure must fall back silently.
    assert Reranker(client=client).rerank("q", PASSAGES) is None


def test_rerank_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)

    client = FakeClient("1, 2")

    assert Reranker(client=client).rerank("q", PASSAGES) is None
    assert client.prompts == []


def test_single_passage_needs_no_ranking():
    client = FakeClient("1")

    assert Reranker(client=client).rerank("q", ["only"]) is None
    assert client.prompts == []


def test_snippets_are_truncated(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_SNIPPET_CHARS", 5)

    client = FakeClient("1")

    Reranker(client=client).rerank("q", ["abcdefghij", "klmnopqrst"])

    assert "1. abcde\n" in client.prompts[0]
    assert "abcdefghij" not in client.prompts[0]
