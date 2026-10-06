"""Tests for question rewriting."""

from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.core.exceptions import UpstreamUnavailableError
from app.rag.query_rewriter import QueryRewriter


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


@pytest.fixture(autouse=True)
def _enable_rewriting(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_REWRITE_ENABLED", True)


def test_rewrite_returns_the_model_query():
    client = FakeClient("Principles of Information Security title page")

    assert (
        QueryRewriter(client=client).rewrite("what is the title of this book?")
        == "Principles of Information Security title page"
    )


def test_rewrite_prompt_contains_the_question():
    client = FakeClient("a query")

    QueryRewriter(client=client).rewrite("who published it?")

    assert "who published it?" in client.prompts[0]


def test_rewrite_strips_quotes_and_whitespace():
    client = FakeClient('  "book title publisher"  \n')

    assert QueryRewriter(client=client).rewrite("q") == "book title publisher"


def test_rewrite_truncates_to_the_limit(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_REWRITE_MAX_CHARS", 10)

    client = FakeClient("a" * 12)

    assert QueryRewriter(client=client).rewrite("q") == "a" * 10


def test_rewrite_discards_an_answer_instead_of_a_query(monkeypatch):
    # A model that answers the question produces a long paragraph, which
    # embeds poorly as a query.
    monkeypatch.setattr(settings, "QUERY_REWRITE_MAX_CHARS", 10)

    client = FakeClient("word " * 50)

    assert QueryRewriter(client=client).rewrite("q") is None


@pytest.mark.parametrize("content", ["", "   ", "\n"])
def test_rewrite_returns_none_for_an_empty_reply(content):
    client = FakeClient(content)

    assert QueryRewriter(client=client).rewrite("q") is None


def test_rewrite_falls_back_when_the_provider_fails():
    client = FakeClient(
        None,
        error=UpstreamUnavailableError("provider down", service="generation"),
    )

    # Rewriting is an enhancement, so a failure must not raise.
    assert QueryRewriter(client=client).rewrite("q") is None


def test_rewrite_falls_back_on_an_unexpected_error():
    client = FakeClient(None, error=RuntimeError("boom"))

    assert QueryRewriter(client=client).rewrite("q") is None


def test_rewrite_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_REWRITE_ENABLED", False)

    client = FakeClient("a query")

    assert QueryRewriter(client=client).rewrite("q") is None
    assert client.prompts == []
