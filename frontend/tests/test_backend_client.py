"""
Tests for the frontend's backend client.

This is the user-facing error contract: what a person actually reads when an
upload is rejected, a provider is rate-limiting, or the backend is down. It is
tested because it is easy to break silently and hard to notice.
"""

import backend_client
from backend_client import (
    backend_error_message,
    retry_hint,
    unreachable_message,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")

        return self._payload


class RecordingPost:
    """A stand-in for requests.post that records how it was called."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))

        return self.response


# --------------------------------------------------------------------------
# Error messages
# --------------------------------------------------------------------------


def test_backend_error_message_surfaces_the_detail():
    response = FakeResponse(400, {"detail": "Only PDF files are allowed."})

    assert backend_error_message(response) == "Only PDF files are allowed."


def test_backend_error_message_joins_a_validation_error_list():
    response = FakeResponse(
        422,
        {"detail": [{"msg": "field required"}, {"msg": "too long"}]},
    )

    message = backend_error_message(response)

    assert "field required" in message
    assert "too long" in message


def test_backend_error_message_falls_back_to_the_status_code():
    response = FakeResponse(500, payload=None)

    assert "500" in backend_error_message(response)


def test_backend_error_message_explains_a_missing_api_key():
    response = FakeResponse(401, {"detail": "Invalid or missing API key."})

    message = backend_error_message(response)

    assert "Invalid or missing API key." in message
    assert "authorised" in message


# --------------------------------------------------------------------------
# Retry hints
# --------------------------------------------------------------------------


def test_retry_hint_reports_the_retry_after_seconds():
    response = FakeResponse(429, headers={"Retry-After": "42"})

    assert "42 seconds" in retry_hint(response)


def test_retry_hint_for_429_without_a_header_is_non_empty():
    assert retry_hint(FakeResponse(429))


def test_retry_hint_for_unavailable_and_busy():
    assert "unavailable" in retry_hint(FakeResponse(503))
    assert "upload" in retry_hint(FakeResponse(409))


def test_retry_hint_is_empty_for_a_status_the_user_cannot_act_on():
    assert retry_hint(FakeResponse(200)) == ""


def test_unreachable_message_names_the_backend():
    message = unreachable_message(ConnectionError("boom"))

    assert "Could not reach the backend" in message
    assert "ConnectionError" in message


# --------------------------------------------------------------------------
# Request wiring
# --------------------------------------------------------------------------


def test_ask_question_sends_the_api_key_header(monkeypatch):
    recorded = RecordingPost(FakeResponse(200, {"answer": "a", "sources": []}))

    monkeypatch.setattr(backend_client.requests, "post", recorded)

    payload, error = backend_client.ask_question("what is it?")

    assert error is None
    assert payload["answer"] == "a"

    url, kwargs = recorded.calls[0]

    assert url.endswith("/chat/")
    assert "X-API-Key" in kwargs["headers"]
    assert kwargs["json"] == {"question": "what is it?"}


def test_upload_pdf_sends_the_api_key_header(monkeypatch):
    recorded = RecordingPost(FakeResponse(200, {"duplicate": False, "pages": 3, "chunks": 9}))

    monkeypatch.setattr(backend_client.requests, "post", recorded)

    class UploadedFile:
        name = "report.pdf"

    ok, message = backend_client.upload_pdf(UploadedFile())

    assert ok
    assert "3 pages" in message

    url, kwargs = recorded.calls[0]

    assert url.endswith("/upload/")
    assert "X-API-Key" in kwargs["headers"]


def test_ask_question_turns_a_401_into_a_readable_error(monkeypatch):
    response = FakeResponse(401, {"detail": "Invalid or missing API key."})

    monkeypatch.setattr(backend_client.requests, "post", RecordingPost(response))

    payload, error = backend_client.ask_question("hi")

    assert payload is None
    assert "Invalid or missing API key." in error


def test_upload_pdf_reports_a_timeout_without_claiming_failure(monkeypatch):
    def timeout(url, **kwargs):
        raise backend_client.requests.Timeout("too slow")

    monkeypatch.setattr(backend_client.requests, "post", timeout)

    class UploadedFile:
        name = "big.pdf"

    ok, message = backend_client.upload_pdf(UploadedFile())

    assert ok is False
    # The request was delivered, so the message must not tell the user it failed.
    assert "may still be indexing" in message
