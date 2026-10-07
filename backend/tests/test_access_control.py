"""
Access control: the paid endpoints require a key and are rate limited.

/upload runs hundreds of paid embedding calls and /chat spends a generation
call, so an unauthenticated request to either is a direct route to draining a
provider account. These tests fix the contract: open liveness/readiness, closed
paid endpoints, and a 429 the UI already knows how to explain.
"""

from fastapi.testclient import TestClient


def _stub_chat(monkeypatch, answer=None):
    from app.services import chat_service

    monkeypatch.setattr(
        chat_service.ChatService,
        "chat",
        staticmethod(
            lambda question: (
                answer
                or {"answer": "ok [p.1]", "sources": [{"page": 1, "chunk": 1}], "retrieval": {}}
            )
        ),
    )


# --------------------------------------------------------------------------
# The key
# --------------------------------------------------------------------------


def test_chat_without_a_key_is_401(no_key_client: TestClient):
    response = no_key_client.post("/chat/", json={"question": "what is it?"})

    assert response.status_code == 401
    assert "API key" in response.json()["detail"]
    assert response.headers["WWW-Authenticate"] == "X-API-Key"


def test_upload_without_a_key_is_401(no_key_client: TestClient, sample_pdf_bytes):
    response = no_key_client.post(
        "/upload/",
        files={"file": ("x.pdf", sample_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 401


def test_chat_with_the_wrong_key_is_401(no_key_client: TestClient):
    response = no_key_client.post(
        "/chat/",
        json={"question": "what is it?"},
        headers={"X-API-Key": "not-the-key"},
    )

    assert response.status_code == 401


def test_chat_with_the_key_succeeds(client: TestClient, monkeypatch):
    _stub_chat(monkeypatch)

    assert client.post("/chat/", json={"question": "what is it?"}).status_code == 200


def test_health_and_ready_need_no_key(no_key_client: TestClient):
    # Platform liveness/readiness probes must not require a credential.
    assert no_key_client.get("/health").status_code == 200
    assert no_key_client.get("/ready").status_code == 200
    assert no_key_client.get("/").status_code == 200


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_chat_rate_limit_returns_429_with_retry_after(client: TestClient, monkeypatch):
    from app.core.config import settings

    _stub_chat(monkeypatch)

    monkeypatch.setattr(settings, "CHAT_RATE_LIMIT_PER_HOUR", 2)

    assert client.post("/chat/", json={"question": "q"}).status_code == 200
    assert client.post("/chat/", json={"question": "q"}).status_code == 200

    blocked = client.post("/chat/", json={"question": "q"})

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert "Rate limit" in blocked.json()["detail"]


def test_upload_rate_limit_returns_429(client: TestClient, monkeypatch, fixtures_dir):
    from app.api import upload as upload_module
    from app.core.config import settings

    monkeypatch.setattr(settings, "UPLOAD_RATE_LIMIT_PER_HOUR", 1)

    monkeypatch.setattr(
        upload_module.PDFService,
        "process",
        staticmethod(
            lambda path: {
                "pages": [{"page": 1}],
                "chunks": [{"text": "x", "page": 1, "chunk": 1}],
                "document_id": "abc",
                "duplicate": False,
                "index_replaced": True,
            }
        ),
    )

    payload = (fixtures_dir / "sample.pdf").read_bytes()

    first = client.post(
        "/upload/",
        files={"file": ("a.pdf", payload, "application/pdf")},
    )

    second = client.post(
        "/upload/",
        files={"file": ("b.pdf", payload, "application/pdf")},
    )

    assert first.status_code == 200
    assert second.status_code == 429


def test_rate_limit_of_zero_disables_the_limit(client: TestClient, monkeypatch):
    from app.core.config import settings

    _stub_chat(monkeypatch)

    monkeypatch.setattr(settings, "CHAT_RATE_LIMIT_PER_HOUR", 0)

    for _ in range(5):
        assert client.post("/chat/", json={"question": "q"}).status_code == 200
