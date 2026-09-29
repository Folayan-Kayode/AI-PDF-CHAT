"""End-to-end tests for the HTTP API. The LLM is always mocked."""

from fastapi.testclient import TestClient

from app.core.exceptions import DocumentTooLargeError, UpstreamUnavailableError

# --------------------------------------------------------------------------
# Metadata and health
# --------------------------------------------------------------------------

def test_root_reports_the_generation_model(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["model"] == "deepseek-chat"
    assert body["project"]


def test_health_reports_the_index(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert "indexed_chunks" in body
    assert body["generation_model"] == "deepseek-chat"


def test_health_returns_503_when_the_index_is_unavailable(
    client: TestClient,
    monkeypatch,
):
    from app.api import health as health_module

    class BrokenDatabase:
        def count(self):
            raise RuntimeError("index is down")

    monkeypatch.setattr(
        health_module,
        "get_database",
        lambda: BrokenDatabase(),
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"]


def test_request_id_is_echoed_back(client: TestClient):
    response = client.get("/health", headers={"X-Request-ID": "abc123"})

    assert response.headers["X-Request-ID"] == "abc123"


def test_request_id_is_generated_when_absent(client: TestClient):
    response = client.get("/health")

    assert response.headers["X-Request-ID"]


# --------------------------------------------------------------------------
# Chat validation
# --------------------------------------------------------------------------

def test_chat_rejects_a_blank_question(client: TestClient):
    response = client.post("/chat/", json={"question": "   "})

    assert response.status_code == 422


def test_chat_rejects_a_long_question(client: TestClient):
    response = client.post("/chat/", json={"question": "x" * 2001})

    assert response.status_code == 422


def test_chat_returns_the_answer_and_sources(client: TestClient, monkeypatch):
    from app.services import chat_service

    expected = {
        "answer": "The answer is 42.",
        "sources": [{"page": 1, "chunk": 2}],
    }

    monkeypatch.setattr(
        chat_service.ChatService,
        "chat",
        staticmethod(lambda question: expected),
    )

    response = client.post("/chat/", json={"question": "what is it?"})

    assert response.status_code == 200
    assert response.json() == expected


def test_chat_maps_an_upstream_failure_to_json_503(
    client: TestClient,
    monkeypatch,
):
    from app.services import chat_service

    def explode(question):
        raise UpstreamUnavailableError("provider is down", service="generation")

    monkeypatch.setattr(
        chat_service.ChatService,
        "chat",
        staticmethod(explode),
    )

    response = client.post("/chat/", json={"question": "what is it?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "provider is down"


def test_chat_maps_an_unexpected_failure_to_json_500(
    client: TestClient,
    monkeypatch,
):
    from app.services import chat_service

    def explode(question):
        raise ValueError("something unexpected")

    monkeypatch.setattr(
        chat_service.ChatService,
        "chat",
        staticmethod(explode),
    )

    response = client.post("/chat/", json={"question": "what is it?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."


# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------

def test_upload_rejects_a_non_pdf_extension(client: TestClient):
    response = client.post(
        "/upload/",
        files={"file": ("notes.txt", b"hello", "application/pdf")},
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_a_file_without_the_pdf_header(client: TestClient):
    response = client.post(
        "/upload/",
        files={"file": ("fake.pdf", b"not really a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


def test_upload_rejects_an_empty_file(client: TestClient):
    response = client.post(
        "/upload/",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400


def test_upload_enforces_the_size_limit(client: TestClient, monkeypatch):
    from app.api import upload as upload_module

    monkeypatch.setattr(
        upload_module.settings,
        "MAX_UPLOAD_SIZE_BYTES",
        1024,
    )

    response = client.post(
        "/upload/",
        files={
            "file": (
                "big.pdf",
                b"%PDF-" + b"0" * 4096,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413


def test_upload_sanitises_the_filename(client: TestClient, monkeypatch, fixtures_dir):
    from app.api import upload as upload_module

    monkeypatch.setattr(
        upload_module.PDFService,
        "process",
        staticmethod(
            lambda path: {
                "pages": [{"page": 1, "text": "x"}],
                "chunks": [{"text": "x", "page": 1, "chunk": 1}],
                "document_id": "abc",
                "duplicate": False,
            }
        ),
    )

    data = (fixtures_dir / "sample.pdf").read_bytes()

    response = client.post(
        "/upload/",
        files={"file": ("..\\..\\escaped.pdf", data, "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "escaped.pdf"


def test_upload_reports_a_duplicate(client: TestClient, monkeypatch, fixtures_dir):
    from app.api import upload as upload_module

    monkeypatch.setattr(
        upload_module.PDFService,
        "process",
        staticmethod(
            lambda path: {
                "pages": [],
                "chunks": [],
                "document_id": "abc",
                "duplicate": True,
            }
        ),
    )

    data = (fixtures_dir / "sample.pdf").read_bytes()

    response = client.post(
        "/upload/",
        files={"file": ("sample.pdf", data, "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["duplicate"] is True


def test_upload_happy_path_returns_counts(client: TestClient, monkeypatch, fixtures_dir):
    from app.api import upload as upload_module

    monkeypatch.setattr(
        upload_module.PDFService,
        "process",
        staticmethod(
            lambda path: {
                "pages": [{"page": 1, "text": "x"}],
                "chunks": [
                    {"text": "x", "page": 1, "chunk": 1},
                    {"text": "y", "page": 1, "chunk": 2},
                ],
                "document_id": "abc",
                "duplicate": False,
            }
        ),
    )

    data = (fixtures_dir / "sample.pdf").read_bytes()

    response = client.post(
        "/upload/",
        files={"file": ("sample.pdf", data, "application/pdf")},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["pages"] == 1
    assert body["chunks"] == 2
    assert body["duplicate"] is False


def test_upload_maps_an_oversized_document_to_413(
    client: TestClient,
    monkeypatch,
    fixtures_dir,
):
    from app.api import upload as upload_module

    def too_large(path):
        raise DocumentTooLargeError("too many chunks")

    monkeypatch.setattr(
        upload_module.PDFService,
        "process",
        staticmethod(too_large),
    )

    data = (fixtures_dir / "sample.pdf").read_bytes()

    response = client.post(
        "/upload/",
        files={"file": ("sample.pdf", data, "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "too many chunks"
