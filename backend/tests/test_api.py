"""End-to-end tests for the HTTP API. The LLM is always mocked."""

import logging

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.exceptions import (
    DocumentTooLargeError,
    IngestionInProgressError,
    UpstreamRateLimitError,
    UpstreamUnavailableError,
)

# --------------------------------------------------------------------------
# Metadata, liveness and readiness
# --------------------------------------------------------------------------


def test_root_reports_the_generation_model(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["model"] == "deepseek-chat"
    assert body["project"]


def test_health_is_a_cheap_liveness_check(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["generation_model"] == "deepseek-chat"
    # Liveness must not touch the vector store.
    assert "indexed_chunks" not in body


def test_health_reports_the_worker_count(client: TestClient):
    # B4 made explicit: the app runs exactly one worker, and /health says so.
    assert client.get("/health").json()["workers"] == 1


def test_ready_reports_registered_documents(client: TestClient):
    # B3: reported so "the volume did not persist" is distinguishable from
    # "nothing was uploaded".
    assert "registered_documents" in client.get("/ready").json()


def test_ready_reports_a_degraded_empty_index(client: TestClient):
    response = client.get("/ready")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "degraded"
    assert body["indexed_chunks"] == 0
    assert body["embedding_model_match"] is True


def test_ready_returns_503_when_the_index_is_unavailable(
    client: TestClient,
    monkeypatch,
):
    from app.api import health as health_module

    class BrokenDatabase:
        def count(self):
            raise RuntimeError("index is down")

        def indexed_fingerprint(self, sample_size=200):
            raise RuntimeError("index is down")

    monkeypatch.setattr(health_module, "get_database", lambda: BrokenDatabase())

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"]


def test_ready_reports_an_embedding_model_mismatch(client: TestClient, monkeypatch):
    from app.api import health as health_module

    class MismatchedDatabase:
        def __init__(self):
            self.collection = type(
                "Collection",
                (),
                {"get": lambda self, **kwargs: {"metadatas": []}},
            )()

        def count(self):
            return 12

        def indexed_fingerprint(self, sample_size=200):
            return {
                "embedding_models": {"some-other-model"},
                "schema_versions": set(),
            }

        def space_matches(self):
            return True

    monkeypatch.setattr(health_module, "get_database", lambda: MismatchedDatabase())

    body = client.get("/ready").json()

    assert body["embedding_model_match"] is False
    assert body["status"] == "degraded"
    assert body["indexed_embedding_models"] == ["some-other-model"]


def test_ready_reports_a_vector_space_mismatch(client: TestClient, monkeypatch):
    from app.api import health as health_module

    class WrongSpaceDatabase:
        def __init__(self):
            self.collection = type(
                "Collection",
                (),
                {"get": lambda self, **kwargs: {"metadatas": []}},
            )()

        def count(self):
            return 12

        def indexed_fingerprint(self, sample_size=200):
            return {"embedding_models": set(), "schema_versions": set()}

        def space_matches(self):
            return False

    monkeypatch.setattr(health_module, "get_database", lambda: WrongSpaceDatabase())

    body = client.get("/ready").json()

    assert body["vector_space_match"] is False
    assert body["status"] == "degraded"


def test_ready_describes_each_document(client: TestClient, monkeypatch):
    from app.api import health as health_module

    class OneDocumentDatabase:
        def __init__(self):
            metadata = {
                "document_id": "doc-a",
                "embedding_model": settings.EMBEDDING_MODEL,
                "schema_version": settings.EMBEDDING_SCHEMA_VERSION,
            }

            self.collection = type(
                "Collection",
                (),
                {"get": lambda self, **kwargs: {"metadatas": [metadata, metadata]}},
            )()

        def count(self):
            return 2

        def indexed_fingerprint(self, sample_size=200):
            return {
                "embedding_models": {settings.EMBEDDING_MODEL},
                "schema_versions": {settings.EMBEDDING_SCHEMA_VERSION},
            }

        def space_matches(self):
            return True

    monkeypatch.setattr(health_module, "get_database", lambda: OneDocumentDatabase())

    body = client.get("/ready").json()

    assert body["status"] == "ready"
    assert body["documents"] == [
        {
            "document_id": "doc-a",
            "chunks": 2,
            "embedding_model": settings.EMBEDDING_MODEL,
            "schema_version": settings.EMBEDDING_SCHEMA_VERSION,
        }
    ]


def test_ready_deep_probe_is_cached(client: TestClient, monkeypatch):
    from app.api import health as health_module

    calls: list[str] = []

    class FakeEmbedder:
        def embed_query(self, query):
            calls.append(query)
            return [0.1, 0.2]

    monkeypatch.setattr(health_module, "get_embedding_model", lambda: FakeEmbedder())
    monkeypatch.setattr(
        health_module,
        "_deep_check",
        {"checked_at": 0.0, "result": None},
    )

    first = client.get("/ready?deep=true").json()
    second = client.get("/ready?deep=true").json()

    assert first["embedding_provider"] == "ok"
    assert second["embedding_provider"] == "ok"
    assert len(calls) == 1, "the probe should be cached"


def test_ready_deep_probe_reports_a_provider_failure(client: TestClient, monkeypatch):
    from app.api import health as health_module

    class BrokenEmbedder:
        def embed_query(self, query):
            raise RuntimeError("quota exhausted")

    monkeypatch.setattr(health_module, "get_embedding_model", lambda: BrokenEmbedder())
    monkeypatch.setattr(
        health_module,
        "_deep_check",
        {"checked_at": 0.0, "result": None},
    )

    body = client.get("/ready?deep=true").json()

    assert body["embedding_provider"] == "error"


def test_request_id_is_echoed_back(client: TestClient):
    response = client.get("/health", headers={"X-Request-ID": "abc123"})

    assert response.headers["X-Request-ID"] == "abc123"


def test_request_id_is_generated_when_absent(client: TestClient):
    assert client.get("/health").headers["X-Request-ID"]


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


def test_chat_rejects_a_blank_question(client: TestClient):
    assert client.post("/chat/", json={"question": "   "}).status_code == 422


def test_chat_rejects_a_long_question(client: TestClient):
    assert client.post("/chat/", json={"question": "x" * 2001}).status_code == 422


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


def test_chat_maps_an_upstream_failure_to_json_503(client: TestClient, monkeypatch):
    from app.services import chat_service

    def explode(question):
        raise UpstreamUnavailableError("provider is down", service="generation")

    monkeypatch.setattr(chat_service.ChatService, "chat", staticmethod(explode))

    response = client.post("/chat/", json={"question": "what is it?"})

    assert response.status_code == 503
    assert response.json()["detail"] == "provider is down"


def test_rate_limit_returns_429_with_retry_after(client: TestClient, monkeypatch):
    from app.services import chat_service

    def explode(question):
        raise UpstreamRateLimitError(
            "slow down",
            service="generation",
            retry_after_seconds=42,
        )

    monkeypatch.setattr(chat_service.ChatService, "chat", staticmethod(explode))

    response = client.post("/chat/", json={"question": "what is it?"})

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "42"
    assert response.json()["detail"] == "slow down"


def test_chat_maps_an_unexpected_failure_to_json_500(client: TestClient, monkeypatch):
    from app.services import chat_service

    def explode(question):
        raise ValueError("something unexpected")

    monkeypatch.setattr(chat_service.ChatService, "chat", staticmethod(explode))

    response = client.post("/chat/", json={"question": "what is it?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."


# --------------------------------------------------------------------------
# Server-error logging (B3)
# --------------------------------------------------------------------------


def test_server_errors_log_the_underlying_cause(
    client: TestClient,
    monkeypatch,
    caplog,
):
    from app.services import chat_service

    def explode(question):
        try:
            raise RuntimeError("the underlying chroma failure")
        except RuntimeError as cause:
            raise UpstreamUnavailableError("Could not write to the document index.") from cause

    monkeypatch.setattr(chat_service.ChatService, "chat", staticmethod(explode))

    with caplog.at_level(logging.ERROR):
        response = client.post("/chat/", json={"question": "what?"})

    assert response.status_code == 503
    assert "the underlying chroma failure" in caplog.text


def test_client_errors_log_without_a_traceback(
    client: TestClient,
    monkeypatch,
    caplog,
):
    from app.services import chat_service

    def explode(question):
        raise DocumentTooLargeError("too big")

    monkeypatch.setattr(chat_service.ChatService, "chat", staticmethod(explode))

    with caplog.at_level(logging.WARNING):
        response = client.post("/chat/", json={"question": "what?"})

    assert response.status_code == 413
    assert "Traceback" not in caplog.text


# --------------------------------------------------------------------------
# Filename handling (A1, A2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("..\\..\\escaped.pdf", "escaped.pdf"),
        ("../../escaped.pdf", "escaped.pdf"),
        ("/etc/escaped.pdf", "escaped.pdf"),
        ("C:\\Users\\someone\\report.pdf", "report.pdf"),
        (".hidden.pdf", "hidden.pdf"),
        ("plain.pdf", "plain.pdf"),
    ],
)
def test_safe_filename_is_platform_independent(raw, expected):
    from app.api.upload import safe_filename

    assert safe_filename(raw) == expected


def test_safe_filename_rejects_a_directory_only_name():
    from app.api.upload import safe_filename

    assert safe_filename("../../") == ""
    assert safe_filename(None) == ""


def test_upload_rejects_an_overlong_filename(client: TestClient, sample_pdf_bytes):
    name = "a" * 250 + ".pdf"

    response = client.post(
        "/upload/",
        files={"file": (name, sample_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 400
    assert "too long" in response.json()["detail"]


def test_upload_maps_filesystem_errors_to_400(
    client: TestClient,
    monkeypatch,
    sample_pdf_bytes,
):
    from app.api import upload as upload_module

    def explode(source, destination):
        raise OSError("invalid file name")

    monkeypatch.setattr(upload_module.os, "replace", explode)

    response = client.post(
        "/upload/",
        files={"file": ("odd.pdf", sample_pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 400
    assert "not usable" in response.json()["detail"]


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

    monkeypatch.setattr(upload_module.settings, "MAX_UPLOAD_SIZE_BYTES", 1024)

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


def test_bad_upload_does_not_overwrite_an_existing_file(
    client: TestClient,
):
    from app.api import upload as upload_module

    target = upload_module.UPLOAD_FOLDER / "keep-me.pdf"
    target.write_bytes(b"%PDF-1.4 the original, good document")

    try:
        response = client.post(
            "/upload/",
            files={"file": ("keep-me.pdf", b"not a pdf at all", "application/pdf")},
        )

        assert response.status_code == 400
        assert target.read_bytes() == b"%PDF-1.4 the original, good document"
    finally:
        target.unlink(missing_ok=True)


def test_staging_files_are_cleaned_up(client: TestClient):
    from app.api import upload as upload_module

    client.post(
        "/upload/",
        files={"file": ("truncated.pdf", b"%PDF-1.4 too short", "application/pdf")},
    )

    leftovers = list(upload_module.UPLOAD_FOLDER.glob("*.part"))

    assert leftovers == []


# --------------------------------------------------------------------------
# Upload outcomes
# --------------------------------------------------------------------------


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
                "index_replaced": True,
            }
        ),
    )

    response = client.post(
        "/upload/",
        files={
            "file": (
                "..\\..\\escaped.pdf",
                (fixtures_dir / "sample.pdf").read_bytes(),
                "application/pdf",
            )
        },
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
                "index_replaced": False,
            }
        ),
    )

    response = client.post(
        "/upload/",
        files={
            "file": (
                "sample.pdf",
                (fixtures_dir / "sample.pdf").read_bytes(),
                "application/pdf",
            )
        },
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
                "index_replaced": True,
            }
        ),
    )

    response = client.post(
        "/upload/",
        files={
            "file": (
                "sample.pdf",
                (fixtures_dir / "sample.pdf").read_bytes(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["pages"] == 1
    assert body["chunks"] == 2
    assert body["duplicate"] is False
    assert body["index_replaced"] is True


def test_upload_maps_an_oversized_document_to_413(
    client: TestClient,
    monkeypatch,
    fixtures_dir,
):
    from app.api import upload as upload_module

    def too_large(path):
        raise DocumentTooLargeError("too many chunks")

    monkeypatch.setattr(upload_module.PDFService, "process", staticmethod(too_large))

    response = client.post(
        "/upload/",
        files={
            "file": (
                "sample.pdf",
                (fixtures_dir / "sample.pdf").read_bytes(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "too many chunks"


def test_concurrent_upload_returns_409(
    client: TestClient,
    monkeypatch,
    fixtures_dir,
):
    from app.api import upload as upload_module

    def busy(path):
        raise IngestionInProgressError("Another document is being ingested.")

    monkeypatch.setattr(upload_module.PDFService, "process", staticmethod(busy))

    response = client.post(
        "/upload/",
        files={
            "file": (
                "busy.pdf",
                (fixtures_dir / "sample.pdf").read_bytes(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 409
    # The rejected upload must not be left behind either.
    assert not (upload_module.UPLOAD_FOLDER / "busy.pdf").exists()


def test_unexpected_ingest_failure_leaves_no_file(
    client: TestClient,
    monkeypatch,
    fixtures_dir,
):
    from app.api import upload as upload_module

    def explode(path):
        raise RuntimeError("unexpected ingest failure")

    monkeypatch.setattr(upload_module.PDFService, "process", staticmethod(explode))

    response = client.post(
        "/upload/",
        files={
            "file": (
                "orphan.pdf",
                (fixtures_dir / "sample.pdf").read_bytes(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."
    assert not (upload_module.UPLOAD_FOLDER / "orphan.pdf").exists()
