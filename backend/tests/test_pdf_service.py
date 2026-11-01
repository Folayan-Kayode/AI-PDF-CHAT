"""Tests for ingestion: limits, metadata, safe replacement and locking."""

from pathlib import Path

import pytest

from app.core.config import settings
from app.core.exceptions import (
    DocumentTooLargeError,
    IngestionInProgressError,
    RetrievalError,
)
from app.services import pdf_service
from app.services.pdf_service import PDFService


class FakeEmbedder:
    """Records the batches it was asked to embed."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts):
        self.batch_sizes.append(len(texts))
        return [[0.1, 0.2] for _ in texts]


class FakeDatabase:
    """
    In-memory stand-in for ChromaDatabase.

    ``add_documents`` can fail before writing or after writing part of the
    document, so both rollback paths can be exercised.
    """

    def __init__(self):
        self.stored: dict[str, tuple] = {}
        self.writes: list[tuple] = []
        self.deleted: list[str] = []
        self.raise_on_write = False
        self.raise_after_store = False

    def has_document(self, document_id: str) -> bool:
        return document_id in self.stored

    def add_documents(self, ids, documents, embeddings, metadatas) -> None:
        self.writes.append((ids, documents, embeddings, metadatas))

        document_id = metadatas[0]["document_id"]

        if self.raise_on_write:
            raise RetrievalError("Could not write to the document index.")

        if self.raise_after_store:
            # Simulate a failure after the first batch was committed.
            self.stored[document_id] = (ids[:1], documents[:1], metadatas[:1])
            raise RetrievalError("Could not write to the document index.")

        self.stored[document_id] = (ids, documents, metadatas)

    def delete_document(self, document_id: str) -> int:
        self.deleted.append(document_id)
        self.stored.pop(document_id, None)

        return 1

    def count(self) -> int:
        return sum(len(entry[0]) for entry in self.stored.values())


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def database() -> FakeDatabase:
    return FakeDatabase()


def test_ingest_records_ids_metadata_and_fingerprint(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
):
    result = PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=embedder,
    )

    assert result["duplicate"] is False
    assert result["index_replaced"] is False
    assert result["document_id"]

    ids, documents, embeddings, metadatas = database.writes[-1]

    assert len(ids) == len(documents) == len(embeddings) == len(metadatas)
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    assert all(identifier.startswith(f"{result['document_id']}_") for identifier in ids)

    first = metadatas[0]

    assert first["embedding_model"] == settings.EMBEDDING_MODEL
    assert first["schema_version"] == settings.EMBEDDING_SCHEMA_VERSION
    assert first["document_id"] == result["document_id"]
    assert first["page"] == 1


def test_chunk_limit_is_enforced(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
    monkeypatch,
):
    monkeypatch.setattr(settings, "MAX_CHUNKS_PER_DOCUMENT", 0)

    with pytest.raises(DocumentTooLargeError):
        PDFService.process(
            fixtures_dir / "sample.pdf",
            database=database,
            embedding_model=embedder,
        )

    assert database.writes == [], "nothing should be written past the limit"


def test_duplicate_is_reported_and_not_reindexed(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
):
    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=embedder,
    )

    writes_after_first = len(database.writes)

    result = PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=embedder,
    )

    assert result["duplicate"] is True
    assert result["index_replaced"] is False
    assert len(database.writes) == writes_after_first


def test_failed_write_keeps_the_previous_document(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
    monkeypatch,
):
    hashes = iter(["document-a", "document-b"])

    monkeypatch.setattr(
        PDFService,
        "file_hash",
        staticmethod(lambda path: next(hashes)),
    )

    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=embedder,
    )

    database.raise_on_write = True

    with pytest.raises(RetrievalError) as excinfo:
        PDFService.process(
            fixtures_dir / "sample.pdf",
            database=database,
            embedding_model=embedder,
        )

    assert "unchanged" in str(excinfo.value).lower()
    assert set(database.stored) == {"document-a"}
    assert "document-b" in database.deleted


def test_failed_write_removes_a_partially_written_document(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
    monkeypatch,
):
    hashes = iter(["document-a", "document-b"])

    monkeypatch.setattr(
        PDFService,
        "file_hash",
        staticmethod(lambda path: next(hashes)),
    )

    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=embedder,
    )

    database.raise_after_store = True

    with pytest.raises(RetrievalError):
        PDFService.process(
            fixtures_dir / "sample.pdf",
            database=database,
            embedding_model=embedder,
        )

    # The half-written document is removed; the earlier one is untouched.
    assert set(database.stored) == {"document-a"}
    assert "document-b" in database.deleted


def test_two_documents_are_appended(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
    monkeypatch,
):
    hashes = iter(["document-a", "document-b"])

    monkeypatch.setattr(
        PDFService,
        "file_hash",
        staticmethod(lambda path: next(hashes)),
    )

    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=embedder,
    )

    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=embedder,
    )

    assert set(database.stored) == {"document-a", "document-b"}
    assert database.count() == sum(len(entry[0]) for entry in database.stored.values())


def test_total_chunk_limit_is_enforced(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
    monkeypatch,
):
    monkeypatch.setattr(settings, "MAX_TOTAL_CHUNKS", 0)

    with pytest.raises(DocumentTooLargeError):
        PDFService.process(
            fixtures_dir / "sample.pdf",
            database=database,
            embedding_model=embedder,
        )

    assert database.writes == [], "nothing should be written past the limit"


def test_document_limit_is_enforced(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
    monkeypatch,
):
    monkeypatch.setattr(settings, "MAX_DOCUMENTS", 0)

    with pytest.raises(DocumentTooLargeError):
        PDFService.process(
            fixtures_dir / "sample.pdf",
            database=database,
            embedding_model=embedder,
        )

    assert database.writes == [], "nothing should be written past the limit"


def test_concurrent_ingest_is_rejected(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
):
    assert pdf_service._INGESTION_LOCK.acquire(timeout=0)

    try:
        with pytest.raises(IngestionInProgressError):
            PDFService.process(
                fixtures_dir / "sample.pdf",
                database=database,
                embedding_model=embedder,
            )
    finally:
        pdf_service._INGESTION_LOCK.release()


def test_lock_is_released_after_a_failure(
    fixtures_dir: Path,
    database: FakeDatabase,
    embedder: FakeEmbedder,
    monkeypatch,
):
    monkeypatch.setattr(settings, "MAX_CHUNKS_PER_DOCUMENT", 0)

    with pytest.raises(DocumentTooLargeError):
        PDFService.process(
            fixtures_dir / "sample.pdf",
            database=database,
            embedding_model=embedder,
        )

    assert not pdf_service._INGESTION_LOCK.locked()


def test_chunks_are_embedded_once(fixtures_dir: Path, database: FakeDatabase):
    embedder = FakeEmbedder()

    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=embedder,
    )

    assert sum(embedder.batch_sizes) == len(database.writes[-1][0])


def test_file_hash_is_stable_and_content_specific(fixtures_dir: Path, tmp_path: Path):
    sample = fixtures_dir / "sample.pdf"
    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.4 different")

    assert PDFService.file_hash(sample) == PDFService.file_hash(sample)
    assert PDFService.file_hash(sample) != PDFService.file_hash(other)


# --------------------------------------------------------------------------
# Document profile chunk
# --------------------------------------------------------------------------


class FakeSummarizer:
    """Stands in for the ingest-time profile builder."""

    def __init__(self, text: str | None = "Title: Example\nPublisher: Cengage"):
        self.text = text
        self.pages_seen: list[list[dict]] = []
        self.outlines_seen: list = []
        self.metadata_seen: list = []

    def profile(self, pages, outline=None, metadata=None):
        self.pages_seen.append(pages)
        self.outlines_seen.append(outline)
        self.metadata_seen.append(metadata)
        return self.text


def test_profile_chunk_is_indexed_first(fixtures_dir: Path, database: FakeDatabase):
    embedder = FakeEmbedder()

    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=embedder,
        summarizer=FakeSummarizer("Title: Sample Book\nPublisher: Cengage"),
    )

    ids, documents, embeddings, metadatas = database.writes[-1]

    assert documents[0].startswith("Document profile:")
    assert "Publisher: Cengage" in documents[0]
    assert metadatas[0]["kind"] == "document_summary"
    assert metadatas[0]["chunk"] == 0
    assert metadatas[1]["kind"] == "content"

    assert len(ids) == len(documents) == len(embeddings) == len(metadatas)
    assert len(ids) == len(set(ids))


def test_profile_receives_the_extracted_pages(fixtures_dir: Path, database: FakeDatabase):
    summarizer = FakeSummarizer()

    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=FakeEmbedder(),
        summarizer=summarizer,
    )

    assert summarizer.pages_seen
    assert summarizer.pages_seen[0][0]["page"] == 1


def test_no_profile_chunk_when_the_profile_is_unavailable(
    fixtures_dir: Path,
    database: FakeDatabase,
):
    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=FakeEmbedder(),
        summarizer=FakeSummarizer(None),
    )

    _, _, _, metadatas = database.writes[-1]

    assert all(metadata["kind"] == "content" for metadata in metadatas)


def test_no_profile_chunk_when_disabled(fixtures_dir: Path, database: FakeDatabase):
    # The default summarizer is used and DOCUMENT_SUMMARY_ENABLED is false.
    PDFService.process(
        fixtures_dir / "sample.pdf",
        database=database,
        embedding_model=FakeEmbedder(),
    )

    _, _, _, metadatas = database.writes[-1]

    assert all(metadata["kind"] == "content" for metadata in metadatas)


def test_profile_counts_toward_the_chunk_limit(
    fixtures_dir: Path,
    database: FakeDatabase,
    monkeypatch,
):
    monkeypatch.setattr(settings, "MAX_CHUNKS_PER_DOCUMENT", 1)

    # The single content chunk plus the profile already exceeds a limit of 1.
    with pytest.raises(DocumentTooLargeError):
        PDFService.process(
            fixtures_dir / "sample.pdf",
            database=database,
            embedding_model=FakeEmbedder(),
            summarizer=FakeSummarizer(),
        )


# --------------------------------------------------------------------------
# Deletion
# --------------------------------------------------------------------------


def _registry_with(tmp_path, *rows):
    from app.database.registry import DocumentRegistry

    registry = DocumentRegistry(path=tmp_path / "registry.sqlite3")

    for row in rows:
        registry.record(
            pages=1,
            chunks=1,
            embedding_model=settings.EMBEDDING_MODEL,
            schema_version=settings.EMBEDDING_SCHEMA_VERSION,
            **row,
        )

    return registry


def test_delete_document_removes_vectors_registry_and_file(tmp_path, monkeypatch):
    registry = _registry_with(tmp_path, {"document_id": "doc-a", "filename": "a.pdf"})

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "a.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(pdf_service, "get_registry", lambda: registry)
    monkeypatch.setattr(settings, "UPLOAD_DIRECTORY", str(uploads))

    database = FakeDatabase()
    database.stored["doc-a"] = (["a_1_1"], ["x"], [{"document_id": "doc-a"}])

    record = PDFService.delete_document("doc-a", database=database)

    assert record is not None
    assert record["filename"] == "a.pdf"
    assert database.deleted == ["doc-a"]
    assert registry.get("doc-a") is None
    assert not (uploads / "a.pdf").exists()


def test_delete_document_keeps_a_file_shared_by_another_document(
    tmp_path,
    monkeypatch,
):
    registry = _registry_with(
        tmp_path,
        {"document_id": "doc-a", "filename": "shared.pdf"},
        {"document_id": "doc-b", "filename": "shared.pdf"},
    )

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "shared.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(pdf_service, "get_registry", lambda: registry)
    monkeypatch.setattr(settings, "UPLOAD_DIRECTORY", str(uploads))

    database = FakeDatabase()
    database.stored["doc-a"] = (["a_1_1"], ["x"], [{"document_id": "doc-a"}])

    PDFService.delete_document("doc-a", database=database)

    # doc-b still uses the same filename, so the file must survive.
    assert (uploads / "shared.pdf").exists()


def test_delete_unknown_document_returns_none(tmp_path, monkeypatch):
    registry = _registry_with(tmp_path)

    monkeypatch.setattr(pdf_service, "get_registry", lambda: registry)

    assert PDFService.delete_document("missing", database=FakeDatabase()) is None


def test_delete_is_rejected_while_ingesting():
    assert pdf_service._INGESTION_LOCK.acquire(timeout=0)

    try:
        with pytest.raises(IngestionInProgressError):
            PDFService.delete_document("doc-a", database=FakeDatabase())
    finally:
        pdf_service._INGESTION_LOCK.release()
