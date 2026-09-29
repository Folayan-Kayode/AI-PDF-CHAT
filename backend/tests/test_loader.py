"""Tests for PDF loading and its error paths."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.core.config import settings
from app.core.exceptions import (
    DocumentTooLargeError,
    EncryptedPDFError,
    PDFProcessingError,
    ScannedPDFError,
)
from app.rag.loader import PDFLoader


def test_extracts_text_from_a_real_pdf(fixtures_dir: Path):
    pages = PDFLoader(fixtures_dir / "sample.pdf").load()

    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "retention period" in pages[0]["text"]


def test_blank_pdf_is_reported_as_scanned(tmp_path: Path):
    path = tmp_path / "blank.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(ScannedPDFError):
        PDFLoader(path).load()


def test_encrypted_pdf_is_rejected(tmp_path: Path):
    path = tmp_path / "encrypted.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret-password")

    with path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(EncryptedPDFError):
        PDFLoader(path).load()


def test_corrupt_file_is_rejected(tmp_path: Path):
    path = tmp_path / "garbage.pdf"
    path.write_bytes(b"this is definitely not a pdf")

    with pytest.raises(PDFProcessingError):
        PDFLoader(path).load()


def test_missing_file_is_rejected(tmp_path: Path):
    with pytest.raises(PDFProcessingError):
        PDFLoader(tmp_path / "does-not-exist.pdf").load()


def test_page_limit_is_enforced(tmp_path: Path, monkeypatch):
    path = tmp_path / "two_pages.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)

    with path.open("wb") as handle:
        writer.write(handle)

    monkeypatch.setattr(settings, "MAX_PAGES_PER_DOCUMENT", 1)

    with pytest.raises(DocumentTooLargeError):
        PDFLoader(path).load()


def test_document_too_large_is_a_413():
    assert DocumentTooLargeError.status_code == 413
