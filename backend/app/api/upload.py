"""PDF upload endpoint."""

import logging
import os
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.services.pdf_service import PDFService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])

UPLOAD_FOLDER = Path(settings.UPLOAD_DIRECTORY)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

PDF_MAGIC = b"%PDF-"
READ_CHUNK_SIZE = 1024 * 1024


def safe_filename(raw_name: str | None) -> str:
    """
    Reduce a client-supplied name to a bare, safe file name.

    Separators are normalised before splitting because Path(...).name only
    strips the separators of the running OS: on Linux a backslash is an
    ordinary character, so `..\\..\\x.pdf` would survive as a literal name.
    Leading dots are stripped so no hidden file can be created.
    """
    name = (raw_name or "").replace("\\", "/").split("/")[-1]

    return name.lstrip(".")


@router.post("/")
async def upload_pdf(
    file: Annotated[UploadFile, File()],
):
    """Store and index an uploaded PDF."""
    filename = safe_filename(file.filename)

    if not filename:
        raise HTTPException(status_code=400, detail="A file name is required.")

    if len(filename) > settings.MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The file name is too long (limit {settings.MAX_FILENAME_LENGTH} characters)."
            ),
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    destination = UPLOAD_FOLDER / filename

    preexisting = destination.exists()

    # Stage the upload beside its destination and only move it into place
    # once it is proven to be a PDF. Writing straight to the destination
    # would truncate an existing good copy before any check had run.
    staging_path = destination.with_name(f".{uuid4().hex}.part")

    total_bytes = 0

    try:
        with staging_path.open("wb") as buffer:
            while True:
                chunk = await file.read(READ_CHUNK_SIZE)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit."),
                    )

                buffer.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        # Verify the real PDF signature instead of trusting content_type.
        with staging_path.open("rb") as handle:
            if handle.read(len(PDF_MAGIC)) != PDF_MAGIC:
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        os.replace(staging_path, destination)

    except HTTPException:
        staging_path.unlink(missing_ok=True)
        raise

    except OSError as exc:
        # A name the filesystem refuses (reserved device names on Windows,
        # for example) is client input, not a server bug.
        staging_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400, detail="The file name is not usable on this server."
        ) from exc

    try:
        # Parsing and embedding are blocking work, so it runs in the
        # threadpool to keep the event loop responsive.
        result = await run_in_threadpool(
            PDFService.process,
            destination,
        )

    except BaseException:
        # No failure path should leave an orphan file, but never delete a
        # copy that already existed before this request.
        if not preexisting:
            destination.unlink(missing_ok=True)

        raise

    logger.info(
        "ingested filename=%s pages=%s chunks=%s duplicate=%s",
        filename,
        len(result["pages"]),
        len(result["chunks"]),
        result["duplicate"],
    )

    if result["duplicate"]:
        return {
            "filename": filename,
            "duplicate": True,
            "message": "This document has already been uploaded.",
            "pages": 0,
            "chunks": 0,
        }

    return {
        "filename": filename,
        "duplicate": False,
        "pages": len(result["pages"]),
        "chunks": len(result["chunks"]),
        "index_replaced": result.get("index_replaced", False),
        "preview": result["chunks"][:3],
    }
