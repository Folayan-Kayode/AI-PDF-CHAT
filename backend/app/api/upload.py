from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.exceptions import PDFProcessingError
from app.services.pdf_service import PDFService

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)

UPLOAD_FOLDER = Path(settings.UPLOAD_DIRECTORY)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

PDF_MAGIC = b"%PDF-"
READ_CHUNK_SIZE = 1024 * 1024


@router.post("/")
async def upload_pdf(
    file: UploadFile = File(...)
):
    # Path(file.filename).name strips any directory components sent by the
    # client, which prevents path traversal.
    filename = Path(file.filename or "").name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="A file name is required."
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    destination = UPLOAD_FOLDER / filename

    total_bytes = 0

    try:
        with destination.open("wb") as buffer:
            while True:
                chunk = await file.read(READ_CHUNK_SIZE)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File exceeds the "
                            f"{settings.MAX_UPLOAD_SIZE_MB} MB limit."
                        )
                    )

                buffer.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Verify the real PDF signature instead of trusting content_type.
        with destination.open("rb") as handle:
            if handle.read(len(PDF_MAGIC)) != PDF_MAGIC:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded file is not a valid PDF."
                )

    except HTTPException:
        destination.unlink(missing_ok=True)
        raise

    try:
        result = PDFService.process(destination)

    except PDFProcessingError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))

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
        "preview": result["chunks"][:3],
    }
