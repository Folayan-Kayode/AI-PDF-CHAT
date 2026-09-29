"""FastAPI application entrypoint."""

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.upload import router as upload_router
from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging_config import configure_logging
from app.core.request_context import request_id_var

configure_logging(settings.LOG_LEVEL)

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
)


@app.middleware("http")
async def request_context(request: Request, call_next: Any) -> Response:
    """Attach a request id and log every request with its latency."""
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]

    token = request_id_var.set(request_id)

    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)

        status_code = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id

        return response

    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s -> %s (%.0f ms)",
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )

        request_id_var.reset(token)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Map application errors onto their intended status code."""
    level = logging.WARNING if exc.status_code < 500 else logging.ERROR

    logger.log(level, "%s: %s", type(exc).__name__, exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last resort.

    Without this, an unexpected failure returns a bare text/plain
    "Internal Server Error" that the UI cannot explain.
    """
    logger.exception("unhandled error: %s", exc)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


app.include_router(health_router)
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/")
def root() -> dict[str, Any]:
    """Basic service metadata."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "model": settings.MODEL_NAME,
    }
