"""FastAPI application entrypoint."""

import logging
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.upload import router as upload_router
from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging_config import configure_logging
from app.core.ratelimit import chat_rate_limit, upload_rate_limit
from app.core.request_context import request_id_var
from app.core.runtime import assert_single_worker
from app.core.security import require_api_key

configure_logging(settings.LOG_LEVEL)

# Fail fast on an unsupported topology (more than one worker) before the app
# starts serving, rather than corrupting the index quietly later.
assert_single_worker()

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# Client errors are expected traffic; anything else deserves a traceback.
_TRACEBACK_STATUS_THRESHOLD = 500


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
)

# CORS is only needed when a browser calls the API directly. The shipped
# Streamlit topology is server-to-server, so the allowlist defaults to empty
# and no middleware is added.
if settings.CORS_ORIGIN_LIST:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGIN_LIST,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
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
    """
    Map application errors onto their intended status code.

    Server-side failures log the underlying cause, not just the generic
    message, otherwise a 5xx cannot be diagnosed from the logs.
    """
    if exc.status_code >= _TRACEBACK_STATUS_THRESHOLD or exc.status_code == 429:
        logger.error(
            "%s: %s",
            type(exc).__name__,
            exc.detail,
            exc_info=exc.__cause__ or exc,
        )
    else:
        logger.warning("%s: %s", type(exc).__name__, exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers or None,
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

# The paid endpoints require the API key and are rate limited. /health, /ready
# and / stay open: platform probes must not need a credential, and those routes
# expose nothing sensitive.
app.include_router(
    upload_router,
    dependencies=[Depends(require_api_key), Depends(upload_rate_limit)],
)

app.include_router(
    chat_router,
    dependencies=[Depends(require_api_key), Depends(chat_rate_limit)],
)


@app.get("/")
def root() -> dict[str, Any]:
    """Basic service metadata."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "model": settings.MODEL_NAME,
    }
