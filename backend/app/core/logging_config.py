"""Logging configuration and request-aware log filtering."""

import logging
import sys

from app.core.request_context import request_id_var

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()

        return True


def configure_logging(level: str = "INFO") -> None:
    """
    Send application logs to stdout with a request id on every line.

    Called once at startup from main.py.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Uvicorn installs its own handlers; route them through ours so nothing
    # is printed twice and log lines keep the same shape.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # The request middleware already logs method, path, status and latency
    # with a request id, so Uvicorn's access log would just duplicate it.
    logging.getLogger("uvicorn.access").disabled = True
