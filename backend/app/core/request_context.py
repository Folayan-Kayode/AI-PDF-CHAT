"""Per-request context shared across logging calls."""

from contextvars import ContextVar

# Populated by the request middleware so every log line for a request can be
# tied back to it via the X-Request-ID header.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
