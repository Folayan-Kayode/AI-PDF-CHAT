"""API-key authentication for the expensive endpoints.

/upload runs hundreds of paid embedding calls and /chat spends a generation
call, so neither may be reachable without a credential. The key is required at
startup (see Settings.__init__), which means an unset value stops the app rather
than leaving the endpoints open by accident.

The key is compared with ``secrets.compare_digest`` so the check does not leak
the key through timing. It is never logged: the request middleware logs only
method, path, status and latency, and nothing here writes the header anywhere.
"""

import secrets

from fastapi import Header, HTTPException

from app.core.config import settings

#: Sent by clients on /upload and /chat. A header rather than a query
#: parameter so it does not end up in proxy access logs or browser history.
API_KEY_HEADER = "X-API-Key"


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """Reject a request whose API key is missing or wrong."""
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
