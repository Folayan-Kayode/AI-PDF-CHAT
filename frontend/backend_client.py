"""
Backend HTTP client for the Streamlit UI.

Kept separate from ``app.py`` (which executes Streamlit calls at import time)
so the user-facing error contract can be unit tested without a running UI.
"""

import os

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

# A near-limit document needs many paced embedding batches, which can exceed
# the chat timeout, so ingest gets its own, longer budget.
UPLOAD_TIMEOUT_SECONDS = int(os.getenv("UPLOAD_TIMEOUT_SECONDS", "1800"))

# The paid endpoints require this. Empty means every /upload and /chat call
# will be rejected with 401, which is the intended fail-closed behaviour.
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")


def api_headers() -> dict[str, str]:
    """Headers every backend call sends."""
    return {"X-API-Key": BACKEND_API_KEY}


def backend_error_message(response) -> str:
    """Turn an error response into something a user can act on."""
    detail = None

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail")

    if isinstance(detail, list):
        # FastAPI validation errors arrive as a list of objects.
        message = "; ".join(
            str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail
        )
    elif detail:
        message = str(detail)
    else:
        message = f"Request failed with HTTP {response.status_code}."

    return message + retry_hint(response)


def retry_hint(response) -> str:
    """Extra guidance for statuses the user can actually act on."""
    if response.status_code == 401:
        return " The request was not authorised; check the backend API key."

    if response.status_code == 429:
        wait = response.headers.get("Retry-After")

        if wait:
            return f" Try again in about {wait} seconds."

        return " The provider asked us to slow down; try again shortly."

    if response.status_code == 503:
        return " The service is temporarily unavailable."

    if response.status_code == 409:
        return " Another upload is still running."

    return ""


def unreachable_message(exc) -> str:
    return f"Could not reach the backend at {BACKEND_URL}. Is it running? ({type(exc).__name__})"


def upload_pdf(uploaded_file) -> tuple[bool, str]:
    """Upload a PDF. Returns (ok, message)."""
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}

    try:
        response = requests.post(
            f"{BACKEND_URL}/upload/",
            files=files,
            headers=api_headers(),
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        # The request was delivered, so the backend may well be indexing it
        # right now. Saying "unreachable" here would be a lie and would push
        # the user into re-uploading a document that is already succeeding.
        return False, (
            f"The upload did not finish within {UPLOAD_TIMEOUT_SECONDS} seconds. "
            "The backend may still be indexing it - wait a moment and check "
            "before uploading again."
        )
    except requests.RequestException as exc:
        return False, unreachable_message(exc)

    if response.status_code != 200:
        return False, backend_error_message(response)

    payload = response.json()

    if payload.get("duplicate"):
        return True, f"{uploaded_file.name} was already uploaded."

    return True, (
        f"{uploaded_file.name} uploaded successfully "
        f"({payload.get('pages')} pages, {payload.get('chunks')} chunks)."
    )


def ask_question(question: str) -> tuple[dict | None, str | None]:
    """Ask the backend a question. Returns (payload, error)."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat/",
            json={"question": question},
            headers=api_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return None, ("The question timed out. The model provider may be busy; please try again.")
    except requests.RequestException as exc:
        return None, unreachable_message(exc)

    if response.status_code != 200:
        return None, backend_error_message(response)

    return response.json(), None
