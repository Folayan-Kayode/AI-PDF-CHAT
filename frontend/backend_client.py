"""
Backend HTTP client for the Streamlit UI.

Kept separate from ``app.py`` (which executes Streamlit calls at import time)
so the user-facing error contract can be unit tested without a running UI.
"""

import os
from pathlib import Path

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

# A near-limit document needs many paced embedding batches, which can exceed
# the chat timeout, so ingest gets its own, longer budget.
UPLOAD_TIMEOUT_SECONDS = int(os.getenv("UPLOAD_TIMEOUT_SECONDS", "1800"))

#: The repo-root .env. Used only as a local-development convenience: in the
#: container this file is absent and BACKEND_API_KEY (supplied by compose) wins.
DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def read_dotenv_value(path: str | Path, name: str) -> str | None:
    """
    Read one ``KEY=value`` from a dotenv file, or None.

    Deliberately small and local instead of adding python-dotenv to the
    frontend image: only one value is ever read, and the value is never logged.
    Handles comments, blank lines, ``export KEY=value`` and surrounding quotes.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for raw in lines:
        line = raw.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        key, _, value = line.partition("=")

        if key.strip() != name:
            continue

        value = value.strip().strip('"').strip("'")

        return value or None

    return None


def resolve_api_key(
    environment: dict[str, str] | None = None,
    dotenv_path: str | Path | None = None,
) -> tuple[str, str | None]:
    """
    Resolve the API key the UI should send, and say where it came from.

    Precedence:
      1. ``BACKEND_API_KEY`` - the explicit deployment setting (compose uses it)
      2. ``API_KEY`` in the process environment - one export configures both
      3. ``API_KEY`` in the repo-root ``.env`` - local development
      4. nothing, which the UI turns into a clear startup error rather than a
         401 on the first upload.

    Returns ``(value, source)``. Source is for messages only; never log value.
    """
    env = os.environ if environment is None else environment

    for name in ("BACKEND_API_KEY", "API_KEY"):
        value = (env.get(name) or "").strip()

        if value:
            return value, name

    path = Path(dotenv_path) if dotenv_path else DEFAULT_DOTENV_PATH

    value = read_dotenv_value(path, "API_KEY")

    if value:
        return value, str(path)

    return "", None


BACKEND_API_KEY, API_KEY_SOURCE = resolve_api_key()


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
