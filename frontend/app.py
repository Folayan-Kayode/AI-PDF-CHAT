"""Streamlit UI for AI PDF Chat."""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

# A near-limit document needs many paced embedding batches, which can exceed
# the chat timeout, so ingest gets its own, longer budget.
UPLOAD_TIMEOUT_SECONDS = int(os.getenv("UPLOAD_TIMEOUT_SECONDS", "1800"))

st.set_page_config(page_title="AI PDF Chat", page_icon="📄", layout="wide")

# ----------------------------
# Session State
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = None

if "failed_file_key" not in st.session_state:
    st.session_state.failed_file_key = None

# ----------------------------
# Backend helpers
# ----------------------------


def backend_error_message(response):
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


def retry_hint(response):
    """Extra guidance for statuses the user can actually act on."""
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


def unreachable_message(exc):
    return f"Could not reach the backend at {BACKEND_URL}. Is it running? ({type(exc).__name__})"


def upload_pdf(uploaded_file):
    """Upload a PDF. Returns (ok, message)."""
    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}

    try:
        response = requests.post(
            f"{BACKEND_URL}/upload/",
            files=files,
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


def ask_question(question):
    """Ask the backend a question. Returns (payload, error)."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat/",
            json={"question": question},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return None, ("The question timed out. The model provider may be busy; please try again.")
    except requests.RequestException as exc:
        return None, unreachable_message(exc)

    if response.status_code != 200:
        return None, backend_error_message(response)

    return response.json(), None


# ----------------------------
# Title
# ----------------------------

st.title("📄 AI PDF Chat")
st.caption("Retrieve information from your document")

st.divider()

# ----------------------------
# Upload Section
# ----------------------------

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is not None:
    # Keyed on name and size, so a different document that happens to reuse
    # a previous name is still sent, and so a failure can be retried.
    file_key = (uploaded_file.name, uploaded_file.size)

    if st.session_state.failed_file_key == file_key:
        st.error("The last upload attempt for this file failed.")

        if st.button("Retry upload"):
            st.session_state.failed_file_key = None

            st.rerun()

    elif file_key != st.session_state.uploaded_file_key:
        with st.spinner("Uploading PDF..."):
            ok, message = upload_pdf(uploaded_file)

        if ok:
            st.success(message)

            st.session_state.uploaded_file_key = file_key
            st.session_state.failed_file_key = None

        else:
            st.error(message)

            st.session_state.failed_file_key = file_key

st.divider()

# ----------------------------
# Chat Section
# ----------------------------

question = st.chat_input("Ask a question about your PDF...")

if question:
    with st.spinner("Model is thinking..."):
        answer, error = ask_question(question)

    if error:
        st.error(error)

    else:
        st.session_state.messages.append(
            {"question": question, "answer": answer["answer"], "sources": answer["sources"]}
        )

# ----------------------------
# Conversation
# ----------------------------

for chat in st.session_state.messages:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

        if chat["sources"]:
            with st.expander("Sources"):
                for source in chat["sources"]:
                    st.write(f"Page {source['page']} • Chunk {source['chunk']}")
