"""Streamlit UI for AI PDF Chat."""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

st.set_page_config(
    page_title="AI PDF Chat",
    page_icon="📄",
    layout="wide"
)

# ----------------------------
# Session State
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# ----------------------------
# Backend helpers
# ----------------------------


def backend_error_message(response):
    """Turn an error response into something a user can act on."""
    try:
        payload = response.json()
    except ValueError:
        return f"Request failed with HTTP {response.status_code}."

    detail = payload.get("detail") if isinstance(payload, dict) else None

    if isinstance(detail, list):
        # FastAPI validation errors arrive as a list of objects.
        return "; ".join(
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        )

    if detail:
        return str(detail)

    return f"Request failed with HTTP {response.status_code}."


def unreachable_message(exc):
    return (
        f"Could not reach the backend at {BACKEND_URL}. "
        f"Is it running? ({type(exc).__name__})"
    )


def upload_pdf(uploaded_file):
    """Upload a PDF. Returns (ok, message)."""
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file,
            "application/pdf"
        )
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/upload/",
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
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
            json={
                "question": question
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
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

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

# Automatically upload only once
if (
    uploaded_file is not None
    and uploaded_file.name != st.session_state.uploaded_file_name
):

    with st.spinner("Uploading PDF..."):

        ok, message = upload_pdf(uploaded_file)

    if ok:

        st.success(message)

        st.session_state.uploaded_file_name = uploaded_file.name

    else:

        st.error(message)

st.divider()

# ----------------------------
# Chat Section
# ----------------------------

question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:

    with st.spinner("Model is thinking..."):

        answer, error = ask_question(question)

    if error:

        st.error(error)

    else:

        st.session_state.messages.append(
            {
                "question": question,
                "answer": answer["answer"],
                "sources": answer["sources"]
            }
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

                    st.write(
                        f"Page {source['page']} • Chunk {source['chunk']}"
                    )
