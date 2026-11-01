"""Streamlit UI for AI PDF Chat."""

import streamlit as st

from backend_client import (
    BACKEND_API_KEY,
    ask_question,
    delete_document,
    list_documents,
    upload_pdf,
)

st.set_page_config(page_title="AI PDF Chat", page_icon="📄", layout="wide")

# Fail before the first upload/chat, with an explanation, instead of letting a
# missing key surface as a 401 the user cannot act on.
if not BACKEND_API_KEY:
    st.error(
        "No backend API key is configured. Set `BACKEND_API_KEY` to the "
        "backend's `API_KEY`, or add `API_KEY` to the repo-root `.env` "
        "(read automatically in local development)."
    )
    st.stop()

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
# Title
# ----------------------------

st.title("📄 AI PDF Chat")
st.caption("Retrieve information from your documents")

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
        with st.spinner("Uploading and indexing PDF..."):
            ok, message = upload_pdf(uploaded_file)

        if ok:
            st.success(message)

            st.session_state.uploaded_file_key = file_key
            st.session_state.failed_file_key = None

            # Refresh the picker so the new document is selectable.
            st.rerun()

        else:
            st.error(message)

            st.session_state.failed_file_key = file_key

st.divider()

# ----------------------------
# Documents
# ----------------------------

documents, documents_error = list_documents()

if documents_error:
    st.error(documents_error)

    documents = []

selected_document_id = None

if documents:
    by_id = {document["document_id"]: document for document in documents}

    labels = {
        f"{document['filename']} — {document.get('pages', '?')} pages, "
        f"{document.get('chunks', '?')} chunks": document["document_id"]
        for document in documents
    }

    selected_label = st.selectbox("Document", list(labels))
    selected_document_id = labels[selected_label]
    selected = by_id[selected_document_id]

    confirm = st.checkbox(f"Confirm deleting **{selected['filename']}**")

    if st.button("Delete selected document", disabled=not confirm):
        ok, message = delete_document(selected_document_id)

        if ok:
            st.success(message)
            st.session_state.uploaded_file_key = None

            st.rerun()

        else:
            st.error(message)

else:
    st.info("No documents yet. Upload a PDF to start.")

st.divider()

# ----------------------------
# Chat Section
# ----------------------------

question = None

if selected_document_id:
    question = st.chat_input(f"Ask a question about {by_id[selected_document_id]['filename']}...")

if question:
    with st.spinner("Model is thinking..."):
        answer, error = ask_question(question, document_id=selected_document_id)

    if error:
        st.error(error)

    else:
        st.session_state.messages.append(
            {
                "question": question,
                "answer": answer["answer"],
                "sources": answer["sources"],
                "retrieval": answer.get("retrieval") or {},
            }
        )

# ----------------------------
# Conversation
# ----------------------------

for chat in st.session_state.messages:
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        retrieval = chat.get("retrieval") or {}

        # A profile-only answer reads as grounded when it is not: nothing came
        # from the document body, so say so.
        if retrieval.get("retrieved_chunks") == 0 and retrieval.get("profile_used"):
            st.warning(
                "No passages matched this question, so the answer comes from "
                "the document profile alone."
            )

        st.write(chat["answer"])

        if chat["sources"]:
            with st.expander("Sources"):
                for source in chat["sources"]:
                    if source.get("kind") == "document_summary":
                        st.write("Document profile")
                    elif source.get("page_end", source.get("page")) > source.get(
                        "page_start", source.get("page", 0)
                    ):
                        st.write(
                            f"Pages {source.get('page_start')}–{source.get('page_end')}"
                            f" • Chunk {source['chunk']}"
                        )
                    else:
                        st.write(f"Page {source['page']} • Chunk {source['chunk']}")
