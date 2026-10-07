"""Streamlit UI for AI PDF Chat."""

import streamlit as st

from backend_client import ask_question, upload_pdf

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
