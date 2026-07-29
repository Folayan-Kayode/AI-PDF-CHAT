import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

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
# Title
# ----------------------------

st.title("📄 AI PDF Chat")
st.caption("Upload a PDF and chat with it using Gemini.")

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

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                "application/pdf"
            )
        }

        response = requests.post(
            f"{BACKEND_URL}/upload/",
            files=files
        )

    if response.status_code == 200:

        st.success(f"{uploaded_file.name} uploaded successfully!")

        st.session_state.uploaded_file_name = uploaded_file.name

    else:

        st.error(response.text)

st.divider()

# ----------------------------
# Chat Section
# ----------------------------

question = st.chat_input(
    "Ask a question about your PDF..."
)

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Gemini is thinking..."):

        response = requests.post(
            f"{BACKEND_URL}/chat/",
            json={
                "question": question
            }
        )

    if response.status_code == 200:

        answer = response.json()

        st.session_state.messages.append(
            {
                "question": question,
                "answer": answer["answer"],
                "sources": answer["sources"]
            }
        )

    else:

        st.error(response.text)

# ----------------------------
# Conversation
# ----------------------------

for chat in st.session_state.messages:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):

        st.write(chat["answer"])

        with st.expander("Sources"):

            for source in chat["sources"]:

                st.write(
                    f"Page {source['page']} • Chunk {source['chunk']}"
                )