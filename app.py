import streamlit as st
from utils.pdf_reader import extract_pdf_text
from utils.chunking import create_chunks
from utils.vector_store import (
    clear_collection,
    store_chunks,
    retrieve_chunks
)
from utils.llm import rewrite_question, generate_answer
# ---------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="PDF AI ChatBot",
    page_icon="📄",
    layout="wide"
)
st.title("📄 PDF AI ChatBot")
st.write(
    "Upload one or more PDFs and ask questions about them."
)
# ---------------------------------
# Session State
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "files" not in st.session_state:
    st.session_state.files = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
# ---------------------------------
# File Upload
# ---------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)
# ---------------------------------
# Sidebar
# ---------------------------------
with st.sidebar:
    st.header("Uploaded PDFs")
    if st.session_state.files:
        for file in st.session_state.files:
            st.write(f"✓ {file}")
    else:
        st.write("No PDFs uploaded.")
    st.write(
        f"Chunks : {st.session_state.total_chunks}"
    )
    status = (
        "Ready"
        if st.session_state.processed
        else "Not Ready"
    )
    st.write(
        f"Status : {status}"
    )
# ---------------------------------
# Process PDFs
# ---------------------------------
if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            docs = extract_pdf_text(uploaded_files)
            chunks = create_chunks(docs)
            clear_collection()
            store_chunks(chunks)
            st.session_state.processed = True
            st.session_state.total_chunks = len(
                chunks
            )
            st.session_state.files = [
                file.name
                for file in uploaded_files
            ]
        st.success(
            "PDFs processed successfully!"
        )
# ---------------------------------
# Display Previous Chat
# ---------------------------------
for message in st.session_state.messages:
    with st.chat_message(
            message["role"]
    ):
        st.markdown(
            message["content"]
        )
# ---------------------------------
# Ask Questions
# ---------------------------------
if not st.session_state.processed:
    st.warning(
        "Please upload and process PDFs first."
    )
else:
    question = st.chat_input(
        "Ask something about your PDFs..."
    )
    if question:
        # -----------------------------
        # Store User Question
        # -----------------------------
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )
        with st.chat_message("user"):
            st.markdown(question)
        # -----------------------------
        # Rewrite Follow-up Question
        # -----------------------------
        standalone_question = rewrite_question(
            question,
            st.session_state.messages
        )
        # -----------------------------
        # Retrieve Chunks
        # -----------------------------
        results = retrieve_chunks(
            standalone_question
        )
        # -----------------------------
        # Generate Answer
        # -----------------------------
        answer, citations = generate_answer(
            results,
            standalone_question,
            st.session_state.messages
        )
        # -----------------------------
        # Save Assistant Response
        # -----------------------------
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        # -----------------------------
        # Display Assistant Response
        # -----------------------------
        with st.chat_message("assistant"):
            st.markdown(answer)
            if citations:
                st.markdown("### Sources")
                for cite in citations:
                    with st.expander(
                        f"{cite['source']} | Page {cite['page']}"
                    ):
                        st.write(
                            cite["excerpt"]
                        )