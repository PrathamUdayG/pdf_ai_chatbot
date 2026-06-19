"""
PDF AI ChatBot — main Streamlit application.

A production-grade RAG chatbot that lets users upload PDFs and ask
questions with streaming answers, hybrid search, and source citations.
"""

from __future__ import annotations

import streamlit as st

from config import get_api_key
from core.pdf_processor import extract_documents, chunk_documents
from core.vector_store import build_vector_store
from core.retriever import build_hybrid_retriever
from core.chain import (
    get_llm,
    build_contextualise_chain,
    build_answer_chain,
    format_documents,
    extract_citations,
    history_to_langchain,
)

# ─────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF AI ChatBot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Custom CSS — premium dark UI
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 24px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1419 0%, #1A1F2E 100%);
        border-right: 1px solid rgba(108,99,255,.15);
    }

    /* Stat cards */
    .stat-row { display: flex; gap: 10px; margin: 10px 0 16px; }
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(108,99,255,.12), rgba(59,130,246,.08));
        border: 1px solid rgba(108,99,255,.2);
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .stat-label {
        font-size: .72rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: .8px;
    }

    /* File badges */
    .file-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(108,99,255,.12);
        border: 1px solid rgba(108,99,255,.25);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: .82rem;
        color: #A5B4FC;
        margin: 3px 2px;
    }

    /* Citation card */
    .cite-card {
        background: rgba(108,99,255,.08);
        border-left: 3px solid #6C63FF;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin: 6px 0;
    }
    .cite-header {
        color: #6C63FF;
        font-weight: 600;
        font-size: .84rem;
        margin-bottom: 4px;
    }
    .cite-text {
        color: #CBD5E1;
        font-size: .82rem;
        line-height: 1.55;
    }

    /* Gradient divider */
    .gdiv {
        height: 2px;
        background: linear-gradient(90deg, transparent, #6C63FF, transparent);
        margin: 14px 0;
        border: none;
    }

    /* Button overrides */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px;
        font-weight: 600;
        transition: transform .2s, box-shadow .2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108,99,255,.35);
    }

    /* Status badge */
    .status-ready {
        color: #22C55E; font-weight: 600;
    }
    .status-not-ready {
        color: #F59E0B; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# API key guard
# ─────────────────────────────────────────────────────────────────
api_key = get_api_key()
if not api_key:
    st.error("❌ **GOOGLE_API_KEY** not found. Set it in `.env` or Streamlit Secrets.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "messages": [],
    "processed": False,
    "file_names": [],
    "total_chunks": 0,
    "total_pages": 0,
    "vector_store": None,
    "chunks": None,
    "retriever": None,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header" style="font-size:1.5rem;">📄 PDF AI ChatBot</p>', unsafe_allow_html=True)
    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDFs (up to 50 MB each)",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    # Process button
    if uploaded_files:
        if st.button("⚡ Process PDFs", use_container_width=True):
            with st.spinner("Extracting text & building index…"):
                docs = extract_documents(uploaded_files)
                chunks = chunk_documents(docs)
                vs = build_vector_store(chunks, api_key)
                retriever = build_hybrid_retriever(vs, chunks)

                st.session_state.vector_store = vs
                st.session_state.chunks = chunks
                st.session_state.retriever = retriever
                st.session_state.processed = True
                st.session_state.total_chunks = len(chunks)
                st.session_state.total_pages = sum(
                    1 for d in docs
                )
                st.session_state.file_names = [f.name for f in uploaded_files]
                # Clear chat on new upload
                st.session_state.messages = []
            st.success("✅ PDFs indexed — start chatting!")

    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)

    # Stats
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-value">{len(st.session_state.file_names)}</div>
                <div class="stat-label">PDFs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{st.session_state.total_pages}</div>
                <div class="stat-label">Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{st.session_state.total_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File list
    if st.session_state.file_names:
        st.markdown("**Loaded files**")
        for fname in st.session_state.file_names:
            st.markdown(f'<span class="file-badge">📎 {fname}</span>', unsafe_allow_html=True)

    # Status
    if st.session_state.processed:
        st.markdown('<p class="status-ready">● Ready</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-not-ready">● Not Ready</p>', unsafe_allow_html=True)

    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)

    # Reset
    if st.session_state.processed:
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            for k in _DEFAULTS:
                st.session_state[k] = _DEFAULTS[k]
            st.rerun()

    # Powered by
    st.caption("Powered by LangChain · Gemini · ChromaDB")

# ─────────────────────────────────────────────────────────────────
# Main area — header
# ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">PDF AI ChatBot</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload PDFs and ask questions — answers are grounded in your documents with source citations.</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# Chat history display
# ─────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show saved citations
        if msg["role"] == "assistant" and msg.get("citations"):
            st.markdown("**📚 Sources**")
            for cite in msg["citations"]:
                st.markdown(
                    f"""<div class="cite-card">
                        <div class="cite-header">📄 {cite['source']} — Page {cite['page']}</div>
                        <div class="cite-text">{cite['excerpt']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────
# Chat input & RAG pipeline
# ─────────────────────────────────────────────────────────────────
if not st.session_state.processed:
    st.info("👈 Upload one or more PDFs from the sidebar and click **Process PDFs** to begin.")
else:
    if question := st.chat_input("Ask something about your PDFs…"):
        # ── Store & display user message ──────────────────────
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # ── Build LangChain objects ───────────────────────────
        llm = get_llm(api_key)
        chat_history = history_to_langchain(st.session_state.messages[:-1])

        # ── Contextualise question ────────────────────────────
        if chat_history:
            ctx_chain = build_contextualise_chain(llm)
            standalone_q = ctx_chain.invoke(
                {"input": question, "chat_history": chat_history}
            )
        else:
            standalone_q = question

        # ── Retrieve relevant chunks ──────────────────────────
        retriever = st.session_state.retriever
        retrieved_docs = retriever.invoke(standalone_q)
        context = format_documents(retrieved_docs)
        citations = extract_citations(retrieved_docs)

        # ── Stream answer ─────────────────────────────────────
        answer_chain = build_answer_chain(llm)
        with st.chat_message("assistant"):
            streamed = st.write_stream(
                answer_chain.stream(
                    {
                        "context": context,
                        "input": question,
                        "chat_history": chat_history,
                    }
                )
            )

            # ── Show citations ────────────────────────────────
            if citations:
                st.markdown("**📚 Sources**")
                for cite in citations:
                    st.markdown(
                        f"""<div class="cite-card">
                            <div class="cite-header">📄 {cite['source']} — Page {cite['page']}</div>
                            <div class="cite-text">{cite['excerpt']}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        # ── Persist assistant message ─────────────────────────
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": streamed,
                "citations": citations,
            }
        )