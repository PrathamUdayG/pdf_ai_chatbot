"""
Vector store management using ChromaDB + Google Generative AI Embeddings.

Embedding Model Choice
──────────────────────
* ``models/embedding-001`` from Google — free tier, 768-dim vectors.
* Accessed via ``langchain-google-genai`` so no local GPU / torch needed.
* ChromaDB runs **in-memory** for Streamlit Cloud compatibility.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import EMBEDDING_MODEL


def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """Return the Google embedding function."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def build_vector_store(chunks, api_key: str) -> Chroma:
    """Create an in-memory ChromaDB vector store from document chunks."""
    embeddings = get_embeddings(api_key)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_collection",
    )
    return vector_store
