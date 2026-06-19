"""
Centralised configuration for PDF AI ChatBot.
All tuneable parameters live here so they are easy to find and change.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── API Key (local .env → Streamlit Secrets) ──────────────────────
def get_api_key() -> str | None:
    """Return the Google API key from env or Streamlit secrets."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError):
            return None
    return key

# ── Model settings ────────────────────────────────────────────────
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"

# ── Chunking ──────────────────────────────────────────────────────
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

# ── Retrieval ─────────────────────────────────────────────────────
TOP_K = 5
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7

# ── Upload ────────────────────────────────────────────────────────
MAX_UPLOAD_MB = 50
