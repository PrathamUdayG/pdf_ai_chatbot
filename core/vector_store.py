"""
Vector store management using ChromaDB + Google Generative AI Embeddings.

Embedding Model Choice
──────────────────────
* ``models/gemini-embedding-001`` from Google — free tier, 3072-dim vectors.
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


import time
import logging

logger = logging.getLogger(__name__)

def build_vector_store(chunks, api_key: str) -> Chroma:
    """Create an in-memory ChromaDB vector store from document chunks with rate limit handling."""
    embeddings = get_embeddings(api_key)
    
    # Process in small batches to respect free tier limits
    batch_size = 10
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
    )
    
    total_batches = (len(chunks) - 1) // batch_size + 1
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Adding batch {batch_num}/{total_batches}")
        
        # Retry with exponential backoff for transient API errors
        max_retries = 5
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(
                    kw in error_str
                    for kw in ("429", "rate", "quota", "resource_exhausted", "500", "internal")
                )
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 2)  # 4s, 8s, 16s, 32s
                    logger.warning(
                        "Google API error on batch %d (attempt %d/%d). "
                        "Retrying in %ds… Error: %s",
                        batch_num, attempt + 1, max_retries, wait,
                        str(e)[:120],
                    )
                    time.sleep(wait)
                else:
                    raise
        
        # Cooldown between successful batches to avoid hitting RPM limits
        if i + batch_size < len(chunks):
            time.sleep(3)
            
    return vector_store
