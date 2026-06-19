"""
Hybrid retriever: BM25 (keyword) + ChromaDB (vector) with manual fusion.

Retrieval Approach
──────────────────
* **Vector search** (weight 0.7): semantic similarity via Google embeddings.
* **BM25 search** (weight 0.3): exact keyword matching for precision.
* Results are merged using Reciprocal Rank Fusion (RRF).
* Returns top-*k* most relevant chunks (default k = 5).
* Includes retry logic with exponential backoff for API rate limits.
"""

from __future__ import annotations

import time
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import TOP_K, BM25_WEIGHT, VECTOR_WEIGHT

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine BM25 keyword search with vector similarity search using RRF."""

    def __init__(
        self,
        vector_store: Chroma,
        chunks: list[Document],
        k: int = TOP_K,
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
    ):
        self.vector_store = vector_store
        self.chunks = chunks
        self.k = k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Build BM25 index
        tokenised = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenised)

    # ── Public API (same interface as LangChain retrievers) ───
    def invoke(self, query: str) -> list[Document]:
        """Retrieve top-k documents using hybrid search."""
        # Vector results (with retry for rate limits)
        vector_docs = self._vector_search_with_retry(query)

        # BM25 results (local, no API call)
        bm25_docs = self._bm25_search(query)

        # Reciprocal Rank Fusion
        return self._rrf_merge(bm25_docs, vector_docs)

    # ── Vector search with retry ──────────────────────────────
    def _vector_search_with_retry(
        self, query: str, max_retries: int = 3
    ) -> list[Document]:
        """Similarity search with exponential backoff for API rate limits."""
        for attempt in range(max_retries):
            try:
                return self.vector_store.similarity_search(query, k=self.k)
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = any(
                    kw in error_str
                    for kw in ("429", "rate", "quota", "resource_exhausted")
                )
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    logger.warning(
                        "Embedding rate-limited (attempt %d/%d). "
                        "Retrying in %ds…",
                        attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    raise
        return []  # unreachable, but keeps type checker happy

    # ── BM25 search (local, no API) ───────────────────────────
    def _bm25_search(self, query: str) -> list[Document]:
        """Keyword search via BM25 — runs locally, never rate-limited."""
        tokenised_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenised_query)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[: self.k]
        return [self.chunks[i] for i in top_indices]

    # ── Reciprocal Rank Fusion ────────────────────────────────
    def _rrf_merge(
        self,
        bm25_docs: list[Document],
        vector_docs: list[Document],
        rrf_k: int = 60,
    ) -> list[Document]:
        """Merge two ranked lists using RRF scoring."""
        scores: dict[int, float] = {}
        doc_map: dict[int, Document] = {}

        for rank, doc in enumerate(bm25_docs):
            doc_id = id(doc)
            matched_id = self._find_match_id(doc, doc_map)
            if matched_id is not None:
                doc_id = matched_id
            scores[doc_id] = scores.get(doc_id, 0) + self.bm25_weight / (rrf_k + rank + 1)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(vector_docs):
            matched_id = self._find_match_id(doc, doc_map)
            if matched_id is not None:
                doc_id = matched_id
            else:
                doc_id = id(doc)
            scores[doc_id] = scores.get(doc_id, 0) + self.vector_weight / (rrf_k + rank + 1)
            doc_map[doc_id] = doc

        sorted_ids = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        return [doc_map[did] for did in sorted_ids[: self.k]]

    @staticmethod
    def _find_match_id(doc: Document, doc_map: dict[int, Document]) -> int | None:
        """Find an existing doc in the map by content match (dedup)."""
        for did, existing in doc_map.items():
            if existing.page_content == doc.page_content:
                return did
        return None


def build_hybrid_retriever(
    vector_store: Chroma,
    chunks: list[Document],
    k: int = TOP_K,
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
) -> HybridRetriever:
    """Factory function to build a hybrid retriever."""
    return HybridRetriever(
        vector_store=vector_store,
        chunks=chunks,
        k=k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )
