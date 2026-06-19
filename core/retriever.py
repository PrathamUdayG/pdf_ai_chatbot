"""
Hybrid retriever: BM25 (keyword) + ChromaDB (vector) with manual fusion.

Retrieval Approach
──────────────────
* **Vector search** (weight 0.7): semantic similarity via Google embeddings.
* **BM25 search** (weight 0.3): exact keyword matching for precision.
* Results are merged using Reciprocal Rank Fusion (RRF).
* Returns top-*k* most relevant chunks (default k = 5).
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import TOP_K, BM25_WEIGHT, VECTOR_WEIGHT


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
        # Vector results
        vector_docs = self.vector_store.similarity_search(query, k=self.k)

        # BM25 results
        tokenised_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenised_query)
        top_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[: self.k]
        bm25_docs = [self.chunks[i] for i in top_bm25_indices]

        # Reciprocal Rank Fusion
        return self._rrf_merge(bm25_docs, vector_docs)

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
            # Try matching by content for deduplication
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
