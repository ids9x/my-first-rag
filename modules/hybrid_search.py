"""
Hybrid Search (Priority 3)

Combines two retrieval methods:
  1. Vector similarity  — semantic meaning (good for paraphrased questions)
  2. BM25 keyword search — exact term matching (good for clause numbers,
     specific terminology like "Section 2.7.3" or "QA Level 1")

Results are fused using Reciprocal Rank Fusion (RRF), which merges
ranked lists without needing score normalization.

Why this matters for nuclear docs:
  - Vector search finds "quality assurance requirements" when you ask
    about "QA standards"
  - BM25 finds the exact "Section 18" when you ask about "NQA-1 Section 18"
  - Together they cover both semantic and lexical retrieval
"""
import pickle
from pathlib import Path
from typing import TYPE_CHECKING
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from config.settings import (
    RETRIEVER_K,
    HYBRID_VECTOR_WEIGHT,
    HYBRID_BM25_WEIGHT,
    BM25_INDEX_PATH,
)

if TYPE_CHECKING:
    from modules.reranking import Reranker


class BM25Index:
    """
    BM25 keyword index over document chunks.
    Can be saved/loaded to avoid rebuilding.
    """

    def __init__(self, documents: list[Document] | None = None):
        self.documents = documents or []
        self._index: BM25Okapi | None = None

        if self.documents:
            self._build()

    def _build(self):
        """Build the BM25 index from document texts."""
        tokenized = [doc.page_content.lower().split() for doc in self.documents]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, k: int = RETRIEVER_K) -> list[Document]:
        """Return top-k documents by BM25 score."""
        if self._index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [self.documents[i] for i in top_indices if scores[i] > 0]

    def save(self, path: str | Path = BM25_INDEX_PATH):
        """Persist the index to disk."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({"documents": self.documents}, f)
        print(f"💾 BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str | Path = BM25_INDEX_PATH) -> "BM25Index":
        """Load a persisted BM25 index."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index at {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(documents=data["documents"])
        print(f"📂 BM25 index loaded ({len(instance.documents)} documents)")
        return instance


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float] | None = None,
    k_constant: int = 60,
) -> list[Document]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum( weight_i / (k + rank_i(d)) )

    This avoids the need to normalize scores across different retrieval methods.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    # Score each document
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, doc in enumerate(ranked_list):
            # Use content hash as key (handles duplicates)
            key = hash(doc.page_content)
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + weight / (k_constant + rank + 1)

    # Sort by fused score
    sorted_keys = sorted(doc_scores, key=lambda k: doc_scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys]


class HybridRetriever(BaseRetriever):
    """
    LangChain-compatible retriever that fuses vector + BM25 results.

    Usage:
        from core.store import VectorStoreManager
        from modules.hybrid_search import HybridRetriever, BM25Index

        manager = VectorStoreManager()
        bm25 = BM25Index.load()

        retriever = HybridRetriever(
            vector_retriever=manager.get_retriever(k=6),
            bm25_index=bm25,
        )
        docs = retriever.invoke("NQA-1 Section 18 requirements")
    """

    vector_retriever: BaseRetriever
    bm25_index: BM25Index
    k: int = RETRIEVER_K
    vector_weight: float = HYBRID_VECTOR_WEIGHT
    bm25_weight: float = HYBRID_BM25_WEIGHT
    reranker: "Reranker | None" = None  # Optional reranker

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Retrieve using both methods, then fuse, then optionally rerank."""
        # Get results from both retrievers (fetch more than k, then fuse to k)
        fetch_k = self.k * 2

        vector_results = self.vector_retriever.invoke(query)[:fetch_k]
        bm25_results = self.bm25_index.search(query, k=fetch_k)

        # Fuse with RRF
        fused = reciprocal_rank_fusion(
            ranked_lists=[vector_results, bm25_results],
            weights=[self.vector_weight, self.bm25_weight],
        )

        # Apply reranking if enabled
        if self.reranker:
            return self.reranker.rerank(query, fused, top_k=self.k)

        return fused[: self.k]


# Rebuild the model to resolve forward references
# This is needed because Reranker is defined in a separate module
try:
    from modules.reranking import Reranker
    HybridRetriever.model_rebuild()
except ImportError:
    pass  # Reranker not available, but HybridRetriever will still work without it
