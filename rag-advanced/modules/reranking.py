"""
Cross-Encoder Reranking (Priority 7)

Reranks retrieved documents using a cross-encoder model that evaluates
query-document pairs jointly. Unlike bi-encoders (embeddings), cross-encoders
provide superior precision by considering the full interaction between query and document.

Key benefits for nuclear docs:
  - Better distinguishes similar sections (Section 18.1 vs 18.2)
  - Improves relevance of technical terminology matches
  - Reduces false positives from semantic similarity
  - Supports fine-tuning on domain-specific nuclear terminology
"""
import torch
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from config.settings import (
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_USE_FP16,
    RERANKER_BATCH_SIZE,
    RERANKER_TOP_K,
    RERANKER_DEVICE,
    RERANKER_NORMALIZE_SCORES,
    RERANKER_CUSTOM_MODEL_PATH,
)


class Reranker:
    """
    Wrapper around FlagReranker with optimizations.

    Features:
    - Lazy model loading (only load when first used)
    - Batch processing for efficiency
    - Score normalization (sigmoid)
    - Support for custom fine-tuned models
    - GPU/CPU device management with auto-fallback
    """

    # Singleton pattern: load model once, reuse across queries
    _instance = None

    def __init__(self):
        """Initialize reranker (model loaded lazily on first use)."""
        self._model = None
        self._device = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of Reranker."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Lazy load the reranker model."""
        if self._model is not None:
            return  # Already loaded

        # Determine device (auto-fallback from GPU to CPU)
        if RERANKER_DEVICE == "cuda" and torch.cuda.is_available():
            self._device = "cuda"
            print("🚀 Loading reranker on GPU...")
        else:
            self._device = "cpu"
            if RERANKER_DEVICE == "cuda":
                print("⚠️  CUDA not available, falling back to CPU...")
            else:
                print("💻 Loading reranker on CPU...")

        # Load custom model if specified, else default
        model_name = RERANKER_CUSTOM_MODEL_PATH or RERANKER_MODEL

        # Initialize FlagReranker
        self._model = FlagReranker(
            model_name,
            use_fp16=RERANKER_USE_FP16 and self._device == "cuda",  # fp16 only on GPU
            device=self._device,
        )

        print(f"✅ Reranker loaded: {model_name}")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = RERANKER_TOP_K,
    ) -> list[Document]:
        """
        Rerank documents by query-document relevance scores.

        Args:
            query: User query
            documents: Retrieved documents to rerank
            top_k: Number of top documents to return

        Returns:
            Top-k documents sorted by reranker score, with scores added to metadata
        """
        if not documents:
            return []

        # Lazy load model if not loaded
        self._load_model()

        # Create query-document pairs for scoring
        pairs = [[query, doc.page_content] for doc in documents]

        # Compute scores in batches
        try:
            scores = self._model.compute_score(
                pairs,
                batch_size=RERANKER_BATCH_SIZE,
                normalize=RERANKER_NORMALIZE_SCORES,
            )
        except Exception as e:
            print(f"⚠️  Reranking failed: {e}")
            print("   Returning original documents without reranking")
            return documents[:top_k]

        # Handle single document case (compute_score returns float instead of list)
        if isinstance(scores, float):
            scores = [scores]

        # Create list of (document, score) tuples
        doc_score_pairs = list(zip(documents, scores))

        # Sort by score descending
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Add reranker scores to metadata and return top-k
        reranked_docs = []
        for doc, score in doc_score_pairs[:top_k]:
            # Create new document with updated metadata
            new_metadata = doc.metadata.copy()
            new_metadata["reranker_score"] = float(score)

            reranked_doc = Document(
                page_content=doc.page_content,
                metadata=new_metadata,
            )
            reranked_docs.append(reranked_doc)

        return reranked_docs


class RerankingRetriever(BaseRetriever):
    """
    LangChain-compatible retriever wrapper with reranking.

    Usage:
        base_retriever = manager.get_retriever(k=8)
        reranker = Reranker.get_instance()
        reranked = RerankingRetriever(
            base_retriever=base_retriever,
            reranker=reranker,
            top_k=4
        )
        docs = reranked.invoke("What are NQA-1 requirements?")
    """

    base_retriever: BaseRetriever
    reranker: Reranker
    top_k: int = RERANKER_TOP_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Retrieve documents from base retriever and rerank.

        Args:
            query: User query
            run_manager: Callback manager (unused but required by interface)

        Returns:
            Top-k reranked documents
        """
        # Get initial results from base retriever
        initial_docs = self.base_retriever.invoke(query)

        # Apply reranking
        reranked_docs = self.reranker.rerank(query, initial_docs, top_k=self.top_k)

        return reranked_docs


def get_reranker() -> Reranker | None:
    """
    Factory function to create reranker based on config.

    Returns:
        Reranker instance if enabled, None otherwise
    """
    if not RERANKER_ENABLED:
        return None
    return Reranker.get_instance()


# ── Fine-tuning utilities (future) ─────────────────────────────────

def create_training_data(
    queries: list[str],
    relevant_docs: list[list[Document]],
    irrelevant_docs: list[list[Document]],
) -> list[dict]:
    """
    Create training data for fine-tuning BGE reranker.

    Format: [{"query": str, "pos": [str, ...], "neg": [str, ...]}, ...]

    Example for nuclear docs:
        Query: "What are QA Level 1 requirements?"
        Positive: NQA-1 Section 18.1 content (relevant)
        Negative: Similar but irrelevant sections

    Args:
        queries: List of query strings
        relevant_docs: List of lists of relevant documents for each query
        irrelevant_docs: List of lists of irrelevant documents for each query

    Returns:
        Training data in BGE format
    """
    training_data = []

    for query, pos_docs, neg_docs in zip(queries, relevant_docs, irrelevant_docs):
        training_data.append({
            "query": query,
            "pos": [doc.page_content for doc in pos_docs],
            "neg": [doc.page_content for doc in neg_docs],
        })

    return training_data


def export_training_data(data: list[dict], output_path: str):
    """
    Export training data in JSONL format for fine-tuning.

    Args:
        data: Training data from create_training_data()
        output_path: Path to save JSONL file
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"💾 Training data exported to {output_path}")
    print(f"   {len(data)} training examples")
